import chainer
import torch

from chainer import function_node
from chainer import links as L
from chainer import functions as F
from chainer.backends.cuda import to_cpu
from pytorch_metric_learning.miners import TripletMarginMiner

from moth_classifier.core.classifier.base import Classifier


def as_torch_tensor(var: chainer.Variable):
	if var is None:
		return None

	return torch.from_numpy(to_cpu(chainer.as_array(var)))

class CustomMiner(TripletMarginMiner):


	def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):

		embeddings = as_torch_tensor(embeddings)
		labels = as_torch_tensor(labels)
		ref_emb = as_torch_tensor(ref_emb)
		ref_labels = as_torch_tensor(ref_labels)

		result = super().forward(embeddings, labels, ref_emb, ref_labels)

		return [tensor.cpu().numpy() for tensor in result]


def reciprocal_triplet_loss(anch, pos, neg, reduce_method="mean"):
	assert reduce_method in ["mean", "no"], \
		f"Reduce method should be either \"mean\" or \"no\", but was {reduce_method}"

	ap_dist = F.sum((anch - pos) ** 2, axis=1)
	an_dist = F.sum((anch - neg) ** 2, axis=1)

	losses = ap_dist + (1 / an_dist)

	if reduce_method == "no":
		return losses

	return F.mean(losses)

class TripletClassifier(Classifier):

	def __init__(self, *args,
				 margin: float = 0.1,
				 embedding_size: int = 128,
				 alpha: float = 0.01,
				 **kwargs):

		super().__init__(*args, **kwargs)
		self._emb_size = embedding_size if embedding_size > 0 else self.feat_size

		with self.init_scope():
			self.embedding = L.Linear(self._emb_size)

		self._alpha = alpha

		self.triplet_miner = CustomMiner(
			margin=margin, type_of_triplets="easy"
		)

	@property
	def output_size(self):
		return self._emb_size

	def extract(self, X):
		feat = self._get_features(X, self.model)
		return self.embedding(feat)


	def forward(self, X, y, sizes=None):
		emb = self.extract(X)
		pred = self.model.clf_layer(emb)

		ce_loss = self.loss(pred, y)
		self.eval_prediction(pred, y, suffix="")

		if self._alpha <= 0:
			self.report(loss=ce_loss)
			return ce_loss

		a_idx, p_idx, n_idx = self.triplet_miner(emb, y)
		anch, pos, neg = emb[a_idx], emb[p_idx], emb[n_idx]
		triplet_loss = reciprocal_triplet_loss(anch, pos, neg)

		loss = ce_loss + self._alpha * triplet_loss
		self.report(
			loss=loss,
			ce_loss=ce_loss,
			triplet_loss=triplet_loss,
		)


		return loss






