import numpy as np

from collections import defaultdict
from fve_layer.backends.chainer.links import GMMLayer
import chainer
import logging

from chainer import functions as F

from moth_classifier.core import dataset

class SizeMixin:
	def __init__(self, *args,
				 use_size_model: bool,
				 loss_alpha: float = 0.5,
				 **kwargs):
		super().__init__(*args, **kwargs)

		self._use_size_model = use_size_model

		with self.init_scope():
			# use 1.0 as default setting for the loss scaling
			self.add_persistent("loss_alpha", loss_alpha)

		if not self._use_size_model:
			return

		n_classes = 69# self.n_classes
		logging.info(f"Initializing size model for {n_classes} classes ({loss_alpha=})")

		with self.init_scope():
			self._size_model = SizeModel(n_classes)#self.n_classes)


	def fit_size_model(self, ds: dataset.Dataset):
		if not self._use_size_model:
			return

		arr = self.xp.array

		if isinstance(ds, chainer.datasets.SubDataset):
			ds = ds._dataset

		self._size_model.fit(arr(ds.sizes), arr(ds.labels))


	def size_model(self, sizes: np.ndarray, pred: chainer.Variable, y):

		if not self._use_size_model or sizes is None:
			return pred

		# (batch_size, n_features, feature_size) is expected
		sizes = self.xp.array(sizes).reshape(-1, 1, 1)
		size_log_probs = self._size_model.log_soft_assignment(sizes)[:, 0, :]

		log_pred = F.log_softmax(pred)
		log_cls_weights = self.xp.log(self._size_model.w)

		self.eval_prediction(pred, y, suffix="0")

		self.report(
			#accu0=self.model.accuracy(pred, y),
			accu_s=self.model.accuracy(size_log_probs, y),
		)

		return log_pred + size_log_probs# - log_cls_weights




class SizeModel(GMMLayer):
	def __init__(self, n_classes, init_mu=0, init_sig=1, dtype=np.float32):
		super().__init__(
			in_size=1,
			n_components=n_classes,
			init_mu=init_mu,
			init_sig=init_sig)

	def fit(self, sizes, labels):

		sizes_per_class = defaultdict(list)
		for size, lab in zip(sizes, labels):
			sizes_per_class[int(lab)].append(size)

		n_classes = len(sizes_per_class)
		for cls_id, cls_sizes in sizes_per_class.items():
			cls_sizes = self.xp.array(cls_sizes)
			self.mu[:, cls_id] = self.xp.mean(cls_sizes)
			self.sig[:, cls_id] = self.xp.var(cls_sizes)
			self.w[cls_id] = len(cls_sizes) / n_classes

		no_cov = self.sig == 0
		self.sig[no_cov] = self.mu[no_cov] * 0.1
		self.sig[:] = np.maximum(self.sig, self.eps)

		self.w[:] = np.maximum(self.w, self.eps)
		self.w[:] /= self.w.sum()
