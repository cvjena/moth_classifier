import abc
import chainer
import logging
import numpy as np
import typing as T

from chainer import functions as F
from cvmodelz import classifiers

from moth_classifier.core.classifier.prediction import Prediction
from moth_classifier.core.classifier.size_model import SizeModel
from moth_classifier.core.dataset import Dataset


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def eval_prediction(pred, gt, evaluations: T.Dict[str, T.Callable], reporter):
	for metric, func in evaluations.items():
		reporter(**{metric: func(pred, gt)})

class BaseClassifier(abc.ABC):

	def __init__(self, only_head: bool, use_size_model: bool, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._only_head = only_head
		self._use_size_model = use_size_model

		if not self._use_size_model:
			return

		n_classes = 69# self.n_classes
		logging.info(f"Initializing size model for {n_classes} classes")

		with self.init_scope():
			self._size_model = SizeModel(n_classes)#self.n_classes)


	def _get_features(self, X, model):
		if self._only_head:
			with chainer.using_config("train", False), chainer.no_backprop_mode():
				features = model(X, layer_name=model.meta.feature_layer)
		else:
			features = model(X, layer_name=model.meta.feature_layer)

		return _unpack(features)


	def eval_prediction(self, pred, gt):

		return eval_prediction(pred, gt,
			evaluations=dict(
				accu=self.model.accuracy,
				prec=Prediction.precision,
				rec=Prediction.recall,
				f1=Prediction.f1_score,
			),
			reporter=self.report)


	def fit_size_model(self, ds: Dataset):
		if not self._use_size_model:
			return

		arr = self.xp.array

		if isinstance(ds, chainer.datasets.SubDataset):
			ds = ds._dataset

		self._size_model.fit(arr(ds.sizes), arr(ds.labels))


	def size_model(self, sizes, pred: chainer.Variable, y):

		if not self._use_size_model or sizes is None:
			return pred

		# (batch_size, n_features, feature_size) is expected
		sizes = self.xp.array(sizes).reshape(-1, 1, 1)
		size_log_probs = self._size_model.log_soft_assignment(sizes)[:, 0, :]

		log_pred = F.log_softmax(pred)
		log_cls_weights = self.xp.log(self._size_model.w)

		self.report(
			accu0=self.model.accuracy(pred, y),
			accu_s=self.model.accuracy(size_log_probs, y),
		)

		return log_pred + size_log_probs - log_cls_weights


class Classifier(BaseClassifier, classifiers.Classifier):

	def forward(self, X, y, sizes=None):
		feat = self._get_features(X, self.model)
		pred = self.model.clf_layer(feat)

		pred = self.size_model(sizes, pred, y)

		self.eval_prediction(pred, y)
		loss = self.loss(pred, y)
		self.report(loss=loss)

		return loss
