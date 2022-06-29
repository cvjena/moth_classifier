import abc
import chainer
import typing as T
import numpy as np
import chainer.functions as F

from cvmodelz import classifiers

from moth_classifier.core.classifier.prediction import Prediction


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def eval_prediction(pred, gt, evaluations: T.Dict[str, T.Callable], reporter):
	for metric, func in evaluations.items():
		reporter(**{metric: func(pred, gt)})

class BaseClassifier(abc.ABC):

	def __init__(self, only_head, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._only_head = only_head


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
				accuracy=self.model.accuracy,
				prec=Prediction.precision,
				rec=Prediction.recall,
				f1=Prediction.f1_score,
				# f1=f1_score,
			),
			reporter=self.report)

class Classifier(BaseClassifier, classifiers.Classifier):

	def forward(self, X, y, size=None):
		feat = self._get_features(X, self.model)
		pred = self.model.clf_layer(feat)

		self.eval_prediction(pred, y)
		loss = self.loss(pred, y)
		self.report(loss=loss)

		return loss

