import abc
import chainer
import numpy as np
import chainer.functions as F

from cvfinetune import classifier
from typing import Dict
from typing import Callable


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def _mean(arrays):
	return F.mean(F.stack(arrays, axis=0), axis=0)

def get_params(opts):
	kwargs = dict(only_head=opts.only_head)
	if opts.parts == "GLOBAL":
		cls = GlobalClassifier

	else:
		cls = PartsClassifier
		kwargs["concat_features"] = opts.concat_features

	return dict(classifier_cls=cls, classifier_kwargs=kwargs)

class OnlyHeadMixin(abc.ABC):

	def __init__(self, only_head, *args, **kwargs):
		super(OnlyHeadMixin, self).__init__(*args, **kwargs)
		self._only_head = only_head

	def _get_features(self, X, model):
		if self._only_head:
			with chainer.using_config("train", False), chainer.no_backprop_mode():
				features = model(X, layer_name=model.meta.feature_layer)
		else:
			features = model(X, layer_name=model.meta.feature_layer)

		return _unpack(features)

def eval_prediction(pred, gt, evaluations: Dict[str, Callable], reporter):
	for metric, func in evaluations.items():
		reporter(**{metric: func(pred, gt)})

def f1_score(pred, gt):
	score, support = F.f1_score(pred, gt)
	xp = score.device.xp
	return xp.nanmean(score.array)

class GlobalClassifier(OnlyHeadMixin, classifier.Classifier):

	def __call__(self, X, y):
		feat = self._get_features(X, self.model)
		pred = self.model.fc(feat)

		eval_prediction(pred, y,
			evaluations=dict(
				accuracy=self.model.accuracy,
				f1=f1_score,
			),
			reporter=self.report)

		loss = self.loss(pred, y)
		self.report(loss=loss)

		return loss


class PartsClassifier(OnlyHeadMixin, classifier.SeparateModelClassifier):
	n_parts = 4

	def __init__(self, concat_features, *args, **kwargs):
		super(PartsClassifier, self).__init__(*args, **kwargs)
		self._concat = concat_features


	@property
	def output_size(self):
		if self._concat:
			return self.n_parts * self.feat_size

		return self.feat_size

	def _encode_parts(self, feats):
		if self._concat:
			# concat all features together
			n, t, feat_size = feats.shape
			return F.reshape(feats, (n, t*feat_size))

		# average over the t-dimension
		return F.mean(feats, axis=1)

	def __call__(self, X, parts, y):
		assert X.ndim == 4 and parts.ndim == 5 , \
			f"Dimensionality of inputs was incorrect ({X.ndim=}, {parts.ndim=})!"
		glob_feat = self._get_features(X, self.separate_model)
		glob_pred = self.separate_model.fc(glob_feat)

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat = self._get_features(part, self.model)
			part_feats.append(part_feat)

		# stack over the t-dimension
		part_feats = F.stack(part_feats, axis=1)
		part_feats = self._encode_parts(part_feats)
		part_pred = self.model.fc(part_feats)

		glob_loss, glob_accu = self.loss(glob_pred, y), self.separate_model.accuracy(glob_pred, y)
		part_loss, part_accu = self.loss(part_pred, y), self.model.accuracy(part_pred, y)

		_mean_pred = _mean([F.softmax(glob_pred), F.softmax(part_pred)])

		eval_prediction(_mean_pred, y,
			evaluations=dict(
				accuracy=self.model.accuracy,
				f1=f1_score,
			),
			reporter=self.report)

		loss = _mean([part_loss, glob_loss])

		self.report(
			loss=loss,
			p_accu=part_accu,
			g_accu=glob_accu,
		)

		return loss
