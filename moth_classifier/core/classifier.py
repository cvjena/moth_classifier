import abc
import chainer
import numpy as np
import chainer.functions as F

from dataclasses import dataclass

from cvmodelz import classifiers
from typing import Dict
from typing import Callable


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def _mean(arrays):
	return F.mean(F.stack(arrays, axis=0), axis=0)

def _to_cpu(var):
	return chainer.cuda.to_cpu(chainer.as_array(var))

def get_params(opts):
	model_kwargs = dict(pooling=opts.pooling)

	if hasattr(opts, "n_classes"):
		model_kwargs["n_classes"] = opts.n_classes

	kwargs = dict(only_head=opts.only_head)
	if opts.parts == "GLOBAL":
		cls = GlobalClassifier

	else:
		cls = PartsClassifier
		kwargs["concat_features"] = opts.concat_features

	return dict(
		classifier_cls=cls,
		classifier_kwargs=kwargs,

		model_kwargs=model_kwargs,
	)

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

class BaseClassifier(OnlyHeadMixin):

	predictions = dict(train=[], val=[])

	def eval_prediction(self, pred, gt):
		entries = (_to_cpu(pred), _to_cpu(gt))
		if chainer.config.train:
			self.predictions["val"].clear()
			self.predictions["train"].append(entries)
		else:
			self.predictions["train"].clear()
			self.predictions["val"].append(entries)

		return eval_prediction(pred, gt,
			evaluations=dict(
				accuracy=self.model.accuracy,
				prec=Prediction.precision,
				rec=Prediction.recall,
				f1=Prediction.f1_score,
				# f1=f1_score,
			),
			reporter=self.report)

class GlobalClassifier(BaseClassifier, classifiers.Classifier):

	def forward(self, X, y):
		feat = self._get_features(X, self.model)
		pred = self.model.clf_layer(feat)

		self.eval_prediction(pred, y)
		loss = self.loss(pred, y)
		self.report(loss=loss)

		return loss


class PartsClassifier(BaseClassifier, classifiers.SeparateModelClassifier):
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

	def forward(self, X, parts, y):
		assert X.ndim == 4 and parts.ndim == 5 , \
			f"Dimensionality of inputs was incorrect ({X.ndim=}, {parts.ndim=})!"
		glob_feat = self._get_features(X, self.separate_model)
		glob_pred = self.separate_model.clf_layer(glob_feat)

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat = self._get_features(part, self.model)
			part_feats.append(part_feat)

		# stack over the t-dimension
		part_feats = F.stack(part_feats, axis=1)
		part_feats = self._encode_parts(part_feats)
		part_pred = self.model.clf_layer(part_feats)

		glob_loss, glob_accu = self.loss(glob_pred, y), self.separate_model.accuracy(glob_pred, y)
		part_loss, part_accu = self.loss(part_pred, y), self.model.accuracy(part_pred, y)

		_mean_pred = _mean([F.softmax(glob_pred), F.softmax(part_pred)])

		self.eval_prediction(_mean_pred, y)

		loss = _mean([part_loss, glob_loss])

		self.report(
			loss=loss,
			p_accu=part_accu,
			g_accu=glob_accu,
		)

		return loss


class Prediction:
	# we just need to "fool" the reporters summary
	ndim = 0
	only_available = True
	beta = 1

	@classmethod
	def precision(cls, logits, gt):
		return cls(logits, gt, metric="precision")

	@classmethod
	def recall(cls, logits, gt):
		return cls(logits, gt, metric="recall")

	@classmethod
	def f1_score(cls, logits, gt):
		return cls(logits, gt, metric="f1_score")

	def _calc_metric(self):
		logits, true = np.vstack(self.logits), np.hstack(self.gt)
		n_cls = max(logits.shape[1], true.max())

		if self.only_available:
			_logits = np.full_like(logits, fill_value=logits.min())
			available = np.unique(true)
			_logits[:, available] = logits[:, available]
			logits = _logits

		pred = logits.argmax(axis=1)

		counts = np.bincount(true, minlength=n_cls + 1)[:n_cls]
		relevant = np.bincount(pred, minlength=n_cls + 1)[:n_cls]

		tp_mask = np.where(pred == true, true, n_cls)
		tp = np.bincount(tp_mask, minlength=n_cls + 1)[:n_cls]


		precision = np.zeros_like(tp, dtype=np.float32)
		recall = np.zeros_like(tp, dtype=np.float32)
		fbeta_score = np.zeros_like(tp, dtype=np.float32)

		count_mask = counts != 0
		relev_mask = relevant != 0

		precision[relev_mask] = tp[relev_mask] / relevant[relev_mask]
		recall[count_mask] = tp[count_mask] / counts[count_mask]

		# F-Measure
		beta_square = self.beta ** 2
		numerator = (1 + beta_square) * precision * recall
		denominator = beta_square * precision + recall
		mask = denominator != 0
		fbeta_score[mask] = numerator[mask] / denominator[mask]

		metrics = {
			"precision": np.nanmean(precision[count_mask]),
			"recall": np.nanmean(recall[count_mask]),
			"f1_score": np.nanmean(fbeta_score[count_mask])
		}

		return metrics.get(self.metric, 0)

	def __init__(self, logits, gt, *, metric):
		super().__init__()

		if isinstance(logits, list):
			self.logits = logits
		else:
			self.logits = [_to_cpu(logits)]

		if isinstance(gt, list):
			self.gt = gt
		else:
			self.gt = [_to_cpu(gt)]
		self.metric = metric

	def _new(self, logits, gt):
		return type(self)(logits, gt, metric=self.metric)

	def __mul(self, other):
		if isinstance(other, (float, int)):
			logits = list(self.logits)
			logits[-1] *= other

		elif isinstance(other, Prediction):
			logits = list(self.logits)
			logits[-1] *= other.logits[-1]

		else:
			raise NotImplementedError

		return self._new(logits, self.gt)


	def __add(self, other):

		if isinstance(other, (float, int)):
			if other != 0:
				raise NotImplementedError
			return self

		elif isinstance(other, Prediction):
			logits = self.logits + other.logits
			gt = self.gt + other.gt

			return self._new(logits, gt)

		raise NotImplementedError


	def __div(self, other):

		if isinstance(other, (float, int)):
			return self._calc_metric()

		raise NotImplementedError


	def __add__(self, num):
		return self.__add(num)

	def __radd__(self, num):
		return self.__add(num)


	def __mul__(self, num):
		return self.__mul(num)

	def __rmul__(self, num):
		return self.__mul(num)

	def __floordiv__(self, num):
		return self.__div(num)

	def __rfloordiv__(self, num):
		return self.__div(num)

	def __truediv__(self, num):
		return self.__div(num)

	def __rtruediv__(self, num):
		return self.__div(num)


	def __sub__(self, other):
		import pdb; pdb.set_trace()

	def __rsub__(self, other):
		import pdb; pdb.set_trace()

	def __pow__(self, other):
		import pdb; pdb.set_trace()
