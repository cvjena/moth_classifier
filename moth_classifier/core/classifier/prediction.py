import abc
import chainer
import numpy as np


def _to_cpu(var, dtype=None):
	res = chainer.cuda.to_cpu(chainer.as_array(var))
	if dtype is None:
		return res
	return res.astype(dtype)

class PredictionAccumulator:

	def __init__(self, logits = None, gt = None):
		super().__init__()

		self._logits = [] if logits is None else [_to_cpu(logits, np.float16)]
		self._gt = [] if gt is None else [_to_cpu(gt, np.int32)]

	def update(self, logits, gt):

		self._metrics = None

		if isinstance(logits, list):
			self._logits += logits
		else:
			self._logits += [_to_cpu(logits, np.float16)]

		if isinstance(gt, list):
			self._gt += gt
		else:
			self._gt += [_to_cpu(gt, np.int32)]

	def calc_metrics(self, *, only_available: bool = True, beta: int = 1):
		if self._metrics is not None:
			return self._metrics

		logits, true = self.reset()
		n_cls = max(logits.shape[1], true.max())

		if only_available:
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
		beta_square = beta ** 2
		numerator = (1 + beta_square) * precision * recall
		denominator = beta_square * precision + recall
		mask = denominator != 0
		fbeta_score[mask] = numerator[mask] / denominator[mask]

		self._metrics = {
			"precision": np.nanmean(precision[count_mask]),
			"recall": np.nanmean(recall[count_mask]),
			f"f{beta}_score": np.nanmean(fbeta_score[count_mask])
		}
		return self._metrics


	def reset(self):
		logits, gt = np.vstack(self._logits), np.hstack(self._gt)
		self._logits, self._gt = [], []
		return logits, gt

class BaseMetric(abc.ABC):
	ndim = 0 # we need this to "fool" the reporter's summary

	def __add__(self, num):
		return self.add(num)

	def __radd__(self, num):
		return self.add(num)


	def __mul__(self, num):
		return self.mul(num)

	def __rmul__(self, num):
		return self.mul(num)

	def __floordiv__(self, num):
		return self.div(num)

	def __rfloordiv__(self, num):
		return self.div(num)

	def __truediv__(self, num):
		return self.div(num)

	def __rtruediv__(self, num):
		return self.div(num)


	def __init__(self, accumulator: PredictionAccumulator, *,
		only_available: bool = True):
		super().__init__()
		self._accum = accumulator
		self._only_available = only_available

	def __call__(self, *args, **kwargs):
		return self

	def mul(self, other):
		return self

	def add(self, other):
		return self


	@abc.abstractmethod
	def div(self, other, **kwargs):
		return self._accum.calc_metrics(only_available = self._only_available, **kwargs)

class Precision(BaseMetric):

	def div(self, other):
		metrics = super().div(other)
		return metrics.get("precision", 0)

class Recall(BaseMetric):

	def div(self, other):
		metrics = super().div(other)
		return metrics.get("recall", 0)

class FScore(BaseMetric):

	def __init__(self, *args, beta: int = 1, **kwargs):
		super().__init__(*args, **kwargs)
		self._beta = beta

	def div(self, other):
		metrics = super().div(other, beta=self._beta)
		return metrics.get(f"f{self._beta}_score", 0)

