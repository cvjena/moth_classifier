import chainer
import numpy as np


def _to_cpu(var):
	return chainer.cuda.to_cpu(chainer.as_array(var))

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
