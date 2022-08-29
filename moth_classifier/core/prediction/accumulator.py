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
