import chainer
import numpy as np

from chainer import functions as F
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer.training import trigger as trigger_module


class EpochSummary(extension.Extension):

	name = "EpochSummary"
	trigger = 1, "iteration"

	# should be called as the first
	priority = extension.PRIORITY_WRITER + 1

	summary_keys = [
		"main/prec",
		"main/rec",
		"main/f1",
		"val/main/prec",
		"val/main/rec",
		"val/main/f1",
		# "class_coverage",
	]


	def __init__(self,
				 n_classes: int,
				 trigger = (1, "epoch"),
				 # key: str = "predictions",
				 beta: float = 1,
				 only_available: bool = True,
			 	):
		self._trigger = trigger_module.get_trigger(trigger)

		# self.key = key
		self.beta = beta
		self.n_classes = n_classes
		self._model = None
		self.only_available = only_available

	@property
	def predictions(self):
		key = "train" if chainer.config.train else "val"
		return self._model.predictions[key]

	def initialize(self, trainer=None):
		# self._reset_summary()
		assert trainer is not None
		self._model = trainer.updater.get_optimizer("main").target

	def finalize(self):
		self._model = None

	def __call__(self, trainer=None, updater=None) -> None:

		reporter = reporter_module.get_current_reporter()

		if trainer is None:
			trainer = object()
			setattr(trainer, "updater", updater)

		if not self._trigger(trainer):
			return

		values = self.summarize()
		reporter.report(values)
		return values

	def summarize(self) -> dict:

		logits, true = zip(*self.predictions)
		logits, true = np.vstack(logits), np.hstack(true)

		n_cls = self.n_classes
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

		return {
			"main/prec": np.nanmean(precision[count_mask]),
			"main/rec": np.nanmean(recall[count_mask]),
			f"main/f{int(self.beta)}": np.nanmean(fbeta_score[count_mask]),
			"class_coverage": np.mean(count_mask),
		}

