import abc
import chainer
import logging
import numpy as np
import typing as T

from chainer import functions as F
from contextlib import contextmanager
from cvmodelz import classifiers

from moth_classifier.core.classifier.size_model import SizeModel
from moth_classifier.core import dataset
from moth_classifier.core import prediction as preds


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def run_evaluations(pred, gt,
					evaluations: T.Dict[str, T.Callable],
					*,
					suffix: str,
					reporter: T.Callable):
	for metric, func in evaluations.items():
		reporter(**{f"{metric}{suffix}": func(pred, gt)})

class BaseClassifier(abc.ABC):

	def __init__(self, only_head: bool,
				 use_size_model: bool,
				 loss_alpha: float = 0.5,
				 *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._only_head = only_head
		self._use_size_model = use_size_model

		self.init_accumulators()

		with self.init_scope():
			# use 1.0 as default setting for the loss scaling
			self.add_persistent("loss_alpha", 1.0)

		if not self._use_size_model:
			return

		# set the alpha only if we use a size model
		self.loss_alpha = loss_alpha

		n_classes = 69# self.n_classes
		logging.info(f"Initializing size model for {n_classes} classes ({loss_alpha=})")

		with self.init_scope():
			self._size_model = SizeModel(n_classes)#self.n_classes)


	def init_accumulators(self, few_shot_count: int = 20, many_shot_count: int = 50) -> None:
		self._accumulators = {
			"train": preds.PredictionAccumulator(
				few_shot_count=few_shot_count,
				many_shot_count=many_shot_count),
			"val": preds.PredictionAccumulator(
				few_shot_count=few_shot_count,
				many_shot_count=many_shot_count),
		}

	@property
	def accumulator(self) -> preds.PredictionAccumulator:
		mode = "train" if chainer.config.train else "val"
		return self._accumulators[mode]

	def load(self, weights, *args, finetune: bool, **kwargs):
		if not finetune:
			self.model.reinitialize_clf(self.n_classes, self.output_size)
		super().load(weights, *args, finetune=finetune, **kwargs)


	def load_model(self, *args, **kwargs):
		kwargs["feat_size"] = kwargs.get("feat_size", self.output_size)
		super().load_model(*args, **kwargs)


	def _get_features(self, X, model):
		if self._only_head:
			with self.eval_mode():
				features = model(X, layer_name=model.meta.feature_layer)
		else:
			features = model(X, layer_name=model.meta.feature_layer)

		return _unpack(features)

	def extract(self, X):
		return self._get_features(X, self.model)

	@contextmanager
	def eval_mode(self):
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			yield


	def eval_prediction(self, pred, gt, suffix=""):

		self.accumulator.update(pred, gt)
		accum = self.accumulator
		evaluations = dict(
			accu=self.model.accuracy,
			accu2=preds.Metric(accum, key="accuracy"),
			prec=preds.Precision(accum),
			rec=preds.Recall(accum),
			f1=preds.FScore(accum, beta=1),

			few_count=preds.Metric(accum, key="few_shot_count"),
			few_cls_count=preds.Metric(accum, key="few_shot_cls_count"),
			prec_fs=preds.Metric(accum, key="precision/few-shot@20"),
			rec_fs=preds.Metric(accum, key="recall/few-shot@20"),
			f1_fs=preds.Metric(accum, key="f1_score/few-shot@20"),

			med_count=preds.Metric(accum, key="med_shot_count"),
			med_cls_count=preds.Metric(accum, key="med_shot_cls_count"),
			prec_mds=preds.Metric(accum, key="precision/med-shot@20-50"),
			rec_mds=preds.Metric(accum, key="recall/med-shot@20-50"),
			f1_mds=preds.Metric(accum, key="f1_score/med-shot@20-50"),

			many_count=preds.Metric(accum, key="many_shot_count"),
			many_cls_count=preds.Metric(accum, key="many_shot_cls_count"),
			prec_ms=preds.Metric(accum, key="precision/many-shot@50"),
			rec_ms=preds.Metric(accum, key="recall/many-shot@50"),
			f1_ms=preds.Metric(accum, key="f1_score/many-shot@50"),
		)



		return run_evaluations(pred, gt,
			evaluations=evaluations,
			suffix=suffix,
			reporter=self.report)


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


class Classifier(BaseClassifier, classifiers.Classifier):


	def forward(self, X, y, sizes=None):
		feat = self.extract(X)

		pred = self.model.clf_layer(feat)
		loss0 = self.loss(pred, y)

		pred = self.size_model(sizes, pred, y)
		loss1 = self.loss(pred, y)

		self.eval_prediction(pred, y)

		loss = self.loss_alpha * loss0 + (1 - self.loss_alpha) * loss1
		self.report(loss=loss)

		return loss

