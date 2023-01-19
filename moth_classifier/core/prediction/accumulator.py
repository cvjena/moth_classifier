import chainer
import numpy as np
import networkx as nx
import typing as T

from chainer import functions as F
from collections import defaultdict
from cvdatasets import Hierarchy


def _to_cpu(var, dtype=None):
	res = chainer.cuda.to_cpu(chainer.as_array(var))
	if dtype is None:
		return res
	return res.astype(dtype)

class Scores(T.NamedTuple):
	accuracy: float = 0.0
	precision: float = 0.0
	recall: float = 0.0
	f_score: float = 0.0

def _calc_h_scores(pred, gt, *, graph, beta: float = 1.0) -> Scores:
	"""
		Calculates accuracy, hierarchical precision, recall and F-beta scores
	"""

	# confusion matrix
	cm = defaultdict(int)
	root = None

	root = next(nx.topological_sort(graph))

	for key in zip(gt, pred):
		cm[key] += 1

	confused = 0
	precision, recall, fbeta_score = 0, 0, 0
	beta_square = beta ** 2

	for (gt_uid, pred_uid), count in cm.items():
		if gt_uid != pred_uid:
			confused += count

		gt_anc = set(nx.ancestors(graph, gt_uid)) | {gt_uid}
		pred_anc = set(nx.ancestors(graph, pred_uid)) | {pred_uid}

		gt_anc -= {root}
		pred_anc -= {root}

		n_matched = len(pred_anc.intersection(gt_anc))
		if n_matched == 0:
			continue

		cur_prec, cur_recall = 0, 0

		if pred_anc:
			cur_prec = n_matched / len(pred_anc)

		if gt_anc:
			cur_recall = n_matched / len(gt_anc)

		cur_prec *= count
		cur_recall *= count

		precision += cur_prec
		recall += cur_recall

		numerator = (1 + beta_square) * cur_prec * cur_recall
		denominator = beta_square * cur_prec + cur_recall
		fbeta_score += numerator / denominator

	N = len(gt)

	return Scores(
		accuracy=(N - confused) / N,
		precision=precision / N,
		recall=recall / N,
		f_score=fbeta_score / N)

class PredictionAccumulator:

	def __init__(self,
		logits = None, gt = None, *,
		few_shot_count: int = -1,
		many_shot_count: int = -1,
		hierarchy: Hierarchy = None,
		use_hc: bool = False):

		super().__init__()

		self._logits = [] if logits is None else [_to_cpu(logits, np.float16)]
		self._gt = [] if gt is None else [_to_cpu(gt, np.int32)]

		self.few_shot_count = few_shot_count
		self.many_shot_count = many_shot_count
		self.hierarchy = hierarchy
		self.use_hc = use_hc

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

	def calc_hierarchical_metrics(self, *, only_available: bool = True, beta: int = 1):
		"""
			partially from https://github.com/cabrust/chia/blob/main/chia/components/evaluators/hierarchical.py#L31
		"""
		logits, true = self.reset()

		# transform GT labels to original label uid
		hc_orig_gt = list(map(self.hierarchy.label_transform, true))

		pred = chainer.as_array(F.sigmoid(logits))
		deembed_pred_dist = self.hierarchy.deembed_dist(pred)

		if only_available:
			_available_leaves = np.unique(hc_orig_gt)
			idx_of_available = set(_available_leaves)

			for label in _available_leaves:
				idx_of_available |= set(nx.ancestors(self.hierarchy.graph, label))

			deembed_pred_dist = [
				list(filter(lambda tup: tup[0] in idx_of_available, preds))
				for preds in deembed_pred_dist
			]

		# argmax
		predictions = np.array([
			sorted(dist, key=lambda tup: tup[1], reverse=True)[0][0]
				for dist in deembed_pred_dist
		])

		scores = _calc_h_scores(predictions, hc_orig_gt,
			beta=beta, graph=self.hierarchy.graph)

		self._metrics = {
			"accuracy": scores.accuracy,
			"precision": scores.precision,
			"recall": scores.recall,
			f"f{beta}_score": scores.f_score,

		}

		return self._metrics

	def calc_metrics2(self, *, only_available: bool = True, beta: int = 1):
		pass


	def calc_metrics(self, *, only_available: bool = True, beta: int = 1):
		if self._metrics is not None:
			return self._metrics

		if self.use_hc:
			return self.calc_hierarchical_metrics(only_available=only_available, beta=beta)

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

		prec = precision[count_mask]
		rec = recall[count_mask]
		fbeta = fbeta_score[count_mask]

		self._metrics = {
			"accuracy": np.mean(pred == true),
			"precision": np.nanmean(prec),
			"recall": np.nanmean(rec),
			f"f{beta}_score": np.nanmean(fbeta),
		}

		if -1 not in [self.few_shot_count, self.many_shot_count]:

			fsc = self.few_shot_count
			msc = self.many_shot_count
			_counts = counts[count_mask]
			fs_mask = _counts < fsc
			ms_mask = _counts > msc
			mds_mask = np.logical_and(~fs_mask, ~ms_mask)

			if fs_mask.sum():
				self._metrics.update({
					"few_shot_cls_count": fs_mask.sum(),
					"few_shot_count": _counts[fs_mask].sum(),
					f"precision/few-shot@{fsc}": np.nanmean(prec[fs_mask]),
					f"recall/few-shot@{fsc}": np.nanmean(rec[fs_mask]),
					f"f{beta}_score/few-shot@{fsc}": np.nanmean(fbeta[fs_mask]),
				})

			if mds_mask.sum():
				self._metrics.update({
					"med_shot_cls_count": mds_mask.sum(),
					"med_shot_count": _counts[mds_mask].sum(),
					f"precision/med-shot@{fsc}-{msc}": np.nanmean(prec[mds_mask]),
					f"recall/med-shot@{fsc}-{msc}": np.nanmean(rec[mds_mask]),
					f"f{beta}_score/med-shot@{fsc}-{msc}": np.nanmean(fbeta[mds_mask]),
					})

			if ms_mask.sum():
				self._metrics.update({
					"many_shot_cls_count": ms_mask.sum(),
					"many_shot_count": _counts[ms_mask].sum(),
					f"precision/many-shot@{msc}": np.nanmean(prec[ms_mask]),
					f"recall/many-shot@{msc}": np.nanmean(rec[ms_mask]),
					f"f{beta}_score/many-shot@{msc}": np.nanmean(fbeta[ms_mask]),
				})


		return self._metrics


	def reset(self):
		logits, gt = np.vstack(self._logits), np.hstack(self._gt)
		self._logits, self._gt = [], []
		return logits, gt
