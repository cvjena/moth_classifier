import chainer
import numpy as np
import networkx as nx

from chainer import functions as F
from collections import defaultdict
from cvdatasets import Hierarchy


def _to_cpu(var, dtype=None):
	res = chainer.cuda.to_cpu(chainer.as_array(var))
	if dtype is None:
		return res
	return res.astype(dtype)

class PredictionAccumulator:

	def __init__(self,
		logits = None, gt = None, *,
		few_shot_count: int = -1,
		many_shot_count: int = -1,
		hierarchy: Hierarchy = None):

		super().__init__()

		self._logits = [] if logits is None else [_to_cpu(logits, np.float16)]
		self._gt = [] if gt is None else [_to_cpu(gt, np.int32)]

		self.few_shot_count = few_shot_count
		self.many_shot_count = many_shot_count
		self.hierarchy = hierarchy

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
		logits, gt = self.reset()
		N = len(logits)

		dim = self.hierarchy.orig_lab_to_dimension.get
		G = self.hierarchy.graph
		root = next(nx.topological_sort(G))

		# transform GT labels to original label uid
		hc_orig_gt = list(map(self.hierarchy.label_transform, gt))


		pred = chainer.as_array(F.sigmoid(logits))
		deembed_pred_dist = self.hierarchy.deembed_dist(pred)

		# argmax and select the correct dimension
		predictions = np.array([
			sorted(dist, key=lambda x: x[1], reverse=True)[0][0]
				for dist in deembed_pred_dist
		])

		# confusion matrix
		cm = defaultdict(int)

		for key in zip(hc_orig_gt, predictions):
			cm[key] += 1

		confused = 0
		precision, recall, fbeta_score = 0, 0, 0
		beta_square = beta ** 2

		for (gt_uid, pred_uid), count in cm.items():
			if gt_uid != pred_uid:
				confused += count

			# Hierarchical Precision / Recall
			gt_anc = set(nx.ancestors(G, gt_uid)) | {gt_uid}
			pred_anc = set(nx.ancestors(G, pred_uid)) | {pred_uid}

			gt_anc -= {root}
			pred_anc -= {root}

			n_matched = len(pred_anc.intersection(gt_anc))
			if n_matched == 0:
				continue

			cur_h_prec, cur_h_recall = 0, 0

			if pred_anc:
				cur_h_prec = n_matched / len(pred_anc)

			if gt_anc:
				cur_h_recall = n_matched / len(gt_anc)

			cur_h_prec *= count
			cur_h_recall *= count

			precision += cur_h_prec
			recall += cur_h_recall

			numerator = (1 + beta_square) * cur_h_prec * cur_h_recall
			denominator = beta_square * cur_h_prec + cur_h_recall
			fbeta_score += numerator / denominator

		self._metrics = {
			"accuracy": (N - confused) / N,
			"recall": recall / N,
			"precision": precision / N,
			f"f{beta}_score": fbeta_score / N,

		}

		return self._metrics


	def calc_metrics(self, *, only_available: bool = True, beta: int = 1):
		if self._metrics is not None:
			return self._metrics

		if self.hierarchy is not None:
			return self.calc_hierarchical_metrics(only_available=only_available, beta=beta)

		n_cls = max(logits.shape[1], true.max())
		if only_available:
			_logits = np.full_like(logits, fill_value=logits.min())
			available = np.unique(true)
			_logits[:, available] = logits[:, available]
			logits = _logits

		pred = logits.argmax(axis=1)

		self._metrics = {
			"accuracy": np.mean(pred == true),
		}

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

		return self._calc_metrics(precision, recall, fbeta_score, counts, beta=beta)


	def _calc_metrics(self, precision, recall, fbeta_score, counts, *, beta):
		count_mask = counts != 0


		prec = precision[count_mask]
		rec = recall[count_mask]
		fbeta = fbeta_score[count_mask]

		self._metrics.update({
			"precision": np.nanmean(prec),
			"recall": np.nanmean(rec),
			f"f{beta}_score": np.nanmean(fbeta),
		})

		if -1 in [self.few_shot_count, self.many_shot_count]:
			return self._metrics

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
