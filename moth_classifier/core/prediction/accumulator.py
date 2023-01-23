import chainer
import networkx as nx
import numpy as np
import typing as T

from chainer import functions as F
from collections import Counter
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

def confusion_matrix(pred, gt) -> dict:

	cm = defaultdict(int)
	for key in zip(gt, pred):
		cm[key] += 1
	return dict(cm)

def _calc_h_scores(pred, gt, *, graph, beta: float = 1.0) -> Scores:
	"""
		Calculates accuracy, hierarchical precision, recall and F-beta scores
	"""

	# confusion matrix
	cm = confusion_matrix(pred, gt)
	root = next(nx.topological_sort(graph))

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

def _calc_scores(pred, gt, *, beta: float = 1.0) -> Scores:
	"""
		Computes class-wise precisions, recalls and fbeta-scores and averages them
	"""
	cm = confusion_matrix(pred, gt)
	counts = Counter(gt)
	relevant_counts = Counter(pred)

	TP, FN, FP = defaultdict(int), defaultdict(int), defaultdict(int)
	for (gt_uid, pred_uid), count in cm.items():
		if gt_uid == pred_uid:
			TP[gt_uid] = count
			continue

		FP[pred_uid] += count
		FN[gt_uid] += count

	classes = sorted(set(TP) | set(FP) | set(FN))

	precision = np.zeros(len(classes), dtype=np.float32)
	recall = np.zeros(len(classes), dtype=np.float32)
	fbeta_score = np.zeros(len(classes), dtype=np.float32)
	beta_square = beta ** 2

	n_correct = 0
	for i, cls in enumerate(classes):
		n_correct += TP[cls]

		n_retrieved = TP[cls] + FP[cls]
		if n_retrieved != 0:
			precision[i] = TP[cls] / n_retrieved

		n_relevant = TP[cls] + FN[cls]
		if n_relevant != 0:
			recall[i] = TP[cls] / n_relevant

		numerator = (1 + beta_square) * precision[i] * recall[i]
		denominator = beta_square * precision[i] + recall[i]
		if denominator != 0:
			fbeta_score[i] = numerator / denominator

	return Scores(
		accuracy=n_correct / len(gt),
		precision=precision.mean(),
		recall=recall.mean(),
		f_score=fbeta_score.mean()
	)


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
		hc_orig_gt = np.fromiter(map(self.hierarchy.label_transform, true), dtype=true.dtype)

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


	def calc_metrics(self, *, only_available: bool = True, beta: int = 1):
		if self._metrics is not None:
			return self._metrics

		logits, true = self.reset()

		preds, gt = self._predict(logits, true, only_available=only_available)

		scores = _calc_scores(preds, gt, beta=beta)
		self._metrics = {
			"accuracy": scores.accuracy,
			"precision": scores.precision,
			"recall": scores.recall,
			f"f{beta}_score": scores.f_score,
		}

		if self.hierarchy is not None:
			h_scores = _calc_h_scores(preds, gt, beta=beta, graph=self.hierarchy.graph)

			assert h_scores.accuracy == scores.accuracy, \
				f"accuracy should be in equal in both scores: ({h_scores.accuracy} != {scores.accuracy})"

			self._metrics.update({
				"h_precision": h_scores.precision,
				"h_recall": h_scores.recall,
				f"h_f{beta}_score": h_scores.f_score,
			})

		return self._metrics

	def _to_uids(self, labels: np.ndarray) -> np.ndarray:
		assert self.hierarchy is not None, \
			"A hierarchy is required!"
		return np.fromiter(map(self.hierarchy.label_transform, labels), dtype=labels.dtype)

	def _predict(self, logits, true, *, only_available: bool = True):

		# simple argmax on the logits, even softmax is not needed here
		if not self.use_hc:
			# unavailable logits can be simply set to the smallest value
			if only_available:
				fill_value = logits.min()
				_, n_cls = logits.shape
				mask_of_available = np.in1d(np.arange(n_cls), np.unique(true))
				logits[:, ~mask_of_available] = fill_value

			preds = logits.argmax(axis=1)

			# if we dont have any hierarchy, then just pass them as
			# simple logit ids
			if self.hierarchy is None:
				return preds, true

			# otherwise, we need to convert them to class uids,
			# so that further evaluation can be done using the
			# hierarchy information
			true_uids = self._to_uids(true)
			pred_uids = self._to_uids(preds)
			return pred_uids, true_uids

		# hierarchical classification is a bit more tricky...
		assert self.hierarchy is not None, \
			"For hierarchical classification, a hierarchy is required!"
		# first, we need to compute the probabilties for each node
		# and "unenbed" them back according to the hierarchical model
		probs = chainer.as_array(F.sigmoid(logits))
		probs_per_node = self.hierarchy.deembed_dist(probs)

		# transform GT labels to original label uid
		class_uids = self._to_uids(true)

		if only_available:
			# get only those present in the subset
			_available_leaves = np.unique(class_uids)
			uid_of_available = set(_available_leaves)

			# ... and their ancestors
			for label in _available_leaves:
				uid_of_available |= set(nx.ancestors(self.hierarchy.graph, label))

			# remove all probabilities that are not available
			probs_per_node = [
				list(filter(lambda tup: tup[0] in uid_of_available, probs))
				for probs in probs_per_node
			]

		# now we can sort by the probability and pick the first element
		# from the tuple, which is the class uid
		predictions = np.array([
			sorted(probs, key=lambda tup: tup[1], reverse=True)[0][0]
				for probs in probs_per_node
		])

		return predictions, class_uids


	def reset(self):
		logits, gt = np.vstack(self._logits), np.hstack(self._gt)
		self._logits, self._gt = [], []
		return logits, gt


	# def calc_metrics(self, *, only_available: bool = True, beta: int = 1):

	# 	if self._metrics is not None:
	# 		return self._metrics

	# 	if self.use_hc:
	# 		return self.calc_hierarchical_metrics(only_available=only_available, beta=beta)

	# 	logits, true = self.reset()
	# 	n_cls = max(logits.shape[1], true.max())
	# 	if only_available:
	# 		_logits = np.full_like(logits, fill_value=logits.min())
	# 		available = np.unique(true)
	# 		_logits[:, available] = logits[:, available]
	# 		logits = _logits

	# 	pred = logits.argmax(axis=1)

	# 	counts = np.bincount(true, minlength=n_cls + 1)[:n_cls]
	# 	relevant = np.bincount(pred, minlength=n_cls + 1)[:n_cls]

	# 	tp_mask = np.where(pred == true, true, n_cls)
	# 	tp = np.bincount(tp_mask, minlength=n_cls + 1)[:n_cls]

	# 	precision = np.zeros_like(tp, dtype=np.float32)
	# 	recall = np.zeros_like(tp, dtype=np.float32)
	# 	fbeta_score = np.zeros_like(tp, dtype=np.float32)

	# 	count_mask = counts != 0
	# 	relev_mask = relevant != 0

	# 	precision[relev_mask] = tp[relev_mask] / relevant[relev_mask]
	# 	recall[count_mask] = tp[count_mask] / counts[count_mask]

	# 	# F-Measure
	# 	beta_square = beta ** 2
	# 	numerator = (1 + beta_square) * precision * recall
	# 	denominator = beta_square * precision + recall
	# 	mask = denominator != 0
	# 	fbeta_score[mask] = numerator[mask] / denominator[mask]

	# 	prec = precision[count_mask]
	# 	rec = recall[count_mask]
	# 	fbeta = fbeta_score[count_mask]

	# 	self._metrics = {
	# 		"accuracy": np.mean(pred == true),
	# 		"precision": np.nanmean(prec),
	# 		"recall": np.nanmean(rec),
	# 		f"f{beta}_score": np.nanmean(fbeta),
	# 	}

	# 	if -1 not in [self.few_shot_count, self.many_shot_count]:

	# 		fsc = self.few_shot_count
	# 		msc = self.many_shot_count
	# 		_counts = counts[count_mask]
	# 		fs_mask = _counts < fsc
	# 		ms_mask = _counts > msc
	# 		mds_mask = np.logical_and(~fs_mask, ~ms_mask)

	# 		if fs_mask.sum():
	# 			self._metrics.update({
	# 				"few_shot_cls_count": fs_mask.sum(),
	# 				"few_shot_count": _counts[fs_mask].sum(),
	# 				f"precision/few-shot@{fsc}": np.nanmean(prec[fs_mask]),
	# 				f"recall/few-shot@{fsc}": np.nanmean(rec[fs_mask]),
	# 				f"f{beta}_score/few-shot@{fsc}": np.nanmean(fbeta[fs_mask]),
	# 			})

	# 		if mds_mask.sum():
	# 			self._metrics.update({
	# 				"med_shot_cls_count": mds_mask.sum(),
	# 				"med_shot_count": _counts[mds_mask].sum(),
	# 				f"precision/med-shot@{fsc}-{msc}": np.nanmean(prec[mds_mask]),
	# 				f"recall/med-shot@{fsc}-{msc}": np.nanmean(rec[mds_mask]),
	# 				f"f{beta}_score/med-shot@{fsc}-{msc}": np.nanmean(fbeta[mds_mask]),
	# 				})

	# 		if ms_mask.sum():
	# 			self._metrics.update({
	# 				"many_shot_cls_count": ms_mask.sum(),
	# 				"many_shot_count": _counts[ms_mask].sum(),
	# 				f"precision/many-shot@{msc}": np.nanmean(prec[ms_mask]),
	# 				f"recall/many-shot@{msc}": np.nanmean(rec[ms_mask]),
	# 				f"f{beta}_score/many-shot@{msc}": np.nanmean(fbeta[ms_mask]),
	# 			})

	# 	return self._metrics
