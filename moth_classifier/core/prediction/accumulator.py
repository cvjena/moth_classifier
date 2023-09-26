import chainer
import multiprocessing as mp
import networkx as nx
import numpy as np
import typing as T
import weakref

from chainer import functions as F
from collections import Counter
from collections import defaultdict
from cvdatasets import Hierarchy
from functools import partial


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


def argmax(arr, axis=1, *, mask=None):
	if mask is None:
		return np.argmax(arr, axis=1)

	# fill the non-mask entries with the smallest value
	arr[:, ~mask] = arr.min()
	return np.argmax(arr, axis=1)

def h_argmax(arr, hierarchy: Hierarchy, *, available_uuids = set()):
	""" Hierarchical argmax. """

	probs = chainer.as_array(F.sigmoid(arr))
	probs_per_node = hierarchy.deembed_dist(probs)

	if available_uuids:
		# remove all probabilities that are not available
		probs_per_node = [
			list(filter(lambda tup: tup[0] in available_uuids, probs))
			for probs in probs_per_node
		]

	# sort by the de-embedded probabilities and
	# return the class uuid of the one with the max prob
	preds = [sorted(probs, key=lambda tup: tup[1], reverse=True)[0][0]
		for probs in probs_per_node]

	return preds

POOL = None

class PredictionAccumulator:

	def __del__(self):
		if hasattr(self, "_pool"):
			self._pool = None

	@property
	def pool(self):
		if self._pool is not None:
			return self._pool()
		return None

	def _init_pool(self, n_jobs):
		global POOL

		if n_jobs is not None and n_jobs <= 1:
			self._pool = None
			return

		if POOL is None:
			POOL = mp.Pool(n_jobs)

		self._pool = weakref.ref(POOL)

	def __init__(self,
		logits = None, gt = None, *,
		few_shot_count: int = -1,
		many_shot_count: int = -1,
		hierarchy: Hierarchy = None,
		use_hc: bool = False,
		n_jobs: int = -1):
		super().__init__()

		self._init_pool(n_jobs)


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

	def calc_metrics(self, *, only_available: bool = True, beta: int = 1):
		if self._metrics is not None:
			return self._metrics

		preds, gt = self.reset(only_available=only_available)

		scores = _calc_scores(preds, gt, beta=beta)
		self._metrics = {
			"accuracy": scores.accuracy,
			"precision": scores.precision,
			"recall": scores.recall,
			f"f{beta}_score": scores.f_score,
		}

		if self.hierarchy is not None and self.use_hc:
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

	def do_work(self, work, payload):
		pool = self.pool
		mapper = map if pool is None else pool.map
		res = mapper(work, payload)

		return np.hstack(list(res))


	def reset(self, *, only_available: bool = True):
		true = np.hstack(self._gt)
		logits = self._logits

		self._logits, self._gt = [], []

		if self.use_hc:
			assert self.hierarchy is not None, \
				"For hierarchical classification, a hierarchy is required!"

			# transform GT labels to original label uid
			true_uids = self._to_uids(true)

			uid_of_available = set()
			if only_available:
				# get only those present in the subset
				_available_leaves = np.unique(true_uids)
				uid_of_available = set(_available_leaves)

				# ... and their ancestors
				for label in _available_leaves:
					uid_of_available |= set(nx.ancestors(self.hierarchy.graph, label))

			work = partial(h_argmax, hierarchy=self.hierarchy, available_uuids=uid_of_available)
			pred_uids = self.do_work(work, logits)
			return pred_uids, true_uids

		else:
			# simple argmax on the logits, even softmax is not needed here
			mask_of_available = None
			if only_available:
				_, n_cls = logits[0].shape
				mask_of_available = np.in1d(np.arange(n_cls), np.unique(true))

			work = partial(argmax, axis=1, mask=mask_of_available)
			preds = self.do_work(work, logits)

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
