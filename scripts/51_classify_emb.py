#!/usr/bin/env python
from __future__ import annotations

if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import typing as T

from cvargparse import Arg
from cvargparse import BaseParser
from functools import partial
from matplotlib import pyplot as plt
from sklearn import metrics
from tabulate import tabulate


try:
	from cuml.manifold import TSNE
	tsne_module = "CuML"
except ImportError:
	from sklearn.manifold import TSNE
	tsne_module = "scikit-learn"

class Features:

	@classmethod
	def new(cls, npz_file, subset: str = "train"):
		feats, labs = npz_file[f"{subset}/features"], npz_file[f"{subset}/labels"]
		return cls(feats, labs, subset)

	def __init__(self, features, labels, subset: str = None):
		super().__init__()

		self._feats = features
		self._labs = labels
		self._classes = np.unique(labels)

		self._subset = subset

		logging.info(f"=== Subset: {subset} ===")
		logging.info(f"{len(self._labs):,d} samples {self._feats.shape}")
		logging.info(f"{len(self._classes):,d} classes")

	def get_data(self):
		return self._feats, self._labs


class Classifier:
	def __init__(self, classes):
		self._n_classes = len(classes)
		_, _ids = np.unique(classes, return_inverse=True)

		self._cls2idx = dict(zip(classes, _ids))
		self._cls_means = None

	def fit(self, feats: Features):
		X, y = feats.get_data()
		N, feat_size = X.shape
		self._cls_means = np.zeros((self._n_classes, feat_size))

		for cls in np.unique(y):
			mask = y == cls
			idx = self._cls2idx[cls]

			self._cls_means[idx] = X[mask].mean(axis=0)
		return self

	def score(self, feats: Features, metric) -> float:
		X, y = feats.get_data()
		dists = metrics.pairwise.euclidean_distances(X, self._cls_means, squared=True)

		preds = dists.argmin(axis=1)

		idxs = self.cls2idx(y)
		return metric(idxs, preds)

	def update(self, X: np.ndarray, cls: int):
		idx = self._cls2idx[cls]
		self._cls_means[idx] = X.mean(axis=0)
		return self

	def cls2idx(self, y):
		return np.array([self._cls2idx[lab] for lab in y])


	def plot(self, ax: plt.Axes = None):
		assert self._cls_means is not None, \
			"fit() was not called yet!"

		ax = ax or plt.gca()

		tsne = TSNE(n_components=2)
		pts = tsne.fit_transform(self._cls_means)

		ax.scatter(*pts.T)

def evaluate(clf, train, val):

	rows = []
	tr_idxs = clf.cls2idx(train._classes)
	val_idxs = clf.cls2idx(val._classes)
	val_only_idxs = clf.cls2idx(val._classes[~np.in1d(val._classes, train._classes)])
	comm_idxs = clf.cls2idx(train._classes[np.in1d(train._classes, val._classes)])

	f1_score = partial(metrics.f1_score, average="macro", zero_division=0)

	score_metrics = [
		(f"Accuracy ({len(val_idxs)})",
			metrics.accuracy_score),

		(f"F1-Score ({len(val_idxs)})",
			partial(f1_score, labels=val_idxs)),

		(f"F1-Score (only trained: {len(tr_idxs)})",
			partial(f1_score, labels=tr_idxs)),

		(f"F1-Score (only common: {len(comm_idxs)})",
			partial(f1_score, labels=comm_idxs)),

		(f"F1-Score (only val, not train: {len(val_only_idxs)})",
			partial(f1_score, labels=val_only_idxs)),
	]

	for name, metric in score_metrics:
		rows.append([
			name,
			f"{clf.score(train, metric):.2%}",
			f"{clf.score(val, metric):.2%}",
		])


	tab = tabulate(
		rows,
		headers=["Metric", "Training", "Validation"],
		tablefmt="fancy_grid"
	)
	print(tab)

def update_clf(clf, features: Features, *,
			   n_samples: int = 1,
			   labels=None,
			   rnd: np.random.RandomState = None):

	X, y = features.get_data()

	rnd = rnd or np.random.RandomState()

	if labels is None:
		labels = y

	selected = []
	ignored = 0

	for cls in np.unique(labels):
		mask = y == cls
		pos_idxs = np.where(mask)[0]
		n = min(n_samples, len(pos_idxs) - n_samples)
		if n <= 0:
			ignored += 1
			continue
		idxs = rnd.choice(pos_idxs, n, replace=False)

		clf.update(X[idxs], cls)
		selected.extend(idxs)

	remaining = np.ones_like(y, dtype=bool)
	remaining[selected] = False

	if ignored != 0:
		logging.info(f"Ignored {ignored} classes during update (too few samples)")

	return Features(X[remaining], y[remaining], f"{features._subset} updated")

def main(args):
	logging.info(f"Imported TSNE from {tsne_module}")
	rnd = np.random.RandomState(args.seed)
	logging.info(f"Using {args.seed=}")

	cont = np.load(args.embeddings)
	train, val = [Features.new(cont, subset) for subset in ["train", "val"]]

	all_classes = np.concatenate([train._classes, val._classes])

	classes = np.unique(all_classes)

	logging.info(f"Overall: {len(classes):,d} classes")

	clf = Classifier(classes)

	clf.fit(train)

	print(args.embeddings)
	evaluate(clf, train, val)

	val_only_cls = val._classes[~np.in1d(val._classes, train._classes)]
	new_val = update_clf(clf, val, labels=val_only_cls, n_samples=1)

	evaluate(clf, train, new_val)

parser = BaseParser()

parser.add_args([
	Arg("embeddings"),

	Arg.int("--seed"),
])

main(parser.parse_args())
