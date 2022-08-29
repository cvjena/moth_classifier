#!/usr/bin/env python
from __future__ import annotations

if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import typing as T

from cvargparse import Arg
from cvargparse import GPUParser
from sklearn import metrics
from tabulate import tabulate
from matplotlib import pyplot as plt
from itertools import cycle as cycler

try:
	from cuml.manifold import TSNE
	from cuml.neighbors import KNeighborsClassifier
	from cuml.cluster import DBSCAN
	from cuml.cluster import KMeans
	used_module = "CuML"
except ImportError as e:
	from sklearn.manifold import TSNE
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.cluster import DBSCAN
	from sklearn.cluster import KMeans
	used_module = "scikit-learn"

	print(e)


colors = cycler([
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
])

markers = cycler([
	"o",
	"P",
	"d",
	"v",
	".",
	"*",
	"H",
	"X",
	"s",
	"p",
	"+",
	"x",
	# "1",
])


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

def evaluate_knn(train: Features, val: Features, *, k):
	X, y = train.get_data()
	X_val, y_val = val.get_data()

	clf = KNeighborsClassifier(n_neighbors=k)

	clf.fit(X, y)

	accuracy = clf.score(X_val, y_val)
	print(f"k-NN Accuracy: {accuracy:.2%}")


def evaluate_clustering(train: Features, val: Features,
	*args,
	markers,
	classes,
	epsilons = [2.0, 0.5],
	**kw):

	rows = []

	metric = lambda true, pred: "{:.2f} | {:.2f} | {:.2f}".format(
		*metrics.homogeneity_completeness_v_measure(true, pred))

	kw["n_clusters"] = len(classes)

	# metric = lambda X, preds: "{:.2f}".format(
	# 	metrics.silhouette_score(X, preds, metric='euclidean'))

	for data, subset in [(train, "train"), (val, "val")]:
		X, y = data.get_data()
		row = [subset]

		# fig, axs = plt.subplots(ncols=len(epsilons), squeeze=False)
		# fig.suptitle(subset)

		for i, eps in enumerate(epsilons):
			# clust = DBSCAN(eps=eps, *args, **kw)
			# clust.fit(X)
			# preds = clust.labels_

			preds = KMeans(*args, **kw).fit_predict(X)

			# ax = axs[np.unravel_index(i, axs.shape)]
			# X2d = TSNE(n_components=2).fit_transform(X)
			# for cls in classes[np.unique(preds)]:
			# 	c, m = markers.get(cls, ("k", "."))# if cls != -1 else

			# 	if cls not in markers:
			# 		print(cls)
			# 	xy = X2d[preds==cls].T
			# 	ax.scatter(*xy, c=c, marker=m)

			# ax.set_title(f"DBSCAN@{eps=}")

			# value = metric(X, preds) if (preds != -1).any() else 0
			value = metric(y, preds) if (preds != -1).any() else 0
			row.append(value)

		rows.append(row)

	tab = tabulate(rows,
		headers=["Subset"] + [f"DBSCAN@{eps=}" for eps in epsilons],
		tablefmt="fancy_grid",
	)

	print(tab)

	# plt.show()
	# plt.close()



def main(args):

	logging.info(f"Using {used_module} module")
	rnd = np.random.RandomState(args.seed)
	logging.info(f"Using {args.seed=}")

	cont = np.load(args.embeddings)
	train, val = [Features.new(cont, subset) for subset in ["train", "val"]]
	X, y = train.get_data()
	X_val, y_val = val.get_data()

	all_classes = np.concatenate([y, y_val])

	classes = np.unique(all_classes)

	cols_markers = {cls: (c, m) for (cls, c, m) in zip(classes, colors, markers)}
	logging.info(f"Overall: {len(classes):,d} classes")

	logging.info("====== Evaluation ======")

	evaluate_knn(train, val, k=args.neighbors)

	evaluate_clustering(train, val,
		classes=classes,
		epsilons=[-1],
		markers=cols_markers)

parser = GPUParser([
	Arg("embeddings"),

	# Arg.int("--tsne", "-tsne", default=-1,
	# 	help="Perform dimensionality reduction with TSNE before clustering"),

	Arg.int("--seed"),
	Arg.int("--neighbors", "-k", default=5,
		help="Number of neighbors for k-NN"),

])


main(parser.parse_args())
