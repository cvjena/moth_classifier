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

from sklearn.mixture import GaussianMixture as GMM

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

def predict(X, clustering: str, *args, n_classes, eps, **kw):

	if clustering == "DBSCAN":

		clust = DBSCAN(eps=eps, *args, **kw)
		clust.fit(X)
		return clust.labels_, f"DBSCAN@{eps=}"


	elif clustering == "KMeans":

		clust = KMeans(*args, n_clusters=n_classes, **kw)
		return clust.fit_predict(X), f"KMeans@k={n_classes}"

	elif clustering == "GMM":

		clust = GMM(*args,
			n_components=n_classes,
			covariance_type="diag", **kw)
		return clust.fit_predict(X), f"GMM@k={n_classes}"

	else:
		raise NotImplementedError(f"Unknown clustering: {clustering}")

def evaluate(metric, X, y, pred):

	if np.all(pred == 0):
		return "n/a"

	if metric == "v-measure":
		values = metrics.homogeneity_completeness_v_measure(y, pred)
		return "{:.2f} | {:.2f} | {:.2f}".format(*values)

	elif metric == "silhoette":
		value = metrics.silhouette_score(X, pred, metric='euclidean')

	elif metric == "rand_idx":
		value = metrics.rand_score(y, pred)

	elif metric == "adj_rand_idx":
		value = metrics.adjusted_rand_score(y, pred)

	elif metric == "mi":
		value = metrics.mutual_info_score(y, pred)

	elif metric == "adj_mi":
		value = metrics.adjusted_mutual_info_score(y, pred)

	elif metric == "norm_mi":
		value = metrics.normalized_mutual_info_score(y, pred)


	else:
		raise NotImplementedError(f"Unknown metric: {metric}")

	return f"{value:.2f}"



def evaluate_clustering(data: T.List[Features], *args,
	markers, classes,
	clustering,
	metrics=["v-measure"],
	epsilons = [2.0, 0.5],
	plot: bool = False,
	**kwargs):

	rows = []

	fig, axs = None, None
	if plot:
		fig, axs = plt.subplots(nrows=len(epsilons), ncols=len(data), squeeze=False)

	headers = None

	for j, ds in enumerate(data):
		X, y = ds.get_data()
		_classes = np.unique(y)
		row = [ds._subset]
		_heads = ["Subset"]

		for i, eps in enumerate(epsilons):


			preds, clust_name = predict(X, clustering,
				*args, n_classes=len(_classes), eps=eps, **kwargs)

			if plot:
				ax = axs[i,j]
				X2d = TSNE(n_components=2, method="fft").fit_transform(X)
				for cls in _classes[np.unique(preds)]:
					c, m = markers.get(cls, ("k", "."))# if cls != -1 else

					# if cls not in markers:
					# 	print(cls)

					xy = X2d[preds==cls].T
					ax.scatter(*xy, c=c, marker=m)

				ax.set_title(f"{ds._subset} | {clust_name}")

			for metric in metrics:
				score = evaluate(metric, X, y, preds)
				row.append(score)
				_heads.append(f"{clust_name}\n{metric}")

		if headers is None:
			headers = _heads
		rows.append(row)

	tab = tabulate(rows,
		headers=headers,
		tablefmt="fancy_grid",
	)

	print(tab)

	return fig, axs



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

	fig, axs = evaluate_clustering([train, val],
		clustering=args.clustering,
		classes=classes,
		# epsilons=[0.5, 2.0, 5.0],
		epsilons=[-1],
		metrics=args.metrics,
		markers=cols_markers,
	)

	if fig is not None:
		fig.suptitle(args.embeddings)

		plt.show()
		plt.close()


parser = GPUParser([
	Arg("embeddings"),

	# Arg.int("--tsne", "-tsne", default=-1,
	# 	help="Perform dimensionality reduction with TSNE before clustering"),

	Arg.int("--seed"),
	Arg.int("--neighbors", "-k", default=5,
		help="Number of neighbors for k-NN"),

	Arg("--clustering", default="KMeans",
		choices=["KMeans", "GMM", "DBSCAN"]
		),

	Arg("--metrics", nargs="+",
		default=[
			"v-measure",
			"adj_rand_idx",
			"adj_mi",
		],
		choices=[
			"v-measure",
			"rand_idx",
			"adj_rand_idx",
			"mi",
			"adj_mi",
			"norm_mi",
		])

])


main(parser.parse_args())
