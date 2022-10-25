#!/usr/bin/env python
from __future__ import annotations

if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import sys
import numpy as np
import typing as T

from cvargparse import Arg
from cvargparse import GPUParser
from sklearn import metrics
from tabulate import tabulate
from tqdm.auto import tqdm
from tqdm.auto import trange
from matplotlib import pyplot as plt
from itertools import cycle as cycler

from sklearn.mixture import GaussianMixture as GMM

try:
	from cuml import TSNE
	from cuml import UMAP
	from cuml import KNeighborsClassifier
	from cuml import DBSCAN
	from cuml import KMeans
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

def evaluate_knn(train: Features, val: Features, *, k, output):
	X, y = train.get_data()
	X_val, y_val = val.get_data()

	clf = KNeighborsClassifier(n_neighbors=k)

	clf.fit(X, y)

	accuracy = clf.score(X_val, y_val)
	print(f"k-NN Accuracy ({k=}): {accuracy:.2%}", file=output)

def predict(X, clustering: str, *args, n_classes, eps, random_state, n_components: int = -1, **kw):

	if n_components > 0:
		logging.debug(f"Reducing data from {X.shape[1]} to {n_components=}")
		reducer = TSNE if n_components == 2 else UMAP
		X = reducer(n_components=n_components).fit_transform(X)

	if clustering == "DBSCAN":
		clust = DBSCAN(
			min_samples=2,
			eps=eps, *args, **kw)
		# X = StandardScaler().fit_transform(X)

		clust.fit(X)
		labels = clust.labels_
		k = len(set(labels)) - (1 if -1 in labels else 0)


		return labels, f"DBSCAN@{eps=},{k=}", X


	elif clustering == "KMeans":

		clust = KMeans(*args, n_clusters=n_classes, random_state=random_state, **kw)
		return clust.fit_predict(X), f"KMeans@k={n_classes}", X

	elif clustering == "GMM":

		clust = GMM(*args,
			n_components=n_classes,
			random_state=random_state,
			covariance_type="diag", **kw)
		return clust.fit_predict(X), f"GMM@k={n_classes}", X

	else:
		raise NotImplementedError(f"Unknown clustering: {clustering}")

def evaluate(metric, X, y, pred):

	if np.all(pred == 0):
		return -1 #"n/a"

	if metric == "v-measure":
		raise NotImplementedError
		values = metrics.homogeneity_completeness_v_measure(y, pred)
		return "{:.2f} | {:.2f} | {:.2f}".format(*values)

	elif metric == "silhoette":
		return metrics.silhouette_score(X, pred, metric='euclidean')

	elif metric == "rand_idx":
		return metrics.rand_score(y, pred)

	elif metric == "adj_rand_idx":
		return metrics.adjusted_rand_score(y, pred)

	elif metric == "mi":
		return metrics.mutual_info_score(y, pred)

	elif metric == "adj_mi":
		return metrics.adjusted_mutual_info_score(y, pred)

	elif metric == "norm_mi":
		return metrics.normalized_mutual_info_score(y, pred)

	raise NotImplementedError(f"Unknown metric: {metric}")

	# if isinstance(value, (int, float)) or np.isscalar(value):
	# 	return f"{value:.2f}"

	# elif len(value) == 1:
	# 	return f"{value[0]:.2f}"

	# else:
	# 	return f"{np.mean(value):.2f} \u00B1 {np.std(value):.2f}"




def evaluate_clustering(data: T.List[Features], *args,
	markers, classes,
	clustering,
	output,
	metrics=["v-measure"],
	epsilons = [2.0, 0.5],
	n_runs: int = 1,
	plot: bool = False,
	rnd = None,
	**kwargs):

	fig, axs = None, None

	if clustering == "no":
		return fig, axs

	rows = []

	rnd = rnd or np.random.RandomState()

	if plot:
		if n_runs != 1:
			logging.warning(f"You set {n_runs=}, this will be ignored due to plotting!")
		n_runs = 1
		fig, axs = plt.subplots(nrows=len(epsilons), ncols=len(data), squeeze=False)

	headers = None

	for j, ds in enumerate(data):
		X, y = ds.get_data()
		_classes = np.unique(y)
		row = [ds._subset]
		_heads = ["Subset"]

		for i, eps in enumerate(epsilons):

			results = []
			for n in trange(n_runs):
				kwargs["random_state"] = rnd.randint(2*32-1)
				preds, clust_name, _X = predict(X, clustering,
					*args, n_classes=len(_classes), eps=eps, **kwargs)

				results.append(preds)

			if plot:
				ax = axs[i,j]
				if _X.shape[1] != 2:
					# X2d = UMAP(n_components=2).fit_transform(_X)
					X2d = TSNE(n_components=2).fit_transform(_X)
				else:
					X2d = _X
				for cls in np.unique(preds):
					c, m = markers.get(cls, ("k", "."))# if cls != -1 else

					# if cls not in markers:
					# 	print(cls)

					xy = X2d[preds==cls].T
					ax.scatter(*xy, c=c, marker=m)

				ax.set_title(f"{ds._subset} | {clust_name}")


			for metric in metrics:
				scores = [evaluate(metric, X, y, preds) for preds in results]

				_heads.append(f"{clust_name}\n{metric}")
				if n_runs != 1:
					std = np.std(scores)
					if np.isclose(std, 0):
						row.append(f"{np.mean(scores):.4f}")
					else:
						row.append(f"{np.mean(scores):.4f} \u00B1 {std:.4f}")

					_heads[-1] += f" ({n_runs=})"

				else:
					row.append(f"{scores[0]:.4f}")


		if headers is None:
			headers = _heads
		rows.append(row)

	tab = tabulate(rows,
		headers=headers,
		tablefmt="fancy_grid",
	)

	print(tab, file=output)

	return fig, axs



def main(args):

	if args.output is None:
		out = sys.stdout
	else:
		out = open(args.output, "a")
	print(args.embeddings, file=out)

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

	evaluate_knn(train, val, k=args.neighbors, output=out)


	kwargs = {}

	eps = [-1]
	if args.clustering == "KMeans":
		kwargs["init"] = "k-means++"
	elif args.clustering == "DBSCAN":
		eps = [
			# 0.05, 0.5, 1.0,
			1.0,
			2.0,
			3.0,
			# 50.0,
			# 100.0,
		]

	fig, axs = evaluate_clustering([train, val],
		clustering=args.clustering,
		classes=classes,
		epsilons=eps,
		n_runs=args.n_runs,
		metrics=args.metrics,
		markers=cols_markers,

		output=out,
		rnd=rnd,

		plot=args.plot,
		n_components=args.n_components,

		**kwargs
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
	Arg.int("--n_runs", default=1),
	Arg.int("--neighbors", "-k", default=5,
		help="Number of neighbors for k-NN"),

	Arg("--clustering", default="KMeans",
		choices=["KMeans", "GMM", "DBSCAN", "no"]
		),

	Arg("--metrics", nargs="+",
		default=[
			# "v-measure",
			"adj_rand_idx",
			"adj_mi",
		],
		choices=[
			"silhoette",
			"v-measure",
			"rand_idx",
			"adj_rand_idx",
			"mi",
			"adj_mi",
			"norm_mi",
		]),

	Arg("--output"),
	Arg.int("--n_components", "-nc", default=-1,
		help="If positive, perform dimensionality reduction to the number of given components"),
	Arg.flag("--plot"),

])


main(parser.parse_args())
