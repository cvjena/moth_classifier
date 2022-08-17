#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import logging

from cvargparse import Arg
from cvargparse import BaseParser
from itertools import cycle as cycler
from matplotlib import pyplot as plt

try:
	from cuml.manifold import TSNE
	from cuml.decomposition import PCA
except ImportError:
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA

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

def main(args):
	content = np.load(args.features_file)
	mask, data, labels = [],  [],  []

	class_names = dict()

	if args.class_names:
		with open(args.class_names) as f:
			entries = [line.strip().partition(" ") for line in f]

		class_names = {int(id): name for (id, _, name) in entries}


	for i, subset in enumerate(args.subsets):
		X, y = content[f"{subset}/features"], content[f"{subset}/labels"]

		logging.info(f"=== Subset: {subset} ===")
		logging.info(f"Found {len(y):,d} samples")
		logging.info(f"with {len(np.unique(y)):,d} classes")

		data.extend(X)
		mask.extend(np.full(len(y), i, dtype=np.int32))
		labels.extend(y)

	mask, data, labels = map(np.array, [mask, data, labels])
	classes = np.unique(labels)
	cols_markers = {cls: (c, m) for (cls, c, m) in zip(classes, colors, markers)}

	logging.info(f"=== Overall ===")
	logging.info(f"Found {len(labels):,d} samples")
	logging.info(f"with {len(classes):,d} classes")

	common_cls_mask = np.ones_like(classes, dtype=bool)
	for i, subset in enumerate(args.subsets):
		cls_i = np.unique(labels[mask == i])
		common_cls_mask &= np.in1d(classes, cls_i)

	# X_2d = PCA(n_components=2).fit_transform(data)
	X_2d = TSNE(n_components=2).fit_transform(data)

	ncols = int(np.ceil(np.sqrt(len(args.subsets))))
	nrows = int(np.ceil(len(args.subsets) / ncols))
	fig, axs = plt.subplots(nrows, ncols, squeeze=False)
	fig.suptitle(args.features_file)

	for i, subset in enumerate(args.subsets):
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.set_title(f"Subset: {subset}")
		x = X_2d[mask == i]
		y = labels[mask == i]

		classes_i = np.unique(y)

		common_labs = np.in1d(y, labels)

		for cls in classes_i:
			c, m = cols_markers[cls]

			x_cls = x[y == cls]

			alpha = 1.0
			if not np.all(common_cls_mask) and cls in classes[common_cls_mask]:
				alpha = 0.1

			if cls not in class_names:
				ax.scatter(*x_cls.T, c=c, marker=m, alpha=alpha)
				continue



			name = class_names[cls]
			if args.class_name_filter and args.class_name_filter.lower() not in name.lower():
				ax.scatter(*x_cls.T, c=c, marker=m, alpha=alpha)
			else:
				ax.scatter(*x_cls.T, c=c, marker=m, label=name, alpha=alpha)
				ax.text(*x_cls.mean(axis=0), s=name,
					ha="center",
					va="center",
					backgroundcolor="#00808055")

		if args.class_name_filter:
			ax.legend()


	plt.show()
	plt.close()


parser = BaseParser()

parser.add_args([
	Arg("features_file"),
	Arg("--subsets", nargs="+", default=["train"]),
	Arg("--class_names", "-names"),
	Arg("--class_name_filter", "-name_filter"),
])

main(parser.parse_args())
