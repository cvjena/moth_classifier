#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np

from cvargparse import Arg
from cvargparse import BaseParser
from itertools import cycle as cycler
from matplotlib import pyplot as plt
try:
	from cuml.manifold import TSNE
except ImportError:
	from sklearn.manifold import TSNE

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

		data.extend(X)
		mask.extend(np.full(len(y), i, dtype=np.int32))
		labels.extend(y)

	mask, data, labels = map(np.array, [mask, data, labels])
	classes = np.unique(labels)

	cols_markers = {cls: (c, m) for (cls, c, m) in zip(classes, colors, markers)}
	X = np.stack(data, axis=0)

	tsne = TSNE(n_components=2)
	X_2d = tsne.fit_transform(X)

	for i, subset in enumerate(args.subsets):

		fig, ax = plt.subplots()
		ax.set_title(f"Subset: {subset}")
		x = X_2d[mask == i]
		y = labels[mask == i]

		for cls in np.unique(y):
			c, m = cols_markers[cls]

			x_cls = x[y == cls]

			ax.scatter(*x_cls.T, c=c, marker=m)

			if cls not in class_names:
				continue
			name = class_names[cls]

			if args.class_name_filter and args.class_name_filter.lower() not in name.lower():
				continue

			ax.text(*x_cls.mean(axis=0), s=name,
				ha="center",
				va="center",
				backgroundcolor="#00808055")



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
