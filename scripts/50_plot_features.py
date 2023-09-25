#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import logging

from cvargparse import Arg
from cvargparse import BaseParser
from itertools import cycle as cycler
from matplotlib import pyplot as plt
from pathlib import Path

try:
	from cuml.manifold import TSNE
	from cuml.decomposition import PCA
	tsne_module = "CuML"
except ImportError:
	from sklearn.manifold import TSNE
	from sklearn.decomposition import PCA
	tsne_module = "scikit-learn"


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

def name_matches(name, name_filter) -> bool:
	return not name_filter or name_filter.lower() in name.lower()

class ZoomHandler:
	DEFAULT_DXY = 3

	def __init__(self, embeddings, labels, subset_mask, subsets, cols_markers,
		*, paths = None, root = None):
		self.x0, self.y0 = None, None
		self.x1, self.y1 = None, None

		self.embs = embeddings
		self.labels = labels
		self.subset_mask = subset_mask
		self.subsets = subsets
		self.cols_markers = cols_markers

		ncols = int(np.ceil(np.sqrt(len(subsets))))
		nrows = int(np.ceil(len(subsets) / ncols))

		self.paths, self.root = paths, root
		if paths is None:
			self.fig, self.axs = plt.subplots(nrows, ncols, squeeze=False)
			self.fig.suptitle("Zoom View")
		else:
			self.fig, self.axs = plt.subplots(nrows, ncols + 1, squeeze=False)
			self.fig.suptitle("Zoom and example View")
			self.fig.canvas.mpl_connect("button_release_event", self.show_example)


	def click(self, event):
		self.x0 = event.xdata
		self.y0 = event.ydata

	def release(self, event):

		self.x0, self.x1 = sorted([self.x0, event.xdata])
		self.y0, self.y1 = sorted([self.y0, event.ydata])

		dx, dy = abs(self.x0 - self.x1), abs(self.y0 - self.y1)
		if min(dx, dy) < ZoomHandler.DEFAULT_DXY:
			self.x0 -= ZoomHandler.DEFAULT_DXY / 2
			self.y0 -= ZoomHandler.DEFAULT_DXY / 2
			self.x1 = self.x0 + ZoomHandler.DEFAULT_DXY
			self.y1 = self.y0 + ZoomHandler.DEFAULT_DXY

		self.zoom()

	def get_ax(self, i):
		return self.axs[np.unravel_index(i, self.axs.shape)]

	def show_example(self, event):
		if self.paths is None:
			return
		x, y = event.xdata, event.ydata

		xs, ys = self.embs.T

		dists = (xs - x)**2 + (ys - y)**2
		idx = dists.argmin()
		im_path = self.root / self.paths[idx]["fname"]
		im = plt.imread(im_path)

		ax = self.get_ax(len(self.subsets))
		ax.cla()
		ax.axis("off")
		ax.set_title(im_path.parent.name)
		ax.imshow(im)
		self.fig.canvas.draw()


	def zoom(self):
		emb_x, emb_y = self.embs.T

		maskx = np.logical_and(emb_x >= self.x0, emb_x <= self.x1)
		# print(self.x0, self.x1, maskx.sum(), maskx.shape)
		masky = np.logical_and(emb_y >= self.y0, emb_y <= self.y1)
		# print(self.y0, self.y1, masky.sum(), masky.shape)

		# print(np.in1d(np.where(maskx)[0], np.where(masky)[0]))
		mask = np.logical_and(masky, maskx)

		X = self.embs[mask]
		labs = self.labels[mask]
		split = self.subset_mask[mask]

		for i, subset in enumerate(self.subsets):
			ax = self.get_ax(i)
			ax.cla()
			ax.set_title(f"Subset: {subset}")
			x = X[split == i]
			y = labs[split == i]
			classes_i = np.unique(y)

			for cls in classes_i:
				c, m = self.cols_markers[cls]
				x_cls = x[y == cls]

				ax.scatter(*x_cls.T, c=c, marker=m)

		self.fig.canvas.draw()
		# print(self.x0, self.y0, self.x1, self.y1)


def main(args):
	logging.info(f"Imported TSNE from {tsne_module}")

	content = np.load(args.features_file)
	mask, data, labels = [],  [],  []

	class_names = dict()

	if args.class_names:
		with open(args.class_names) as f:
			entries = [line.strip().partition(" ") for line in f]

		class_names = {int(id): name for (id, _, name) in entries}

	paths, root = None, None
	reordered_paths = []

	if args.image_paths:
		root = Path(args.image_paths).parent
		im_paths = np.loadtxt(args.image_paths, dtype=[("id", np.int32), ("fname", "U255")])
		split_ids = np.loadtxt(root / "tr_ID.txt", dtype=np.int32)

		paths = {split: im_paths[split_ids == split] for split in np.unique(split_ids)}
		root = root / "images"


	for i, subset in enumerate(args.subsets):
		X, y = content[f"{subset}/features"], content[f"{subset}/labels"]

		logging.info(f"=== Subset: {subset} ===")
		logging.info(f"Found {len(y):,d} samples {X.shape}")
		logging.info(f"with {len(np.unique(y)):,d} classes")

		data.extend(X)
		mask.extend(np.full(len(y), i, dtype=np.int32))
		labels.extend(y)

		if paths is not None:
			reordered_paths.extend(paths[i])


	mask, data, labels = map(np.array, [mask, data, labels])
	if paths is not None:
		paths = np.array(reordered_paths)

	classes = np.unique(labels)
	cols_markers = {cls: (c, m) for (cls, c, m) in zip(classes, colors, markers)}

	logging.info("=== Overall ===")
	logging.info(f"Found {len(labels):,d} samples")
	logging.info(f"with {len(classes):,d} classes")

	common_cls_mask = np.ones_like(classes, dtype=bool)
	for i, subset in enumerate(args.subsets):
		cls_i = np.unique(labels[mask == i])
		common_cls_mask &= np.in1d(classes, cls_i)

	if args.pca_dim > 2:
		data = PCA(n_components=args.pca_dim).fit_transform(data)

	# X_2d = PCA(n_components=2).fit_transform(data)
	X_2d = TSNE(n_components=2, method="fft", perplexity=args.perplexity).fit_transform(data)

	ncols = int(np.ceil(np.sqrt(len(args.subsets))))
	nrows = int(np.ceil(len(args.subsets) / ncols))
	fig, axs = plt.subplots(nrows, ncols, squeeze=False)
	fig.suptitle(args.features_file)

	handler = ZoomHandler(X_2d, labels, mask, args.subsets, cols_markers, paths=paths, root=root)
	fig.canvas.mpl_connect("button_press_event", handler.click)
	fig.canvas.mpl_connect("button_release_event", handler.release)

	legend_fig = None

	for i, subset in enumerate(args.subsets):
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.set_title(f"Subset: {subset}")
		x = X_2d[mask == i]
		y = labels[mask == i]

		classes_i = np.unique(y)

		common_labs = np.in1d(y, labels)

		has_labels = False
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
			label = f"{cls} | {name}"
			has_labels = True

			if name_matches(name, args.class_name_filter):
				ax.text(*x_cls.mean(axis=0), s=name,
					ha="center",
					va="center",
					backgroundcolor="#00808055")
			# else:
			# 	alpha = 0.1

			ax.scatter(*x_cls.T, c=c, marker=m, label=label, alpha=alpha)

		if has_labels and legend_fig is None:
			legend_fig = plt.figure()
			ax_lines, ax_labels = ax.get_legend_handles_labels()
			legend_fig.legend(ax_lines, ax_labels, ncol=3)

		if args.class_name_filter:
			ax_lines, ax_labels = ax.get_legend_handles_labels()
			idxs = [i for i, name in enumerate(ax_labels) if name_matches(name, args.class_name_filter)]
			ax_lines, ax_labels = [np.array(l)[idxs] for l in [ax_lines, ax_labels]]
			ax.legend(ax_lines, ax_labels)


	plt.tight_layout()
	plt.show()
	plt.close()


parser = BaseParser()

parser.add_args([
	Arg("features_file"),
	Arg("--subsets", nargs="+", default=["val", "train"]),
	Arg.float("--perplexity", "-perp", default=30),
	Arg("--class_names", "-names"),
	Arg("--image_paths", "-paths"),
	Arg("--class_name_filter", "-name_filter"),
	Arg.int("--pca_dim", "-pca", default=-1),
])

main(parser.parse_args())
