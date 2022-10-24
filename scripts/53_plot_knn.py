#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import re

from collections import defaultdict
from cvargparse import Arg
from cvargparse import BaseParser

from itertools import cycle as cycler
from matplotlib import pyplot as plt

ACCU_REGEX = re.compile(r"(\d{1,3}\.\d{1,2})")
SETUP_REGEX = re.compile(r"\/adam\/([\w\-\.]+)_(no[\w\-\.]+)\/")
kNN_REGEX = re.compile(r"k(\d+)\.")


colors = [
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
]

class ResultFile:

	def __init__(self, fname: str):
		super().__init__()

		with open(fname) as f:
			content = [line.strip() for line in f]


		self.results = defaultdict(list)
		self.configs = set()
		# first line is the setup, then comes the result
		for setup, accu in zip(content[::2], content[1::2]):
			accu_match = ACCU_REGEX.search(accu)
			assert accu_match is not None
			accu = float(accu_match.group(1))
			setup_match = SETUP_REGEX.search(setup)
			assert setup_match is not None
			model, config = setup_match.groups()

			self.results[model].append((config, accu))
			self.configs.add(config)

		for key, values in self.results.items():
			self.results[key] = dict(values)


setup2name = {
	"InceptionV3_imagenet_LR1e-3":						"IncV3 (ImNet)",
	"InceptionV3_imagenet_labsmooth_LR1e-3":			"IncV3 (ImNet) LS",
	"InceptionV3_inat_LR1e-3":							"IncV3 (iNat)",
	"InceptionV3_inat_labsmooth_LR1e-3":				"IncV3 (iNat) LS",
	"ResNet50_448px_LR1e-3":							"RN50 448px",
	"ResNet50_448px_labsmooth_LR1e-3":					"RN50 448px LS",
	"ResNet50_LR1e-3":									"RN50 224px",
	"ResNet50_labsmooth_LR1e-3": 						"RN50 224px LS",
	"VGG19":											"VGG19",
	"VGG19_labSmooth":									"VGG19 LS",
	"chainercv2_InceptionResNetV1_LR1e-3":				"ch2.IncRNV1",
	"chainercv2_InceptionResNetV1_labsmooth_LR1e-3":	"ch2.IncRNV1 LS",
	"chainercv2_InceptionV3_LR1e-3":					"ch2.IncV3",
	"chainercv2_InceptionV3_labsmooth_LR1e-3":			"ch2.IncV3 LS",
	"chainercv2_ResNet50_LR1e-3":						"ch2.RN50",
	"chainercv2_ResNet50_labsmooth_LR1e-3":				"ch2.RN50 LS",
	"chainercv2_ResNext50-32x4d_LR1e-3":				"ch2.RNx50",
	"chainercv2_ResNext50-32x4d_labsmooth_LR1e-3":		"ch2.RNx50 LS",
}

setup_groups = {

	("InceptionV3_inat_LR1e-3",
	"InceptionV3_inat_labsmooth_LR1e-3"): "IncV3 iNat",

	("InceptionV3_imagenet_LR1e-3",
	"InceptionV3_imagenet_labsmooth_LR1e-3"): "IncV3 ImNet",

	("ResNet50_LR1e-3",
	"ResNet50_labsmooth_LR1e-3"): "RN50 224px",

	("ResNet50_448px_LR1e-3",
	"ResNet50_448px_labsmooth_LR1e-3",): "RN50 448px",

	("VGG19",
	"VGG19_labSmooth"): "VGG19",

	("chainercv2_InceptionResNetV1_LR1e-3",
		"chainercv2_InceptionResNetV1_labsmooth_LR1e-3",): "ch2.IncRNv1",
	("chainercv2_InceptionV3_LR1e-3",
		"chainercv2_InceptionV3_labsmooth_LR1e-3",): "ch2.IncV3",
	("chainercv2_ResNet50_LR1e-3",
		"chainercv2_ResNet50_labsmooth_LR1e-3", ): "ch2.RN50",
	("chainercv2_ResNext50-32x4d_LR1e-3",
		"chainercv2_ResNext50-32x4d_labsmooth_LR1e-3"): "ch2.RNx50",
}

config2name = {
	"no_margin_alpha0.01": "Triplet, no margin",
	"no_triplet": "no Triplet",
}

def _rows_cols(n):
	nrows = int(np.ceil(np.sqrt(n)))
	ncols = int(np.ceil(n / nrows))
	return min(ncols, nrows), max(ncols, nrows)



def plot_bars(results, configs, setups):
	nrows, ncols = _rows_cols(len(results))
	fig, axs = plt.subplots(nrows, ncols)
	ks = sorted(results.keys())
	for i, k in enumerate(ks):
		res = results[k]
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.set_title(f"{k=}")

		width = 1 / len(configs)
		width_factor = 0.75

		for offset, config in zip([-1, 1], configs):
			accus = [res.results[setup][config] for setup in setups]
			xs = np.arange(len(accus))
			bars = ax.bar(xs + offset*(width - (1 - width_factor)/2) / 2,
				accus,
				width=width * width_factor,
				label=config2name[config])

			ax.bar_label(bars, rotation=90, fmt="%.1f", padding=5)

			ax.set_xticks(xs)
			ax.set_xticklabels([setup2name[s] for s in setups], rotation=90)

		ax.set_ylim(65, 90)
		ax.legend()


def plot_lines(results, configs, setups):

	nrows, ncols = _rows_cols(len(setup_groups))
	fig, axs = plt.subplots(nrows, ncols, squeeze=False)

	ks = sorted(results.keys())

	xtick_labels = [f"{k=}" for k in ks]

	for i, (group_setups, group_name) in enumerate(setup_groups.items()):
		ax = axs[np.unravel_index(i, axs.shape)]
		_setups = [setup for setup in setups if setup in group_setups]
		ax.set_title(group_name)

		for c, setup in enumerate(_setups):
			for config, linestyle in zip(configs, ["solid", "dashed"]):

				accus = [results[k].results[setup][config] for k in ks]
				xs = np.arange(len(accus))
				lab = f"{setup2name[setup]} ({config2name[config]})"

				ax.plot(xs, accus, label=lab, linestyle=linestyle, color=colors[c])

		ax.legend()
		ax.grid()
		ax.set_xticks(np.arange(len(xtick_labels)))
		ax.set_xticklabels(xtick_labels)

def main(args):
	configs = set()
	setups = set()

	results = {}
	for fname in args.files:
		knn_match = kNN_REGEX.search(fname)
		assert knn_match is not None
		k = int(knn_match.group(1))
		result = ResultFile(fname)
		results[k] = result

		setups |= result.results.keys()
		configs |= result.configs

	setups = sorted(setups)

	# plot_bars(results, configs, setups)
	plot_lines(results, configs, setups)

	plt.tight_layout()
	plt.show()
	plt.close()

parser = BaseParser([
	Arg("files", nargs="+"),

])


main(parser.parse_args())
