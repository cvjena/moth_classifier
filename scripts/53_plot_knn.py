#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import numpy as np
import re

from collections import defaultdict
from cvargparse import Arg
from cvargparse import BaseParser

from matplotlib import pyplot as plt

ACCU_REGEX = re.compile(r"(\d{1,3}\.\d{1,2})")
SETUP_REGEX = re.compile(r"\/adam\/([\w\-\.]+)_(no[\w\-\.]+)\/")
kNN_REGEX = re.compile(r"k(\d+)\.")

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
	"InceptionV3_imagenet_LR1e-3":						"InceptionV3 (ImageNet)",
	"InceptionV3_imagenet_labsmooth_LR1e-3":			"InceptionV3 (ImageNet) LS",
	"InceptionV3_inat_LR1e-3":							"InceptionV3 (iNat)",
	"InceptionV3_inat_labsmooth_LR1e-3":				"InceptionV3 (iNat) LS",
	"ResNet50_448px_LR1e-3":							"ResNet50 448px",
	"ResNet50_448px_labsmooth_LR1e-3":					"ResNet50 448px LS",
	"ResNet50_LR1e-3":									"ResNet50",
	"ResNet50_labsmooth_LR1e-3": 						"ResNet50 LS",
	"VGG19":											"VGG19",
	"VGG19_labSmooth":									"VGG19 LS",
	"chainercv2_InceptionResNetV1_LR1e-3":				"ch2.InceptionResNetV1",
	"chainercv2_InceptionResNetV1_labsmooth_LR1e-3":	"ch2.InceptionResNetV1 LS",
	"chainercv2_InceptionV3_LR1e-3":					"ch2.InceptionV3",
	"chainercv2_InceptionV3_labsmooth_LR1e-3":			"ch2.InceptionV3 LS",
	"chainercv2_ResNet50_LR1e-3":						"ch2.ResNet50",
	"chainercv2_ResNet50_labsmooth_LR1e-3":				"ch2.ResNet50 LS",
	"chainercv2_ResNext50-32x4d_LR1e-3":				"ch2.ResNext50-32x4d",
	"chainercv2_ResNext50-32x4d_labsmooth_LR1e-3":		"ch2.ResNext50-32x4d LS",
}

config2name = {
	"no_margin_alpha0.01": "Triplet, no margin",
	"no_triplet": "no Triplet",
}

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


	# print("\n".join(sorted(setups)))

	setups = sorted(setups)
	n = len(results)
	nrows = int(np.ceil(np.sqrt(n)))
	ncols = int(np.ceil(n / nrows))
	fig, axs = plt.subplots(nrows, ncols)
	for i, k in enumerate(sorted(results.keys())):
		result = results[k]
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.set_title(f"{k=}")

		width = 1 / len(configs)
		width_factor = 0.75
		min_accu = np.inf
		for offset, config in zip([-1, 1], configs):
			accus = [result.results[setup][config] for setup in setups]
			min_accu = min(min(accus), min_accu)
			xs = np.arange(len(accus))
			bars = ax.bar(xs + offset*(width - (1 - width_factor)/2) / 2,
				accus,
				width=width * width_factor,
				label=config2name[config])

			ax.bar_label(bars, rotation=90, label_type="center")

			ax.set_xticks(xs)
			ax.set_xticklabels([setup2name[s] for s in setups], rotation="-90")

		# ax.set_ylim(60, 90)
		ax.legend()
	plt.tight_layout()
	plt.show()
	plt.close()

parser = BaseParser([
	Arg("files", nargs="+"),

])


main(parser.parse_args())
