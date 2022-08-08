#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import typing as T
import numpy as np

from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.utils import sacred as sacred_utils
from matplotlib import pyplot as plt

class Setup(T.NamedTuple):
	use_size_model: bool
	oversample: int


class Factory(T.NamedTuple):
	dataset: str
	model_type: str

	@property
	def default_query(self):
		return {
			"experiment.name": "Moth classifier",
			"config.dataset": self.dataset,
			"config.model_type": self.model_type,

			"config.optimizer": "adam",

		}

	def __call__(self, setup: Setup):
		query = self.default_query

		query["config.use_size_model"] = setup.use_size_model
		query["config.oversample"] = setup.oversample

		return query

	def setup_to_label(self, setup: Setup, values: T.List[float]):
		return "\n".join([
			f"With{'' if setup.use_size_model else 'out'} size model",
			f"oversample: {setup.oversample}" if setup.oversample > 0 else "no oversample",
		])

def set_ylim(axs, ymin=None, ymax=None):

	if None in [ymin, ymax]:
		ymin, ymax = np.inf, -np.inf

		for ax in axs:
			_ymin, _ymax = ax.get_ylim()
			ymin, ymax = min(_ymin, ymin), max(_ymax, ymax)

	for ax in axs:
		ax.set_ylim(ymin, ymax)

def main(args):
	creds = sacred_utils.Experiment.get_creds()

	plotter = sacred_utils.SacredPlotter(creds)


	query_factory = Factory(args.dataset, args.model_type)

	params = dict(
		use_size_model=[False, True],
		oversample=[-1, 5, 10, 30, 50, 100],
	)

	metric_axs = {m: [] for m in args.metrics}

	for use_size_model in params["use_size_model"]:

		fig = plt.figure()
		axs = plotter.plot(
			metrics=args.metrics,

			setups=[
				Setup(use_size_model, oversample)
					for oversample in params["oversample"]
			],

			query_factory=query_factory,
			setup_to_label=query_factory.setup_to_label,

			include_running=args.include_running,
			showfliers=args.outliers,
		)

		for m, ax in zip(args.metrics, axs):
			metric_axs[m].append(ax)

	for m, axs in metric_axs.items():
		set_ylim(axs, args.ymin, args.ymax)


	metric_axs = {m: [] for m in args.metrics}

	for oversample in params["oversample"]:
		break

		fig = plt.figure()
		axs = plotter.plot(
			metrics=args.metrics,

			setups=[
				Setup(use_size_model, oversample)
					for use_size_model in params["use_size_model"]
			],

			query_factory=query_factory,
			setup_to_label=query_factory.setup_to_label,

			include_running=args.include_running,
			showfliers=args.outliers,
		)
		for m, ax in zip(args.metrics, axs):
			metric_axs[m].append(ax)

	for m, axs in metric_axs.items():
		set_ylim(axs, args.ymin, args.ymax)

	plt.show()
	plt.close()


args = BaseParser([
	Arg("--metrics", "-m", nargs="+", default=["accu"]),

	Arg("--dataset", "-ds", default="JENA_MOTHS_CROPPED_COMM"),
	Arg("--model_type", "-mt", default="cvmodelz.InceptionV3"),

	Arg.float("--ymin", "-ymin"),
	Arg.float("--ymax", "-ymax"),

	Arg.flag("--outliers"),
	Arg.flag("--include_running"),

])
main(args.parse_args())
