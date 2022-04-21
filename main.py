#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import matplotlib
import numpy as np
import pyaml
matplotlib.use('Agg')

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.parser.utils import populate_args
from cvfinetune.training import Trainer
from pathlib import Path

from moth_classifier.core import classifier
from moth_classifier.core import dataset
from moth_classifier.utils import parser

def get_updater_params(opts):
	kwargs = dict()
	if opts.mode == "train" and opts.update_size > opts.batch_size:
		cls = MiniBatchUpdater
		kwargs["update_size"] = opts.update_size

	else:
		cls = StandardUpdater

	return dict(updater_cls=cls, updater_kwargs=kwargs)

def new_finetuner(opts, experiment_name):
	mpi = opts.mode == "train" and opts.mpi


	tuner_factory = FinetunerFactory.new(mpi=mpi)

	tuner = tuner_factory(
		opts=opts,
		experiment_name=experiment_name,
		manual_gc=True,

		**classifier.get_params(opts),
		**get_updater_params(opts),

		dataset_cls=dataset.Dataset,
		dataset_kwargs_factory=dataset.Dataset.kwargs(opts),
	)

	return tuner, tuner_factory.get("comm")

def main(args, experiment_name="Moth classifier"):

	if args.mode == "evaluate":
		populate_args(args,
			ignore=[
				"mode", "load", "gpu",
				"mpi", "n_jobs", "batch_size",
				"center_crop_on_val",
				"only_klass",
			],
			fc_params=[
				"model/fc/b",
				"model/fc6/b",
				"model/wrapped/output/fc/b",
			]
		)


	chainer.set_debug(args.debug)

	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	tuner, comm = new_finetuner(args, experiment_name)

	logging.info("Profiling the image processing: ")
	with tuner.train_data.enable_img_profiler():
		data = tuner.train_data
		data[np.random.randint(len(data))]

	if args.mode == "train":
		tuner.run(opts=args,
			trainer_cls=Trainer
		)

	elif args.mode == "evaluate":

		dest_folder = Path(args.load).parent
		eval_fname = dest_folder / "evaluation.yml"
		if eval_fname.exists() and not args.force:
			print(f"Evaluation file exists already, skipping \"{args.load}\"")
			return
		res = tuner.evaluator()
		res = {key: float(value) for key, value in res.items()}
		with open(eval_fname, "w") as f:
			pyaml.dump(res, f, sort_keys=False)

	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")

main(parser.parse_args())
