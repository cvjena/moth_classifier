#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.training.trainer import SacredTrainer

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


def main(args, experiment_name="Moth Classifier"):

	chainer.set_debug(args.debug)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	tuner_factory = FinetunerFactory.new(args)
	comm = tuner_factory.get("comm")

	tuner = tuner_factory(opts=args,
		**classifier.get_params(args),

		model_kwargs=dict(pooling=args.pooling),

		dataset_cls=dataset.Dataset,
		dataset_kwargs_factory=dataset.Dataset.kwargs,

		**get_updater_params(args),
	)

	logging.info("Profiling the image processing: ")
	with tuner.train_data.enable_img_profiler():
		data = tuner.train_data
		data[np.random.randint(len(data))]

	if args.mode == "train":
		tuner.run(opts=args,
			trainer_cls=SacredTrainer,

			sacred_params=dict(
				name=experiment_name,
				comm=comm,
				no_observer=args.no_sacred
			)
		)

	elif args.mode == "evaluate":
		assert args.load != None, \
			"For the evaluation, load parameter must be set!"

		tuner.evaluate(trainer_cls=SacredTrainer)

	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")

main(parser.parse_args())
