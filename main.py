#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging

from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune import classifier
from cvfinetune.finetuner import FinetunerFactory
from cvfinetune.training.trainer import SacredTrainer

from moth_classifier.core import dataset
from moth_classifier.utils import parser

def updater_setup(opts):
	if opts.mode == "train" and opts.update_size > opts.batch_size:
		updater_cls = MiniBatchUpdater
		updater_kwargs = dict(update_size=opts.update_size)

	else:
		updater_cls = StandardUpdater
		updater_kwargs = dict()

	return updater_cls, updater_kwargs


def main(args, experiment_name="Moth Classifier"):

	chainer.set_debug(args.debug)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	tuner_factory = FinetunerFactory.new(args)
	comm = tuner_factory.get("comm")

	updater_cls, updater_kwargs = updater_setup(args)

	tuner = tuner_factory(opts=args,
		classifier_cls=classifier.Classifier,
		classifier_kwargs={},

		model_kwargs=dict(pooling=args.pooling),

		dataset_cls=dataset.Dataset,
		dataset_kwargs_factory=None,

		updater_cls=updater_cls,
		updater_kwargs=updater_kwargs,
	)

	if args.mode == "train":
		tuner.run(opts=args,
			trainer_cls = SacredTrainer,

			sacred_params=dict(
				name=experiment_name,
				comm=comm,
				no_observer=args.no_sacred
			)
		)

	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")

main(parser.parse_args())
