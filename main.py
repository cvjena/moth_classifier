#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')

from cvfinetune.parser.utils import populate_args
from pathlib import Path

from moth_classifier.core import finetuner
from moth_classifier.core.training import Trainer
from moth_classifier.utils import parser



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

	args.dtype = np.empty(0, dtype=chainer.get_dtype()).dtype.name
	logging.info(f"Default dtype: {args.dtype}")


	tuner, comm = finetuner.new(args, experiment_name)
	tuner.profile_images()

	logging.info("Fitting size model, if possible")
	tuner.clf.fit_size_model(tuner.train_data)

	if args.mode == "train":
		tuner.run(opts=args,
			trainer_cls=Trainer
		)

	elif args.mode == "evaluate":

		dest_folder = Path(args.load).parent
		eval_fname = dest_folder / "evaluation.yml"

		tuner.evaluate(eval_fname, force=args.force)
	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")

main(parser.parse_args())
