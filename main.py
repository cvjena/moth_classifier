#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")
# ruff: noqa: E402

import chainer
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')

from cvfinetune.parser.utils import populate_args
from pathlib import Path

from moth_classifier import core
from moth_classifier.utils import parser



def main(args, experiment_name="Moth classifier"):

	if args.mode in ["evaluate", "extract"]:
		populate_args(args,
			ignore=[
				"mode", "load", "load_path", "gpu",
				"mpi", "n_jobs", "batch_size",
				"test_fold_id",
				"center_crop_on_val",
				"only_klass",
			],
			fc_params=[
				"fc/b",
				"fc8/b",
				"fc6/b",
				"wrapped/output/fc/b",
				"wrapped/output/fc2/b",
			]
		)

		if args.cross_dataset:
			args.dataset = args.cross_dataset

	chainer.set_debug(args.debug)

	MiB = 1024**2
	chainer.backends.cuda.set_max_workspace_size(512 * MiB)
	if args.debug:
		logging.warning("DEBUG MODE ENABLED!")

	args.dtype = np.empty(0, dtype=chainer.get_dtype()).dtype.name
	logging.info(f"Default dtype: {args.dtype}")


	tuner, comm = core.finetuner.new(args, experiment_name)
	tuner.profile_images()

	logging.info("Fitting size model, if possible")
	tuner.clf.fit_size_model(tuner.train_data)

	if args.mode == "train":
		tuner.run(opts=args,
			trainer_cls=core.Trainer
		)

	elif args.mode == "evaluate":

		dest_folder = Path(args.load).parent
		eval_fname = dest_folder / args.eval_output

		tuner.evaluate(eval_fname, force=args.force)

	elif args.mode == "extract":
		dest_folder = Path(args.load).parent
		if args.suffix:
			feats = dest_folder / f"features.{args.suffix}.npz"
		else:
			feats = dest_folder / "features.npz"

		tuner.extract_to(feats)
	else:
		raise NotImplementedError(f"mode not implemented: {args.mode}")

main(parser.parse_args())
