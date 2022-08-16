import chainer
import numpy as np
import logging
import wandb

from chainer.backends.cuda import to_cpu
from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvdatasets import AnnotationArgs
from cvfinetune import finetuner as ft
from cvfinetune.training.extensions import WandbReport
from datetime import datetime as dt
from tqdm import tqdm

from moth_classifier.core import annotation as annot
from moth_classifier.core import classifier
from moth_classifier.core import dataset


class MothClassifierMixin:

	def read_annotations(self):
		args = AnnotationArgs(
			self.info_file,
			self.dataset_name,
			self.part_type,
			self.feature_model
		)

		self.annot = annot.MothAnnotations.new(args, load_strict=False)
		self.dataset_cls.label_shift = self._label_shift

	def extract_to(self, features_file):
		subsets = (("train", self.train_data), ("val", self.val_data))
		kwargs = dict(
			repeat=False,
			shuffle=False,
			n_jobs=self._n_jobs,
			batch_size=self._batch_size)

		converter = self.evaluator.converter
		device = self.device
		clf = self.clf

		data = dict()

		for subset, ds in subsets:
			it, n = self.new_iterator(ds, **kwargs)
			desc = f"{subset=}"
			feats, labs = np.zeros((len(ds), clf.output_size), dtype=np.float32), np.zeros(len(ds), dtype=np.int32)

			for i, batch in enumerate(tqdm(it, total=n, desc=desc)):
				X, y, *_ = converter(batch, device)
				i0 = i * self._batch_size

				with clf.eval_mode():
					f = clf.extract(X)

				f = to_cpu(chainer.as_array(f))
				y = to_cpu(chainer.as_array(y))
				feats[i0: i0 + len(f)] = f
				labs[i0: i0 + len(y)] = y

			data[f"{subset}/features"] = feats
			data[f"{subset}/labels"] = labs

		logging.info(f"Saving features to {features_file}")
		np.savez(features_file, **data)


class DefaultFinetuner(MothClassifierMixin, ft.DefaultFinetuner):
	pass

class MPIFinetuner(MothClassifierMixin, ft.MPIFinetuner):
	pass


def get_updater_params(opts):
	kwargs = dict()
	if opts.mode == "train" and opts.update_size > opts.batch_size:
		cls = MiniBatchUpdater
		kwargs["update_size"] = opts.update_size

	else:
		cls = StandardUpdater

	return dict(updater_cls=cls, updater_kwargs=kwargs)

def new(opts, experiment_name):

	tuner_factory = ft.FinetunerFactory(default=DefaultFinetuner, mpi_tuner=MPIFinetuner)

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
