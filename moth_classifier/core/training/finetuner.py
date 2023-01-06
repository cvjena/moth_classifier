import chainer
import logging
import numpy as np
import wandb
import yaml

from chainer.backends.cuda import to_cpu
from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvdatasets import AnnotationArgs
from cvfinetune import finetuner as ft
from cvfinetune.training.extensions import WandbReport
from datetime import datetime as dt
from munch import munchify
from tqdm import tqdm

from cluster_parts import core
from cluster_parts import utils

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
			feats = np.zeros((len(ds), clf.output_size), dtype=np.float32)
			labs = np.zeros(len(ds), dtype=np.int32)

			logging.info(f"{subset} features: {feats.shape}")

			for i, batch in enumerate(tqdm(it, total=n, desc=desc)):
				X, y, *_ = converter(batch, device)
				i0 = i * self._batch_size

				with clf.eval_mode():
					f = clf.extract(X)

				f = to_cpu(chainer.as_array(f))
				y = to_cpu(chainer.as_array(y))
				y = ds._annot._orig_labels[y]

				feats[i0: i0 + len(f)] = f
				labs[i0: i0 + len(y)] = y

			data[f"{subset}/features"] = feats
			data[f"{subset}/labels"] = labs

		logging.info(f"Saving features to {features_file}")
		np.savez(features_file, **data)

	@property
	def dataset_kwargs_factory(self, *args, **kwargs):
		return self.patched_dataset_kwargs_factory

	@dataset_kwargs_factory.setter
	def dataset_kwargs_factory(self, func):
		self._dataset_kwargs_factory = func


	def patched_dataset_kwargs_factory(self, *args, **kwargs) -> dict:
		res = self._dataset_kwargs_factory(*args, **kwargs)
		config_file = self.config["cs_config"]
		assert config_file is not None, \
			"CS config file was not set!"

		with open(config_file) as f:
			opts = munchify(yaml.safe_load(f))

		extractor = core.BoundingBoxPartExtractor(
			corrector=core.Corrector(gamma=opts.gamma, sigma=opts.sigma),

			K=opts.n_parts,
			fit_object=opts.fit_object,

			thresh_type=opts.thresh_type,
			cluster_init=utils.ClusterInitType.MAXIMAS,
			feature_composition=opts.feature_composition,

		)

		return dict(res,
			extractor=extractor,
			cs=opts.classification_specific,
			model=self.model,
		)

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
		**dataset.get_params(opts),
	)

	return tuner, tuner_factory.get("comm")
