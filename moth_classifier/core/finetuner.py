from chainer.training.updaters import StandardUpdater
from chainer_addons.training import MiniBatchUpdater
from cvfinetune import finetuner as ft
from cvdatasets import AnnotationArgs

from moth_classifier.core import classifier
from moth_classifier.core import dataset
from moth_classifier.core import annotation as annot


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
