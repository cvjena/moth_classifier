from cvfinetune.training import Trainer as BaseTrainer

from moth_classifier.core.training import extensions


class Trainer(BaseTrainer):
	"""docstring for Trainer"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		ds = self.updater.get_iterator("main").dataset
		max_label = ds.labels.max() - ds.label_shift
		# self.extend(extensions.EpochSummary(max_label + 1))


	def reportables(self, args):

		print_values, plot_values = super().reportables(args)

		print_values.extend(extensions.EpochSummary.summary_keys)

		return print_values, plot_values
