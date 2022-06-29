import chainer

from cvfinetune.training import Trainer as BaseTrainer

class Trainer(BaseTrainer):
	"""docstring for Trainer"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		ds = self.updater.get_iterator("main").dataset

		if isinstance(ds, chainer.datasets.SubDataset):
			ds = ds._dataset

		max_label = ds.labels.max() - ds.label_shift


	def reportables(self, args):

		print_values, plot_values = super().reportables(args)

		print_values.extend([
			"main/prec",
			"main/rec",
			"main/f1",
			self.eval_name("main/prec"),
			self.eval_name("main/rec"),
			self.eval_name("main/f1"),
		])

		return print_values, plot_values
