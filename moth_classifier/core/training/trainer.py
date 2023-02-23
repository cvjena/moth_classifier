import chainer

from cvfinetune.training import Trainer as BaseTrainer

class Trainer(BaseTrainer):
	"""docstring for Trainer"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def reportables(self, args):

		# print_values, plot_values = super().reportables(args)

		print_values = [
			"elapsed_time",
			"epoch",

			"main/loss"
		] + ([
			"main/accu0"
			# "main/accu_s",
		] if args.use_size_model else []) + [
			"main/accu",
			"main/prec",
			"main/rec",
			"main/f1",
			"main/hprec",
			"main/hrec",
			"main/hf1",
		]

		print_values.extend([
			self.eval_name(val) for val in print_values if val.startswith("main/")]
		)

		plot_values = dict(
			loss=[
				"main/loss", self.eval_name("main/loss"),
			],
			accuracy=([
				"main/accu0", self.eval_name("main/accu0"),
				"main/accu_s", self.eval_name("main/accu_s"),
			] if args.use_size_model else []) + [
				"main/accu", self.eval_name("main/accu"),
			],
			prec_rec=[
				"main/prec", self.eval_name("main/prec"),
				"main/rec", self.eval_name("main/rec"),
				"main/f1", self.eval_name("main/f1"),
			],

		)

		return print_values, plot_values
