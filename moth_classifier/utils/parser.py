
from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.parser import FineTuneParser

def add_evaluation_args(parser: BaseParser):
	pass

def add_training_args(parser: BaseParser):

	parser.add_args([
		Arg("--update_size", type=int, default=-1,
			help="if positive, MiniBatchUpdater is used." + \
			"It accumulates gradients over the training and " + \
			"updates the weights only if the update size is exceeded"),

		Arg("--mpi", action="store_true",
			help="Indicates that OpenMPI is used!"),

	], group_name="Training arguments")


	parser.add_args([
		Arg("--no_sacred", action="store_true",
			help="do save outputs to sacred"),

	], group_name="Sacred arguments")


def parse_args(args=None, namespace=None):
	main_parser = BaseParser()

	subp = main_parser.add_subparsers(
		title="Execution modes",
		dest="mode",
		required=True
	)
	_common_parser = FineTuneParser(add_help=False, nologging=True)

	_common_parser.add_args([
		Arg("--concat_features", action="store_true",
			help="If set, concatenate part features instead of averaging."),
	], group_name="Model arguments")

	parser = subp.add_parser("train",
		help="Starts moth classifier training",
		parents=[_common_parser])

	add_training_args(parser)

	parser = subp.add_parser("evaluate",
		help="Starts moth classifier evaluation",
		parents=[_common_parser])

	add_evaluation_args(parser)

	return main_parser.parse_args(args=args, namespace=namespace)
