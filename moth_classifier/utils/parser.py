
from cvargparse import Arg
from cvargparse import BaseParser
from cvfinetune.parser import FineTuneParser

def add_evaluation_args(parser: BaseParser):
	parser.add_args([
		Arg("--force", action="store_true",
			help="forces evaluation even if the evaluation output file already exists"),
	], group_name="Evaluation arguments")


def add_extraction_args(parser: BaseParser):

	parser.add_args([
		Arg("--suffix",
			help="if set, this suffix will be appended to the features file"),
	], group_name="Feature extraction arguments")

def add_training_args(parser: BaseParser):

	parser.add_args([
		Arg("--update_size", type=int, default=-1,
			help="if positive, MiniBatchUpdater is used." + \
			"It accumulates gradients over the training and " + \
			"updates the weights only if the update size is exceeded"),

		Arg("--test_fold_id", type=int, default=0),

	], group_name="Training arguments")

	group = parser.add_mutually_exclusive_group()
	group.add_argument("--undersample", type=int, default=-1)
	group.add_argument("--oversample", type=int, default=-1)


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
		Arg.flag("--concat_features",
			help="If set, concatenate part features instead of averaging."),

		Arg.flag("--use_size_model", "-size_model",
			help="If set, incorporate a size model fitted on the training data."),

		Arg.float("--loss_alpha", default=0.5,
			help="Weight factor for the losses from the CNN and the size model."),

		Arg.float("--triplet_margin", "-margin", default=None,
			help="If set, then utilize the triplet loss. If >0, use miner considering this margin, if <=0 mine all triplets"),

		Arg.float("--triplet_alpha", default=0.01,
			help="Triplet loss weighting parameter"),

		Arg.flag("--triplet_only",
			help="Use only the triplet loss for training, ignore the CE loss"),

	], group_name="Model arguments")

	_common_parser.add_args([
		Arg("--cs_config",
			help="Config file with settings for the extraction of CS parts")
	], group_name="CS Parts")

	parser = subp.add_parser("train",
		help="Starts moth classifier training",
		parents=[_common_parser])

	add_training_args(parser)

	_common_parser.add_args([
		Arg("--cross_dataset", "-cross",
			help="if set, use this dataset instead of the dataset used during training"),
	], group_name="Dataset arguments")

	parser = subp.add_parser("evaluate",
		help="Starts moth classifier evaluation",
		parents=[_common_parser])

	add_evaluation_args(parser)

	parser = subp.add_parser("extract",
		help="Extract features for a trained model",
		parents=[_common_parser])

	add_extraction_args(parser)

	return main_parser.parse_args(args=args, namespace=namespace)
