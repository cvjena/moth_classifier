from moth_classifier.core.classifier.base import Classifier
from moth_classifier.core.classifier.triplet_classifier import TripletClassifier
from moth_classifier.core.classifier.part_classifier import PartClassifier


def get_params(opts):
	model_kwargs = dict(pooling=opts.pooling)

	if hasattr(opts, "n_classes"):
		model_kwargs["n_classes"] = opts.n_classes

	kwargs = dict(
		only_head=opts.only_head,
		use_size_model=opts.use_size_model,
		loss_alpha=opts.loss_alpha,
	)

	if opts.parts == "GLOBAL":
		if opts.triplet_margin is not None:
			margin = opts.triplet_margin
			cls = TripletClassifier
			kwargs["margin"] = margin if margin > 0 else None
			kwargs["alpha"] = args.triplet_alpha

		else:
			cls = Classifier

	else:
		cls = PartClassifier
		kwargs["concat_features"] = opts.concat_features

	return dict(
		classifier_cls=cls,
		classifier_kwargs=kwargs,

		model_kwargs=model_kwargs,
	)


__all__ = [
	"Classifier",
	"PartClassifier",
	"get_params"
]
