import abc

from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import IteratorMixin

from chainer_addons.dataset import AugmentationMixin
from chainer_addons.dataset import PreprocessMixin

class _unwrap(abc.ABC):

	def get_example(self, i):
		im_obj = super(_unwrap, self).get_example(i)
		im, parts, label = im_obj.as_tuple()

		return im, label + self.label_shift

class Dataset(
	# augmentation and preprocessing
	AugmentationMixin, PreprocessMixin,
	_unwrap,
	IteratorMixin, AnnotationsReadMixin):

	pass
