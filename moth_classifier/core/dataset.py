import chainer
import logging
import numpy as np

from chainercv import transforms as tr
from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import IteratorMixin
from cvdatasets.dataset import TransformMixin
from cvdatasets.utils import transforms as tr2

class Dataset(TransformMixin, IteratorMixin, AnnotationsReadMixin):
	label_shift = None

	@classmethod
	def kwargs(self, opts, subset):
		return dict(opts=opts)

	def __init__(self, *args, opts, prepare, center_crop_on_val, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)

		self.prepare = prepare
		# for these models, we need to scale from 0..1 to -1..1
		self.zero_mean = opts.model_type in ["inception", "inception_imagenet"],
		self._setup_augmentations(opts)

	def _setup_augmentations(self, opts):

		min_value, max_value = (0, 1) if self.zero_mean else (None, None)

		pos_augs = dict(
			random_crop=(tr.random_crop, dict(size=self._size)),

			center_crop=(tr.center_crop, dict(size=self._size)),

			random_flip=(tr.random_flip, dict(x_random=True, y_random=False)),

			random_rotate=(tr.random_rotate, dict()),

			color_jitter=(tr2.color_jitter, dict(
				brightness=opts.brightness_jitter,
				contrast=opts.contrast_jitter,
				saturation=opts.saturation_jitter,
				channel_order="BGR" if opts.swap_channels else "RGB",
				min_value=min_value,
				max_value=max_value,
			)),

		)

		logging.info("Enabled following augmentations in the training phase: " + ", ".join(opts.augmentations))

		self._train_augs = [pos_augs.get(aug) for aug in opts.augmentations]
		self._val_augs = []

		if opts.center_crop_on_val:
			logging.info("During evaluation, center crop is used!")
			self._val_augs.append(pos_augs["center_crop"])

	@property
	def augmentations(self):
		return self._train_augs if chainer.config.train else self._val_augs

	def transform(self, im_obj):

		im, parts, lab = self.preprocess(im_obj)
		im, parts = self.augment(im, parts)
		im, parts = self.postprocess(im, parts)

		if len(parts) == 0:
			return im, lab

		else:
			return im, parts, lab

		im, parts, label = im_obj.as_tuple()

		return im,


	def preprocess(self, im_obj):
		im, _, lab = im_obj.as_tuple()
		im = self.prepare(im, size=self.size)

		lab -= (self.label_shift or 0)

		parts = []

		if self._annot.part_type != "GLOBAL":
			for i, part in enumerate(im_obj.visible_crops(self.ratio)):
				part = self.prepare(part, size=self.part_size)
				parts.append(part)


		return im, parts, lab


	def augment(self, im, parts):

		for aug, params in self.augmentations:
			im = aug(im, **params)

		aug_parts = []
		for i, part in enumerate(parts):
			for aug, params in self.augmentations:

				# override the "default" size param
				if "size" in params:
					params = dict(params, size=self._part_size)

				part = aug(part, **params)
				aug_parts.append(part)

		return im, aug_parts

	def postprocess(self, im, parts):

		parts = np.array(parts, dtype=im.dtype)
		if self.zero_mean:
			# 0..1 -> -1..1
			im = im * 2 - 1
			parts = parts * 2 - 1

		return im, parts
