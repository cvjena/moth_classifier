import numpy as np

from collections import defaultdict
from fve_layer.backends.chainer.links import GMMLayer


class SizeModel(GMMLayer):
	def __init__(self, n_classes, init_mu=0, init_sig=1, dtype=np.float32):
		super().__init__(
			in_size=1,
			n_components=n_classes,
			init_mu=init_mu,
			init_sig=init_sig)

	def fit(self, sizes, labels):

		sizes_per_class = defaultdict(list)
		for size, lab in zip(sizes, labels):
			sizes_per_class[int(lab)].append(size)

		n_classes = len(sizes_per_class)
		for cls_id, cls_sizes in sizes_per_class.items():
			cls_sizes = self.xp.array(cls_sizes)
			self.mu[:, cls_id] = self.xp.mean(cls_sizes)
			self.sig[:, cls_id] = self.xp.var(cls_sizes)
			self.w[cls_id] = len(cls_sizes) / n_classes

		no_cov = self.sig == 0
		self.sig[no_cov] = self.mu[no_cov] * 0.1
		self.sig[:] = np.maximum(self.sig, self.eps)

		self.w[:] = np.maximum(self.w, self.eps)
		self.w[:] /= self.w.sum()
