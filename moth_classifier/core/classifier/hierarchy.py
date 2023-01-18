import chainer
import numpy as np
import networkx as nx

from chainer import functions as F

from cvdatasets import Hierarchy

class HierarchyMixin:

	def __init__(self, *args,
		hierarchy: Hierarchy = None,
		**kwargs):
		super().__init__(*args, **kwargs)

		self.hierarchy = hierarchy


	def loss(self, pred: chainer.Variable, y: chainer.Variable) -> chainer.Variable:
		if self.hierarchy is None:
			return super().loss(pred, y)

		hc_y = self.hierarchy.embed_labels(y, xp=self.xp)
		loss_mask = self.hierarchy.loss_mask(y, xp=self.xp)

		hc_y[~loss_mask] = -1

		loss = F.sigmoid_cross_entropy(pred, hc_y,
			normalize=False, reduce="mean")

		return loss
