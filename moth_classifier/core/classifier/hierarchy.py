import chainer
import numpy as np
import networkx as nx

from chainer import functions as F

from cvdatasets import Hierarchy

class HierarchyMixin:

	def __init__(self, *args,
		hierarchy: Hierarchy = None,
		**kwargs):
		self.hierarchy = hierarchy
		super().__init__(*args, **kwargs)


		# if self.hierarchy is not None:
		# 	import matplotlib.pyplot as plt
		# 	fig, axs = plt.subplots()
		# 	G = self.hierarchy.graph
		# 	pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
		# 	nx.draw(G, pos, with_labels=True)

		# 	plt.show()
		# 	plt.close()


	def loss(self, pred: chainer.Variable, y: chainer.Variable) -> chainer.Variable:
		if self.hierarchy is None:
			return super().loss(pred, y)

		hc_y = self.hierarchy.embed_labels(y, xp=self.xp)
		loss_mask = self.hierarchy.loss_mask(y, xp=self.xp)

		hc_y[~loss_mask] = -1

		loss = F.sigmoid_cross_entropy(pred, hc_y,
			normalize=False, reduce="mean")

		return loss

	def fuse_prediction(self, glob_pred, part_pred):
		if self.hierarchy is None:
			return super().fuse_prediction(glob_pred, part_pred)

		import pdb; pdb.set_trace()

	def accuracy(self, pred, gt):
		if self.hierarchy is None:
			return super().accuracy(pred, gt)

		pred = chainer.cuda.to_cpu(chainer.as_array(F.sigmoid(pred)))
		deembed_pred_dist = self.hierarchy.deembed_dist(pred)
		dim = self.hierarchy.orig_lab_to_dimension.get

		# argmax and select the correct dimension
		predictions = np.array([
			dim(sorted(dist, key=lambda x: x[1], reverse=True)[0][0])
				for dist in deembed_pred_dist
		])


		# transform GT labels to original label uid
		hc_orig_gt = list(map(self.hierarchy.label_transform, gt))
		# ... and to the according dimension
		hc_idxs_gt = np.array(list(map(dim, hc_orig_gt)))

		return (predictions == hc_idxs_gt).mean()

		import pdb; pdb.set_trace()



		count = 0
		for sample_pred, sample_gt in zip(pred.argmax(axis=1), hc_idxs):
			count += int(sample_pred == sample_gt)

		return count / len(gt)

		root = next(nx.topological_sort(self.hierarchy.graph))
		hc_y = self.hierarchy.embed_labels(gt, xp=np)
		xs, ys = np.where(hc_y)

		results = [
			[[], -1, -1]
			for _ in range(len(gt))
		]

		count = 0
		for x, y in zip(xs, ys):
			sample_pred = pred[x]
			sample_gt = hc_idxs[x]

			if y != dim(root):
				results[x][0].append(y)
			results[x][1] = sample_gt
			results[x][2] = sample_pred.argmax()

			count += int(sample_gt == sample_pred.argmax())

		# count = 0
		# for hc_gt, sample_gt, sample_pred in results:
		# 	count += int(sample_gt == sample_pred)

		return count / len(gt)



	def predict_dist(self, features, *, model=None):
		pred = self.predict(features, model=model)
		if self.hierarchy is None:
			return pred

		return self.hierarchy.deembed_dist(F.sigmoid(pred))

	# def predict_dist(self, feature_batch):
	#     embedded_predictions = self.predict_embedded(feature_batch).numpy()
	#     return self.deembed_dist(embedded_predictions)
