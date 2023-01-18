from cvmodelz import classifiers

from moth_classifier.core.classifier.base import BaseClassifier
from moth_classifier.core.classifier.hierarchy import HierarchyMixin
from moth_classifier.core.classifier.size_model import SizeMixin


class Classifier(HierarchyMixin, SizeMixin,
	BaseClassifier, classifiers.Classifier):

	def forward(self, X, y, sizes=None):
		feat = self.extract(X)

		pred = self.predict(feat)
		loss = self.loss(pred, y)

		if self._use_size_model:
			pred = self.size_model(sizes, pred, y)
			size_loss = self.loss(pred, y)
			loss = self.loss_alpha * loss + (1 - self.loss_alpha) * size_loss

		self.eval_prediction(pred, y)

		self.report(loss=loss)

		return loss

