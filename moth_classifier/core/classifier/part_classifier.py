import chainer.functions as F

from cvmodelz import classifiers

from moth_classifier.core.classifier.base import BaseClassifier

def _mean(arrays):
	return F.mean(F.stack(arrays, axis=0), axis=0)


class PartClassifier(BaseClassifier, classifiers.SeparateModelClassifier):
	n_parts = 4

	def __init__(self, concat_features, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._concat = concat_features


	def load_model(self, *args, finetune: bool = False, **kwargs):
		super().load_model(*args, finetune=finetune, **kwargs)

		if finetune:
			if self.copy_mode == "share":
				clf_name = self.model.clf_layer_name
				new_clf = L.Linear(self.model.meta.feature_size, self.n_classes)
				setattr(self.model, clf_name, new_clf)

			self.model.reinitialize_clf(self.n_classes, self.model.meta.feature_size)


	@property
	def output_size(self):
		if self._concat:
			return self.n_parts * self.feat_size

		return self.feat_size

	def _encode_parts(self, feats):
		if self._concat:
			# concat all features together
			n, t, feat_size = feats.shape
			return F.reshape(feats, (n, t*feat_size))

		# average over the t-dimension
		return F.mean(feats, axis=1)

	def extract(self, X, parts=None):
		glob_feat = self._get_features(X, self.model)

		if parts is None:
			return glob_feat, []

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat = self._get_features(part, self.separate_model)
			part_feats.append(part_feat)

		# stack over the t-dimension
		part_feats = F.stack(part_feats, axis=1)
		part_feats = self._encode_parts(part_feats)

		return glob_feat, part_feats


	def forward(self, X, parts, y, sizes=None):
		assert X.ndim == 4 and parts.ndim == 5 , \
			f"Dimensionality of inputs was incorrect ({X.ndim=}, {parts.ndim=})!"

		glob_feat, part_feats = self.extract(X, parts)

		glob_pred = self.model.clf_layer(glob_feat)
		part_pred = self.separate_model.clf_layer(part_feats)

		glob_loss, glob_accu = self.loss(glob_pred, y), self.model.accuracy(glob_pred, y)
		part_loss, part_accu = self.loss(part_pred, y), self.separate_model.accuracy(part_pred, y)

		_mean_pred = _mean([F.softmax(glob_pred), F.softmax(part_pred)])

		self.eval_prediction(_mean_pred, y)

		loss = _mean([part_loss, glob_loss])

		self.report(
			loss=loss,
			p_accu=part_accu,
			g_accu=glob_accu,
		)

		return loss

