import chainer.functions as F

from cvfinetune import classifier


def _unpack(var):
	return var[0] if isinstance(var, tuple) else var

def _mean(arrays):
	return F.mean(F.stack(arrays, axis=0), axis=0)

def get_classifier(opts):
	if opts.parts == "GLOBAL":
		return GlobalClassifier

	else:
		return PartsClassifier

class GlobalClassifier(classifier.Classifier):
	# no change is needed here
	pass

class PartsClassifier(classifier.SeparateModelClassifier):
	n_parts = 4

	@property
	def output_size(self):
		return self.n_parts * self.feat_size

	def _encode_parts(self, feats):
		n, t, feat_size = feats.shape

		# concat all features together
		return F.reshape(feats, (n, t*feat_size))

		# average over the t-dimension
		return F.mean(part_feats, axis=1)

	def __call__(self, X, parts, y):
		assert X.ndim == 4 and parts.ndim == 5 , \
			f"Dimensionality of inputs was incorrect ({X.ndim=}, {parts.ndim=})!"
		glob_pred = _unpack(self.separate_model(X, layer_name=self.layer_name))

		part_feats = []
		for part in parts.transpose(1,0,2,3,4):
			part_feat = _unpack(self.model(part,
				layer_name=self.model.meta.feature_layer))
			part_feats.append(part_feat)

		# stack over the t-dimension
		part_feats = F.stack(part_feats, axis=1)
		part_feats = self._encode_parts(part_feats)
		part_pred = self.model.fc(part_feats)

		glob_loss, glob_accu = self.loss(glob_pred, y), self.separate_model.accuracy(glob_pred, y)
		part_loss, part_accu = self.loss(part_pred, y), self.model.accuracy(part_pred, y)

		_mean_pred = _mean([F.softmax(glob_pred), F.softmax(part_pred)])
		accu = self.model.accuracy(_mean_pred, y)
		loss = _mean([part_loss, glob_loss])

		self.report(
			loss=loss,
			accuracy=accu,
			p_accu=part_accu,
			g_accu=glob_accu,
		)

		return loss
