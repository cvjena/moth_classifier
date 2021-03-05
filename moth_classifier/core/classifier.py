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
		# average over the t-dimension
		part_pred = self.model.fc(F.mean(part_feats, axis=1))

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
