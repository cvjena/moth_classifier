import abc
import chainer
import numpy as np

from moth_classifier.core.prediction.accumulator import PredictionAccumulator

class BaseMetric(abc.ABC):
	ndim = 0 # we need this to "fool" the reporter's summary

	def __add__(self, num):
		return self.add(num)

	def __radd__(self, num):
		return self.add(num)


	def __mul__(self, num):
		return self.mul(num)

	def __rmul__(self, num):
		return self.mul(num)

	def __floordiv__(self, num):
		return self.div(num)

	def __rfloordiv__(self, num):
		return self.div(num)

	def __truediv__(self, num):
		return self.div(num)

	def __rtruediv__(self, num):
		return self.div(num)


	def __init__(self, accumulator: PredictionAccumulator, *,
		only_available: bool = True):
		super().__init__()
		self._accum = accumulator
		self._only_available = only_available

	def __call__(self, *args, **kwargs):
		return self

	def mul(self, other):
		return self

	def add(self, other):
		return self


	@abc.abstractmethod
	def div(self, other, **kwargs):
		return self._accum.calc_metrics(only_available = self._only_available, **kwargs)

class Precision(BaseMetric):

	def div(self, other):
		metrics = super().div(other)
		return metrics.get("precision", 0)

class Recall(BaseMetric):

	def div(self, other):
		metrics = super().div(other)
		return metrics.get("recall", 0)

class FScore(BaseMetric):

	def __init__(self, *args, beta: int = 1, **kwargs):
		super().__init__(*args, **kwargs)
		self._beta = beta

	def div(self, other):
		metrics = super().div(other, beta=self._beta)
		return metrics.get(f"f{self._beta}_score", 0)

