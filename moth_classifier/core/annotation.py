import logging
import numpy as np

from cvdatasets import FileListAnnotations
from cvdatasets.annotation.files import AnnotationFiles


class MothAnnotations(FileListAnnotations):


	def load_files(self, file_obj) -> AnnotationFiles:
		file_obj = super().load_files(file_obj)
		file_obj.load_files(px_per_mm=("px_per_mm.txt", False))
		return file_obj

	def parse_annotations(self):
		super().parse_annotations()
		self._parse_sizes()

	def _parse_sizes(self):
		if self.files.px_per_mm is None:
			logging.debug("Sizes were not loaded!")
			return

		px_per_mm = list(map(float, self.files.px_per_mm))
		self.px_per_mm = np.array(px_per_mm, dtype=np.float32)

	def scale(self, uuid):
		return self.px_per_mm[self.uuid_to_idx[uuid]]
