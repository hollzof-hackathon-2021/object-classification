import os
from typing import List, Tuple

from imageai.Detection import ObjectDetection


class ObjectClassifier:
    def __init__(
        self,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
    ):
        self._dataset_path = dataset_path
        self._model_path = model_path
        self._best_param_path = best_param_path
        self._detector = ObjectDetection()
        self._detector.setModelTypeAsYOLOv3()
        if self._model_path is not None:
            self._detector.setModelPath(self._model_path)
        # TODO: add an image classifier for the classification.

    def _load_args(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)

    def train(
        self,
        dataset_path: str = None,
        model_path: str = None,
        best_param_path: str = None,
    ):
        self._load_args(
            _dataset_path=dataset_path,
            _model_path=model_path,
            _best_param_path=best_param_path,
        )
        if self._best_param_path is None:
            dataset_dir, dataset_file = os.path.split(self._dataset_path)
            self._best_param_path = os.path.join(dataset_dir, f"{dataset_file}.best_params")
        if self._model_path is not None:
            self._detector.loadModel()
        # TODO: write the rest.

    def evaluate(self, dataset_path: str = None, model_path: str = None):
        self._load_args(_dataset_path=dataset_path, _model_path=model_path)
        self._detector.loadModel()
        # TODO: write the rest.

    def predict(self, dataset_path: str = None, model_path: str = None) -> List[List[Tuple[Tuple[int, int, int, int], float]]]:
        """Performs prediction for each image in the dataset and for each
        image, it outputs a list of ((a,b,c,d), p) where a,b,c,d are bounds
        of a bounding box in pixels and p is the probability of wearing a mask.
        """
        self._load_args(_dataset_path=dataset_path, _model_path=model_path)
        self._detector.loadModel()
        # TODO: write the rest.
