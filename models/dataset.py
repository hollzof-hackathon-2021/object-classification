import os
from typing import Callable, Dict, Optional

import numpy as np
import torch as th
from imageai.Detection import ObjectDetection
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


def identity_transform(x):
    return x


class DetectedObjectDataset(Dataset):
    def __init__(self, dataset_path: str, detector_model_path: str, device: th.device, transform: Optional[Callable] = None):
        super().__init__()
        self._dataset_path = dataset_path
        self._detector_model_path = detector_model_path
        self._device = device
        self._transform: Callable = identity_transform if transform is None else transform
        self._image_paths = tuple(item.name for item in os.scandir(self._dataset_path) if item.is_file())
        self._detector: Optional[ObjectDetection] = None
        self._custom_objects: Optional[Dict[str, str]] = None

    def _init_detector(self):
        self._detector = ObjectDetection()
        self._detector.setModelTypeAsYOLOv3()
        self._detector.setModelPath(self._detector_model_path)
        self._detector.loadModel(detection_speed="normal")
        self._custom_objects = self._detector.CustomObjects(person=True)

    def __getitem__(self, index: int) -> T_co:
        if self._detector is None:
            self._init_detector()
        image_pil = Image.open(os.path.join(self._dataset_path, self._image_paths[index])).convert("RGB")
        image_array = np.array(image_pil)
        _, detections = self._detector.detectCustomObjectsFromImage(
            custom_objects=self._custom_objects,
            input_type="array",
            input_image=image_array,
            output_type="array",
            minimum_percentage_probability=70,
        )
        return self._transform(image_pil), detections

    def __len__(self) -> int:
        return len(self._image_paths)
