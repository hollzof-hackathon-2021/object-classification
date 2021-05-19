import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import optuna
import torch as th
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from imageai.Detection import ObjectDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


def tensor_add_noise(x: th.Tensor) -> th.Tensor:
    return x + 0.01 * th.randn_like(x)


class ObjectClassifier:
    def __init__(
        self,
        dataset_path: str = None,
        detector_model_path: str = None,
        classifier_model_path: str = None,
        best_param_path: str = None,
        is_training: bool = True,
    ):
        self._dataset_path = dataset_path
        self._detector_model_path = detector_model_path
        self._classifier_model_path = classifier_model_path
        self._best_param_path = best_param_path
        self._is_training = is_training
        self._detector: Optional[ObjectDetection] = None
        self._classifier: Optional[EfficientNet] = None
        self._dataloader_train: Optional[DataLoader] = None
        self._dataloader_test: Optional[DataLoader] = None
        self._dataloader_validation: Optional[DataLoader] = None
        self._is_cuda_available = th.cuda.is_available()
        self._device = th.device("cuda" if self._is_cuda_available else "cpu")

    def _init_data_loaders(self, batch_size: int = 8):
        common_tfms_list = [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        common_tfms = transforms.Compose(common_tfms_list)
        if self._is_training:
            train_tfms_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
            ]
            train_tfms_list.extend(common_tfms_list)
            train_tfms_list.extend([transforms.Lambda(tensor_add_noise)])
            train_tfms = transforms.Compose(train_tfms_list)
            dataset_train = ImageFolder(os.path.join(self._dataset_path, "train"), transform=train_tfms)
            self._dataloader_train = DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=self._is_cuda_available,
                persistent_workers=True,
            )
            dataset_validation = ImageFolder(os.path.join(self._dataset_path, "validation"), transform=common_tfms)
            self._dataloader_validation = DataLoader(
                dataset_validation,
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=self._is_cuda_available,
                persistent_workers=True,
            )
            dataset_test = ImageFolder(os.path.join(self._dataset_path, "test"), transform=common_tfms)
        else:
            dataset_test = ImageFolder(self._dataset_path, transform=common_tfms)
        self._dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=self._is_cuda_available,
            persistent_workers=True,
        )

    def _init_classifier(self):
        if self._classifier_model_path is not None:
            self._classifier = EfficientNet.from_pretrained(
                "efficientnet-b0", weights_path=self._classifier_model_path, num_classes=2
            )
        else:
            self._classifier = EfficientNet.from_name("efficientnet-b0", num_classes=2)
        self._classifier.to(self._device)

    def _load_args(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)

    def _train_classifier(self, lr: float, early_stopping_epochs: int) -> float:
        self._init_classifier()
        epochs_since_improvements, min_val_loss = 0, float("inf")
        optimizer = th.optim.Adam(self._classifier.parameters(), lr=lr)
        for epoch in range(300):
            self._classifier.train()
            for batch_idx, (data, target) in enumerate(self._dataloader_train):
                data, target = data.to(self._device), target.to(self._device)
                optimizer.zero_grad()
                output = self._classifier(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
            # Perform validation and possibly early stopping.
            epochs_since_improvements += 1
            self._classifier.eval()
            val_loss = 0.0
            for batch_idx, (data, target) in enumerate(self._dataloader_validation):
                data, target = data.to(self._device), target.to(self._device)
                with th.no_grad():
                    output = self._classifier(data)
                    val_loss += F.nll_loss(output, target).item()
            val_loss /= len(self._dataloader_validation)
            if val_loss < min_val_loss:
                min_val_loss = min_val_loss
                epochs_since_improvements = 0
            if epochs_since_improvements >= early_stopping_epochs:
                break
        return val_loss

    def _get_classifier_best_params(self) -> Dict[str, Any]:
        def objective(trial: optuna.Trial):
            lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
            early_stopping_epochs = trial.suggest_int("early_stopping_epochs", 1, 10)
            return self._train_classifier(lr, early_stopping_epochs)

        if os.path.isfile(self._best_param_path):
            with open(self._best_param_path, "rb") as f:
                best_params: Dict[str, Any] = pickle.load(file=f)
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)
            with open(self._best_param_path, "wb") as f:
                pickle.dump(study.best_params, file=f)

        return best_params

    def train(
        self,
        dataset_path: str = None,
        detector_model_path: str = None,
        classifier_model_path: str = None,
        best_param_path: str = None,
    ):
        self._load_args(
            _dataset_path=dataset_path,
            _detector_model_path=detector_model_path,
            _classifier_model_path=classifier_model_path,
            _best_param_path=best_param_path,
        )
        self._init_data_loaders()
        if self._best_param_path is None:
            dataset_dir, dataset_file = os.path.split(self._dataset_path)
            self._best_param_path = os.path.join(dataset_dir, f"{dataset_file}.best_params")
        best_params = self._get_classifier_best_params()
        self._train_classifier(**best_params)
        th.save(self._classifier, self._classifier_model_path)

    def predict(
        self,
        dataset_path: str = None,
        detector_model_path: str = None,
        classifier_model_path: str = None,
    ) -> List[List[Tuple[Tuple[int, int, int, int], float]]]:
        """Performs prediction for each image in the dataset and for each
        image, it outputs a list of ((a,b,c,d), p) where a,b,c,d are bounds
        of a bounding box in pixels and p is the probability of wearing a mask.
        """
        self._load_args(
            _dataset_path=dataset_path,
            _detector_model_path=detector_model_path,
            _classifier_model_path=classifier_model_path,
        )
        self._init_data_loaders()
        # initialize detector
        self._detector = ObjectDetection()
        self._detector.setModelTypeAsYOLOv3()
        self._detector.setModelPath(self._detector_model_path)
        self._detector.loadModel()
        # initialize classifier
        self._init_classifier()
        # TODO: write the rest.
