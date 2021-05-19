import argparse

from models import ObjectClassifier
from utils import assert_file_path, assert_folder_path, assert_newfile_path


def train_mode(args: argparse.Namespace):
    """Main execution function for when "train" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    assert_folder_path(args.dataset_path)
    if args.detector_model_path:
        assert_newfile_path(args.detector_model_path)
    if args.classifier_model_path:
        assert_newfile_path(args.classifier_model_path)
    if args.best_param_path:
        assert_newfile_path(args.best_param_path)
    classifier = ObjectClassifier(
        args.dataset_path, args.detector_model_path, args.classifier_model_path, args.best_param_path
    )
    classifier.train()


def pred_mode(args: argparse.Namespace):
    """Main execution function for when "pred" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    assert_folder_path(args.dataset_path)
    assert_file_path(args.detector_model_path)
    assert_file_path(args.classifier_model_path)
    classifier = ObjectClassifier(
        dataset_path=args.dataset_path,
        detector_detector_model_path=args.detector_model_path,
        classifier_model_path=args.classifier_model_path,
        is_training=False,
    )
    print(classifier.predict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    train_parser = subparser.add_parser("train")
    train_parser.add_argument(
        "dataset_path",
        help="Path to the folder containing image dataset for "
        "training the classifier (no need for training the object detector)",
    )
    train_parser.add_argument("--detector-model-path", help="Path for the trained object detection model")
    train_parser.add_argument("--classifier-model-path", help="Path for the trained classification model")
    train_parser.add_argument("--best-param-path", help="Path for the training parameters")
    train_parser.set_defaults(func=train_mode)

    pred_parser = subparser.add_parser("pred")
    pred_parser.add_argument("dataset_path", help="Path to the dataset for prediction")
    pred_parser.add_argument("detector_model_path", help="Path for loading the detector model")
    pred_parser.add_argument("classifier_model_path", help="Path for loading the classifier model")
    pred_parser.set_defaults(func=pred_mode)
    args = parser.parse_args()
    args.func(args)
