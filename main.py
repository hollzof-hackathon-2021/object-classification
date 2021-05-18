import argparse

from models import ObjectClassifier
from utils import assert_file_path, assert_folder_path, assert_newfile_path


def train_mode(args: argparse.Namespace):
    """Main execution function for when "train" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    assert_folder_path(args.dataset_path)
    if args.model_path:
        assert_newfile_path(args.model_path)
    if args.best_param_path:
        assert_newfile_path(args.best_param_path)
    classifier = ObjectClassifier(args.dataset_path, args.model_path, args.best_param_path)
    classifier.train()


def eval_mode(args: argparse.Namespace):
    """Main execution function for when "eval" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    assert_folder_path(args.dataset_path)
    assert_file_path(args.model_path)
    classifier = ObjectClassifier(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
    )
    classifier.evaluate()


def pred_mode(args: argparse.Namespace):
    """Main execution function for when "pred" subcommand is used.

    Args:
      args: A Namespace object containing command line arguments.
    """
    assert_folder_path(args.dataset_path)
    assert_file_path(args.model_path)
    classifier = ObjectClassifier(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
    )
    print(classifier.predict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    train_parser = subparser.add_parser("train")
    train_parser.add_argument(
        "dataset_path",
        help="Path to the folder containing image dataset for training",
    )
    train_parser.add_argument("--model-path", help="Path for the trained model")
    train_parser.add_argument("--best-param-path", help="Path for the training parameters")
    train_parser.set_defaults(func=train_mode)

    eval_parser = subparser.add_parser("eval")
    eval_parser.add_argument("dataset_path", help="Path to the dataset for evaluation")
    eval_parser.add_argument("model_path", help="Path for loading the model")
    eval_parser.set_defaults(func=eval_mode)

    pred_parser = subparser.add_parser("pred")
    pred_parser.add_argument("dataset_path", help="Path to the dataset for prediction")
    pred_parser.add_argument("model_path", help="Path for loading the model")
    pred_parser.set_defaults(func=pred_mode)
    args = parser.parse_args()
    args.func(args)
