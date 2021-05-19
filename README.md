# Siemens HOLLZOF challenge - Object Classification

Performs person detection on an image and classifies each person in the frame as either wearing a mask or not.

## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.

In case the above method doesn't work, try running `pip install <package>==<version>` for each package with its version as listed in `requirements.txt`.

## **Usage:**
### Subcommands
    usage: main.py [-h] {train,pred} ...

    positional arguments:
    {train,pred}

    optional arguments:
    -h, --help    show this help message and exit

### Train Subcommand
    usage: main.py train [-h] [--detector-model-path DETECTOR_MODEL_PATH]
                        [--classifier-model-path CLASSIFIER_MODEL_PATH]
                        [--best-param-path BEST_PARAM_PATH]
                        dataset_path

    positional arguments:
    dataset_path          Path to the folder containing image dataset for
                            training the classifier (no need for training the
                            object detector)

    optional arguments:
    -h, --help            show this help message and exit
    --detector-model-path DETECTOR_MODEL_PATH
                            Path for the trained object detection model
    --classifier-model-path CLASSIFIER_MODEL_PATH
                            Path for the trained classification model
    --best-param-path BEST_PARAM_PATH
                            Path for the training parameters

### Predict Subcommand
    usage: main.py pred [-h] [--use-cuda-if-available]
                        dataset_path detector_model_path classifier_model_path
                        pred_json_path

    positional arguments:
    dataset_path          Path to the dataset for prediction
    detector_model_path   Path for loading the detector model
    classifier_model_path
                            Path for loading the classifier model
    pred_json_path        Path to the .json file to be created for prediction
                            outputs.

    optional arguments:
    -h, --help            show this help message and exit
    --use-cuda-if-available
                            The flag indicating whether to use a CUDA device if
                            available for prediction.
