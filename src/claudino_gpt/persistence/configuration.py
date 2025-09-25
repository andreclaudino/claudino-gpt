import os
import shutil
import toml
from claudino_gpt.configurations.model_configuration import ModelConfiguration
from claudino_gpt.configurations.training_configuration import TrainingConfiguration


def load_model_configuration(file_path: str) -> ModelConfiguration:
    config_data = toml.load(file_path)   
    model_configuration = ModelConfiguration(**config_data)

    return model_configuration


def load_training_configuration(file_path: str) -> TrainingConfiguration:
    config_data = toml.load(file_path)   
    training_configuration = TrainingConfiguration(**config_data)

    return training_configuration


def copy_configuration_to_training_folder(
    model_configuration_path: str,
    training_configuration_path: str,
    training_folder_path: str
):
    os.makedirs(training_folder_path, exist_ok=True)

    shutil.copy(model_configuration_path, os.path.join(training_folder_path, "model.toml"))
    shutil.copy(training_configuration_path, os.path.join(training_folder_path, "training.toml"))
