import click
import torch

from claudino_gpt.model.claudino_gpt import ClaudinoGPT
from claudino_gpt.persistence.configuration import copy_configuration_to_training_folder, load_model_configuration,\
    load_training_configuration
from claudino_gpt.persistence.training import load_parquet_data
from claudino_gpt.training.optimization import create_optimizer
from claudino_gpt.training.training_loop import run_training_loop

FEATURES_COLUMN_NAME = "context"
NEXT_TOKEN_COLUMN_NAME = "next_token"

@click.command()
@click.option("--training-configuration-path", type=click.STRING)
@click.option("--model-configuration-path", type=click.STRING)
def main(training_configuration_path: str, model_configuration_path: str):
    training_configuration = load_training_configuration(training_configuration_path)
    model_configuration = load_model_configuration(model_configuration_path)
    copy_configuration_to_training_folder(
        model_configuration_path,
        training_configuration_path,
        training_configuration.output_path
    )

    torch.manual_seed(training_configuration.random_seed)
    
    model = ClaudinoGPT(model_configuration)
    
    train_dataset = load_parquet_data(
        training_configuration.train_source_path,
        features_column_name=FEATURES_COLUMN_NAME,
        label_column_name=NEXT_TOKEN_COLUMN_NAME,
        batch_size=training_configuration.batch_size,
        max_seq_length=model_configuration.context_legth
    )

    validation_dataset = load_parquet_data(
        training_configuration.validation_source_path,
        features_column_name=FEATURES_COLUMN_NAME,
        label_column_name=NEXT_TOKEN_COLUMN_NAME,
        batch_size=training_configuration.batch_size,
        max_seq_length=model_configuration.context_legth
    )
    
    optimizer = create_optimizer(model, training_configuration)

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    run_training_loop(model, train_dataset, validation_dataset, optimizer, device, training_configuration)


if __name__ == "__main__":
    main()
