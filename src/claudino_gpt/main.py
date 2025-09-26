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
@click.option("--training-configuration-path", type=click.STRING, required=True)
@click.option("--model-configuration-path", type=click.STRING, required=True)
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
        max_seq_length=model_configuration.context_legth,
        data_loading_workers_count=training_configuration.data_loading_workers_count,
        epochs=training_configuration.number_of_epochs,
        random_seed=training_configuration.random_seed
    )

    validation_dataset = load_parquet_data(
        training_configuration.validation_source_path,
        features_column_name=FEATURES_COLUMN_NAME,
        label_column_name=NEXT_TOKEN_COLUMN_NAME,
        batch_size=training_configuration.batch_size,
        max_seq_length=model_configuration.context_legth,
        data_loading_workers_count=training_configuration.data_loading_workers_count,
        infinite=True,
        random_seed=training_configuration.random_seed
    )
    
    optimizer = create_optimizer(model, training_configuration)

    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on {device} ({device_name})")

    total_parameters = model.get_human_readable_number_of_parameters_human()
    print(f"Model has {total_parameters} parameters")
    
    run_training_loop(model, train_dataset, validation_dataset, optimizer, device, training_configuration)


if __name__ == "__main__":
    main()
