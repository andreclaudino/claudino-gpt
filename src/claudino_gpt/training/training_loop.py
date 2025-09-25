import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from claudino_gpt.configurations.training_configuration import TrainingConfiguration
from claudino_gpt.model.claudino_gpt import ClaudinoGPT
from claudino_gpt.persistence.model import save_claudino_gpt_model, cleanup_old_models
from claudino_gpt.training.metrics import calculate_batch_loss, calculate_dataset_loss,\
    create_tensorboard_writer, write_metric_to_tensorboard,\
    write_model_architecture_to_tensorboard


def run_training_loop(
    model: ClaudinoGPT,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    configuration: TrainingConfiguration,
):
    train_losses = []
    validation_losses = []
    track_token_seen = []

    tokens_seen = 0
    global_step = -1

    scaler = torch.GradScaler(device=device)

    tensorboard_folder_path = os.path.join(configuration.output_path, "tensorboard")
    os.makedirs(tensorboard_folder_path, exist_ok=True)
    tensorboard_writer = create_tensorboard_writer(tensorboard_folder_path)


    for epoch in range(configuration.number_of_epochs):
        model.to(device)
        model.train()

        print(f"Epoch {epoch + 1}/{configuration.number_of_epochs}")

        for (input_batch, target_batch) in tqdm(train_data_loader, desc=f"Training epoch #{epoch}"):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # type: ignore
                loss = calculate_batch_loss(input_batch, target_batch, model, device)
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()

                tokens_seen = tokens_seen + input_batch.size(0)
                global_step = global_step + 1

                if global_step % configuration.evaluation_frequency == 0:
                    train_loss, validation_loss = _evaluate_model(
                        model, train_data_loader, validation_data_loader, device, configuration.evaluation_iteration
                    )

                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    track_token_seen.append(tokens_seen)

                    write_metric_to_tensorboard("loss", "train", train_loss, global_step, tensorboard_writer)
                    write_metric_to_tensorboard("loss", "validation", validation_loss, global_step, tensorboard_writer)

                    tensorboard_writer.flush()

                if global_step % configuration.checkpoint_frequency == 0:
                    checkpoints_dir = os.path.join(configuration.output_path, "checkpoints")
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_step_{global_step}.pt")
                    save_claudino_gpt_model(model, optimizer, checkpoint_path)
                    
                    cleanup_old_models(checkpoints_dir, configuration.total_models_to_keep)

                write_model_architecture_to_tensorboard(model, input_batch, tensorboard_writer)

    return train_losses, validation_losses, track_token_seen



def _evaluate_model(
        model: ClaudinoGPT,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: str,
        evaluation_iteration,
):
    model.eval()

    with torch.no_grad():
        training_loss = calculate_dataset_loss(
            train_loader, model, device, evaluation_iteration,
        )

        validation_loss = calculate_dataset_loss(
            validation_loader, model, device, evaluation_iteration,
        )
    model.train()

    return training_loss, validation_loss


