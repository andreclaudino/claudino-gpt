import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from claudino_gpt.configurations.training_configuration import TrainingConfiguration
from claudino_gpt.model.claudino_gpt import ClaudinoGPT
from claudino_gpt.persistence.model import (
    save_training_state,
    load_latest_training_state,
)
from claudino_gpt.training.metrics import (
    calculate_batch_loss,
    calculate_dataset_loss,
    create_tensorboard_writer,
    write_metric_to_tensorboard,
)


def run_training_loop(
    model: ClaudinoGPT,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    configuration: TrainingConfiguration,
):
    # Diretórios
    checkpoints_dir = os.path.join(configuration.output_path, "checkpoints")
    tensorboard_folder_path = os.path.join(configuration.output_path, "tensorboard")
    os.makedirs(tensorboard_folder_path, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)


    # Carregar checkpoint mais recente, se existir
    loaded_step, loaded_tokens, success = load_latest_training_state(model, optimizer, configuration, device=device)
    if success and loaded_step is not None:
        last_step = loaded_step
        tokens_seen = loaded_tokens
        print(f"✅ Retomando do step {last_step} com {tokens_seen} tokens processados.")
    else:
        last_step = -1
        tokens_seen = 0
        print("🆕 Iniciando do zero.")

    # TensorBoard
    tensorboard_writer = create_tensorboard_writer(tensorboard_folder_path)

    # Escalonador para mixed-precision
    scaler = torch.GradScaler(device=device)

    model = model.to(device)    
    model.train()

    for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_data_loader)):

        # Pula batches já processados
        if batch_idx <= last_step:
            continue
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):  # type: ignore
            loss = calculate_batch_loss(input_batch, target_batch, model, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        tokens_seen += input_batch.size(0)

        # Avaliação periódica
        if batch_idx % configuration.evaluation_frequency == 0:
            train_loss, validation_loss = _evaluate_model(
                model, train_data_loader, validation_data_loader, device, configuration.evaluation_iteration
            )

            write_metric_to_tensorboard("loss", "train", train_loss, batch_idx, tensorboard_writer)
            write_metric_to_tensorboard("loss", "validation", validation_loss, batch_idx, tensorboard_writer)
            tensorboard_writer.flush()

        # Salvamento periódico
        if batch_idx % configuration.checkpoint_frequency == 0:
            save_training_state(model, optimizer, batch_idx, configuration, tokens_seen)

    save_training_state(model, optimizer, batch_idx, configuration, tokens_seen) # type: ignore



def _evaluate_model(
    model: ClaudinoGPT,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: str,
    evaluation_iteration,
):
    model.eval()
    with torch.no_grad():
        training_loss = calculate_dataset_loss(train_loader, model, device, evaluation_iteration)
        validation_loss = calculate_dataset_loss(validation_loader, model, device, evaluation_iteration)
    model.train()
    return training_loss, validation_loss