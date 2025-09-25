from typing import Optional
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from claudino_gpt.model.claudino_gpt import ClaudinoGPT


def calculate_batch_loss(
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        model: ClaudinoGPT,
        device: str
) -> torch.Tensor:
    
    input_batch.to(device)
    target_batch.to(device)

    logits = model(input_batch)[:, -1, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss


def calculate_dataset_loss(
        data_loader: DataLoader,
        model: ClaudinoGPT,
        device: str,
        number_of_batches: Optional[int] = None
) -> float:
    
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif number_of_batches is None:
        number_of_batches = len(data_loader)
    else:
        number_of_batches = min(number_of_batches, len(data_loader))
    
    for batch_number, (input_batch, target_batch) in enumerate(data_loader):
        if batch_number < number_of_batches:
            loss = calculate_batch_loss(input_batch, target_batch, model, device)
            total_loss = loss.item()
        else:
            break

    return total_loss / number_of_batches


def create_tensorboard_writer(folder_path: str) -> SummaryWriter:
    return SummaryWriter(log_dir=folder_path)


def write_metric_to_tensorboard(scope: str, metric_name: str, metric_value: float, step: int, writer: SummaryWriter):
    metric_name_with_scope = f"{metric_name}/{scope}"
    writer.add_scalar(metric_name_with_scope, metric_value, step)


def write_model_architecture_to_tensorboard(model: ClaudinoGPT, batch: torch.Tensor, writer: SummaryWriter):
    model.eval()
    writer.add_graph(model, batch)
    model.train()
