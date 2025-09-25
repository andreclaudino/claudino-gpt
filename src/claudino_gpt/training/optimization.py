import torch

from claudino_gpt.configurations.training_configuration import TrainingConfiguration
from claudino_gpt.model.claudino_gpt import ClaudinoGPT


def create_optimizer(model: ClaudinoGPT, configuration: TrainingConfiguration) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay
    )

    return optimizer