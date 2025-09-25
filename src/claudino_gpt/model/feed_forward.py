from torch import nn

from claudino_gpt.configurations.model_configuration import ModelConfiguration

class FeedForwardBlock(nn.Module):

    def __init__(self, configuration: ModelConfiguration):
        super().__init__()

        self._sequential_layers = nn.Sequential(
            nn.Linear(
                configuration.embedding_dimension,
                configuration.embedding_multiplier_factor * configuration.embedding_dimension
            ),
            nn.GELU(),
            nn.Linear(
                configuration.embedding_multiplier_factor * configuration.embedding_dimension,
                configuration.embedding_dimension
            )
        )

    def forward(self, x):
        return self._sequential_layers(x)
