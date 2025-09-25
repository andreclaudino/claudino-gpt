from readable_number import ReadableNumber
import torch
from torch import nn

from claudino_gpt.configurations.model_configuration import ModelConfiguration
from claudino_gpt.model.transformer_block import TransformerBlock


class ClaudinoGPT(nn.Module):
    def __init__(self, configuration: ModelConfiguration) -> None:
        super().__init__()

        self._token_embedder = nn.Embedding(configuration.vocabulary_size, configuration.embedding_dimension)
        self._positional_embedder = nn.Embedding(configuration.context_legth, configuration.embedding_dimension)
        self._dropout_embedding = nn.Dropout(configuration.dropout_rate)

        transformer_blocks = [TransformerBlock(configuration) for _ in range(configuration.number_of_transformer_blocks)]
        self._transformer_blocks = nn.Sequential(*transformer_blocks)

        self._final_normalization = nn.LayerNorm(configuration.embedding_dimension)
        self._output_head = nn.Linear(
            configuration.embedding_dimension,
            configuration.vocabulary_size,
            bias=False
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        _, sequence_length = x.shape
        
        token_embeddings = self._token_embedder(x)
        positional_embeddings = self._positional_embedder(
            torch.arange(sequence_length, device=x.device).unsqueeze(0)
        )

        x = token_embeddings + positional_embeddings
        x = self._dropout_embedding(x)
        x = self._transformer_blocks(x)
        x = self._final_normalization(x)

        logits = self._output_head(x)

        return logits

    def get_number_of_parameters(self) -> int:
        """
        Returns the total number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_human_readable_number_of_parameters_human(self) -> str:
        """
        Returns the total number of trainable parameters in the model
        as human readable format.

        Returns:
            str: Number of trainable parameters as human readable format
        """
        number_of_parameters = self.get_number_of_parameters()
        human_readable = ReadableNumber(number_of_parameters, use_shortform=True)

        return str(human_readable)