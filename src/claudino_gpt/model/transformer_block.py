from torch import nn

from claudino_gpt.configurations.model_configuration import ModelConfiguration
from claudino_gpt.model.feed_forward import FeedForwardBlock
from claudino_gpt.model.multi_head_attention import MultiHeadFlashAttention

class TransformerBlock(nn.Module):

    def __init__(self, configuration: ModelConfiguration):
        super().__init__()

        self._attention = MultiHeadFlashAttention(
            input_dimension = configuration.embedding_dimension,
            output_dimension=configuration.embedding_dimension,
            number_of_heads=configuration.number_of_attention_heads,
            context_length=configuration.context_legth,
            dropout_rate=configuration.dropout_rate,
            use_qkv_bias=configuration.use_qkv_bias
        )

        self._feed_forward = FeedForwardBlock(configuration)
        self._layer_normalization1 = nn.LayerNorm(configuration.embedding_dimension)
        self._layer_normalization2 = nn.LayerNorm(configuration.embedding_dimension)

        self._dropout_shortcut = nn.Dropout(configuration.dropout_rate)

    def forward(self, x):
        shortcut_connection = x

        x = self._layer_normalization1(x)
        x = self._attention(x)
        x = self._dropout_shortcut(x)
        x = x + shortcut_connection

        shortcut_connection = x
        x = self._layer_normalization2(x)
        x = self._feed_forward(x)
        x = self._dropout_shortcut(x)
        x = x + shortcut_connection

        return x

