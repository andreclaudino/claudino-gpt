from torch import nn


class MultiHeadFlashAttention(nn.Module):
    def __init__(
            self,
            input_dimension: int,
            output_dimension: int,
            number_of_heads: int,
            context_length: int,
            dropout_rate: float,
            use_qkv_bias: bool
        ):
        super().__init__()

        assert output_dimension % number_of_heads == 0, "d_out is indivisible by num_heads"

        self._number_of_heads = number_of_heads
        self._context_length = context_length
        self._heads_dimension = output_dimension // number_of_heads
        self._output_dimension = output_dimension

        self._qkv = nn.Linear(input_dimension, 3 * output_dimension, bias=use_qkv_bias)
        self._projection = nn.Linear(output_dimension, output_dimension)
        self._dropout_rate = dropout_rate

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        qkv = self._qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self._number_of_heads, self._heads_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv
        dropout_rate = 0. if not self.training else self._dropout_rate
        context_vector = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=dropout_rate, is_causal=True)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, num_tokens, self._output_dimension)
        context_vector = self._projection(context_vector)

        return context_vector