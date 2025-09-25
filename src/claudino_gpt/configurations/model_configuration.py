from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    vocabulary_size: int
    context_legth: int
    embedding_dimension: int
    number_of_attention_heads: int
    number_of_transformer_blocks: int
    dropout_rate: float
    use_qkv_bias: bool
    
    embedding_multiplier_factor: int