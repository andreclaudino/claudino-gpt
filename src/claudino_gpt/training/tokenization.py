import torch
from transformers import AutoTokenizer

from claudino_gpt.model.claudino_gpt import ClaudinoGPT


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Loads a tokenizer based on the provided name or path.
    
    Args:
        tokenizer_name (str): Path or name of the tokenizer to load
        
    Returns:
        AutoTokenizer: Loaded tokenizer instance
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)

def generate_encoded_sentence(
        model: ClaudinoGPT,
        encoded_sentence: torch.Tensor,
        max_new_tokens: int,
        context_length: int
) -> torch.Tensor:
    """
    Generates text using the provided model and encoded tensor.
    Args:
        model (ClaudinoGPT): The GPT model to use for generating text
        encoded_tensor (torch.Tensor): Encoded input tensor
        max_new_tokens (int): Maximum number of tokens to generate
        context_length (int): Length of the context in the input tensor
        
        return Generated text
    """
    
    for _ in range(max_new_tokens):
        index_condition = encoded_sentence[:, -context_length:]

        with torch.no_grad():
            logits = model(index_condition)

        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        index_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        encoded_sentence = torch.cat((encoded_sentence, index_next), dim=-1)

    return encoded_sentence

def generate_text(
    model: ClaudinoGPT,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    tokenizer: AutoTokenizer
) -> str:
    """
    Gera texto a partir de um prompt em string, codificando-o com o tokenizer
    e ajustando o comprimento para `context_length` com left-padding ou truncamento.
    
    Padding é feito no início da sequência usando:
        pad_token_id = tokenizer.pad_token_id if defined, else tokenizer.unk_token_id
    """
    # Codifica o prompt em IDs de tokens
    encoded = tokenizer.encode(prompt, add_special_tokens=False)  # type: ignore # lista de inteiros

    # Converte para tensor
    encoded_tensor = torch.tensor(encoded, dtype=torch.long)

    # Determina o token de padding com fallback seguro
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id # type: ignore
    if pad_token_id is None:
        # Caso extremo: nem pad_token_id nem unk_token_id estão definidos
        pad_token_id = 0  # fallback absoluto (comum em muitos tokenizers)

    # Ajusta o comprimento para context_length
    if encoded_tensor.size(0) > context_length:
        # Trunca do início: mantém os últimos `context_length` tokens
        encoded_tensor = encoded_tensor[-context_length:]
    else:
        # Preenche no início (left-padding)
        pad_length = context_length - encoded_tensor.size(0)
        pad_tensor = torch.full((pad_length,), pad_token_id, dtype=torch.long)
        encoded_tensor = torch.cat([pad_tensor, encoded_tensor], dim=0)

    # Adiciona dimensão de batch: (1, context_length)
    encoded_tensor = encoded_tensor.unsqueeze(0)

    # Gera nova sequência
    generated_tensor = generate_encoded_sentence(
        model=model,
        encoded_sentence=encoded_tensor,
        max_new_tokens=max_new_tokens,
        context_length=context_length
    )

    # Decodifica para string
    generated_ids = generated_tensor.squeeze(0).tolist()
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True) # type: ignore

    return decoded_text