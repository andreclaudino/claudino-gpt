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
    encoded_sentence: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    tokenizer: AutoTokenizer
) -> str:
    encoded_sentence = generate_encoded_sentence(model, encoded_sentence, max_new_tokens, context_length)
    encoded_sentence_list = encoded_sentence.squeeze(0).tolist()
    decoded_sentence = tokenizer.decode(encoded_sentence_list) # type: ignore
    
    return decoded_sentence