from transformers import AutoTokenizer


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Loads a tokenizer based on the provided name or path.
    
    Args:
        tokenizer_name (str): Path or name of the tokenizer to load
        
    Returns:
        AutoTokenizer: Loaded tokenizer instance
    """
    return AutoTokenizer.from_pretrained(tokenizer_name, resume_download=True)