import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for newer models not yet in tiktoken's registry
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))