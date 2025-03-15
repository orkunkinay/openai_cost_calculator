import re

def parse_completion_usage(usage_string):
    """
    Parses a CompletionUsage string and extracts its details into a dictionary.
    
    Args:
        usage_string (str): The CompletionUsage string representation.
    
    Returns:
        dict: Parsed token usage details.
    
    Raises:
        ValueError: If the usage_string is not a non-empty string.
    """
    if not isinstance(usage_string, str) or not usage_string.strip():
        raise ValueError("usage_string must be a non-empty string")
        
    patterns = {
        "completion_tokens": r"completion_tokens=(\d+)",
        "prompt_tokens": r"prompt_tokens=(\d+)",
        "total_tokens": r"total_tokens=(\d+)",
        "accepted_prediction_tokens": r"accepted_prediction_tokens=(\d+)",
        "audio_tokens_completion": r"audio_tokens=(\d+)",  # within completion_tokens_details
        "reasoning_tokens": r"reasoning_tokens=(\d+)",
        "rejected_prediction_tokens": r"rejected_prediction_tokens=(\d+)",
        "audio_tokens_prompt": r"prompt_tokens_details=.*?audio_tokens=(\d+)",
        "cached_tokens": r"cached_tokens=(\d+)"
    }
    
    parsed_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, usage_string)
        if match:
            parsed_data[key] = int(match.group(1))
        else:
            # Option: Log or warn here if needed.
            parsed_data[key] = 0  # Default to 0 if pattern not found
    
    return parsed_data


def extract_model_details(model_string):
    """
    Extracts the model name and model date from the OpenAI model string.
    
    Args:
        model_string (str): The model string from ChatCompletionChunk (e.g., "gpt-4o-mini-2024-07-18").
    
    Returns:
        dict: A dictionary with 'model_name' and 'model_date'.
    
    Raises:
        ValueError: If model_string is not a valid non-empty string or if details cannot be extracted.
    """
    if not isinstance(model_string, str) or not model_string.strip():
        raise ValueError("model_string must be a non-empty string")
    
    model_match = re.match(r"^(.*)-(\d{4}-\d{2}-\d{2})$", model_string)
    if model_match:
        return {
            "model_name": model_match.group(1),
            "model_date": model_match.group(2)
        }
    else:
        raise ValueError(f"Could not extract model details from '{model_string}'")
