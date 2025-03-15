def calculate_cost(usage_data, pricing_rates_text, pricing_rates_audio):
    """
    Calculates the cost of API usage based on both text and audio token types.
    
    Args:
        usage_data (dict): Token usage details.
        pricing_rates_text (dict or None): Pricing details for text tokens.
        pricing_rates_audio (dict or None): Pricing details for audio tokens.
    
    Returns:
        dict: Cost breakdown and total cost in dollars.
    
    Raises:
        ValueError: If usage_data is not a dictionary or missing required keys,
                    or if required pricing fields are missing.
    """
    if not isinstance(usage_data, dict):
        raise ValueError("usage_data must be a dictionary")
    
    # Check for expected keys in usage_data
    required_keys = ["prompt_tokens", "completion_tokens", "audio_tokens_prompt", "audio_tokens_completion"]
    for key in required_keys:
        if key not in usage_data:
            raise ValueError(f"Missing expected key '{key}' in usage_data")
    
    # Extract token counts
    prompt_tokens = usage_data.get("prompt_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens", 0)
    audio_tokens_prompt = usage_data.get("audio_tokens_prompt", 0)
    audio_tokens_completion = usage_data.get("audio_tokens_completion", 0)
    
    total_cost = 0
    cost_breakdown = {}
    
    # Calculate cost for Text tokens if pricing provided
    if pricing_rates_text is not None:
        if "input_price" not in pricing_rates_text or "output_price" not in pricing_rates_text:
            raise ValueError("Missing required pricing fields for text tokens in pricing_rates_text")
        
        text_input_cost = (prompt_tokens / 1_000_000) * pricing_rates_text["input_price"]
        text_output_cost = (completion_tokens / 1_000_000) * pricing_rates_text["output_price"]
        text_total_cost = text_input_cost + text_output_cost
        
        cost_breakdown["text_input_cost"] = format(text_input_cost, ".8f")
        cost_breakdown["text_output_cost"] = format(text_output_cost, ".8f")
        cost_breakdown["text_total_cost"] = format(text_total_cost, ".8f")
        
        total_cost += text_total_cost

    # Calculate cost for Audio tokens if pricing provided
    if pricing_rates_audio is not None:
        if "input_price" not in pricing_rates_audio or "output_price" not in pricing_rates_audio:
            raise ValueError("Missing required pricing fields for audio tokens in pricing_rates_audio")
        
        audio_input_cost = (audio_tokens_prompt / 1_000_000) * pricing_rates_audio["input_price"]
        audio_output_cost = (audio_tokens_completion / 1_000_000) * pricing_rates_audio["output_price"]
        audio_total_cost = audio_input_cost + audio_output_cost
        
        cost_breakdown["audio_input_cost"] = format(audio_input_cost, ".8f")
        cost_breakdown["audio_output_cost"] = format(audio_output_cost, ".8f")
        cost_breakdown["audio_total_cost"] = format(audio_total_cost, ".8f")
        
        total_cost += audio_total_cost

    cost_breakdown["total_cost"] = format(total_cost, ".8f")
    
    return cost_breakdown