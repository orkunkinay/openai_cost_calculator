from .pricing import load_pricing
from .usage_parser import parse_completion_usage, extract_model_details
from .cost_calculator import calculate_cost

class CostEstimator:
    def __init__(self):
        """Loads pricing data from the package's internal data folder."""
        self.pricing = load_pricing()

    def estimate_cost(self, response):
        """
        Estimate cost by extracting model details, determining token type, 
        and applying the appropriate pricing for both Text and Audio tokens.
        
        This method supports either a streaming response (an iterable of 
        ChatCompletionChunk objects) or a single ChatCompletion object.
        
        Args:
            response (iterable or ChatCompletion): The response from OpenAI.
        
        Returns:
            dict: Cost breakdown including both text and audio cost components.
            
        Raises:
            ValueError: If response is empty, None, or does not contain valid data.
            TypeError: If response items do not have the required attributes.
        """
        # Ensure response is not None
        if response is None:
            raise ValueError("Response is None. Expected a ChatCompletion or an iterable of ChatCompletionChunk objects.")

        # Determine if response is iterable; if not, assume it's a single ChatCompletion object.
        if hasattr(response, '__iter__') and not hasattr(response, "model"):
            # Handle streaming response (iterable)
            chunk = None
            count = 0
            for item in response:
                count += 1
                # Validate each chunk has required attributes
                if not hasattr(item, "model") or not hasattr(item, "usage"):
                    raise TypeError("Each item in response must be a ChatCompletionChunk with 'model' and 'usage' attributes. Please ensure that you have stream=True and stream_options={\"include_usage\": True}.")
                chunk = item
            if count == 0 or chunk is None:
                raise ValueError("Response stream is empty. Cannot estimate cost.")
        elif hasattr(response, "model") and hasattr(response, "usage"):
            # Single ChatCompletion object
            chunk = response
        else:
            raise TypeError("Response must be either an iterable of ChatCompletionChunk objects or a single ChatCompletion object with 'model' and 'usage' attributes.")

        # Extract model details
        try:
            model_details = extract_model_details(chunk.model)
        except Exception as e:
            raise ValueError(f"Error extracting model details: {e}")

        model_name = model_details.get("model_name")
        model_date = model_details.get("model_date")
        if not model_name or not model_date:
            raise ValueError("Model details could not be extracted properly.")

        # Convert usage object to string and parse usage data
        try:
            usage_str = str(chunk.usage)
            usage_data = parse_completion_usage(usage_str)
        except Exception as e:
            raise ValueError(f"Error parsing usage data: {e}")

        # Retrieve pricing for both Text and Audio tokens
        pricing_rates_text = self.pricing.get((model_name, model_date, "Text"))
        pricing_rates_audio = self.pricing.get((model_name, model_date, "Audio"))
        if not pricing_rates_text and not pricing_rates_audio:
            raise ValueError(f"Pricing data not found for model '{model_name}' ({model_date}).")

        # Calculate cost for text and audio tokens separately, then sum them
        try:
            cost_breakdown = calculate_cost(usage_data, pricing_rates_text, pricing_rates_audio)
        except Exception as e:
            raise ValueError(f"Error calculating cost: {e}")

        return cost_breakdown
