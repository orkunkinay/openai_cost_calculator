import requests
import csv
import io
import time

PRICING_CSV_URL = "https://raw.githubusercontent.com/orkunkinay/openai_api_data/refs/heads/main/gpt_pricing_data.csv"

# Global variables for caching
_pricing_cache = None
_cache_timestamp = 0
_CACHE_DURATION = 60 * 60 * 24  # Cache for 24 hours

def load_pricing():
    """
    Fetches pricing data from a remote CSV file, using a local cache to avoid excessive requests.

    Returns:
        dict: Pricing data with keys as (Model Name, Model Date, Token Type).
    
    Raises:
        RuntimeError: If fetching the pricing data fails.
    """
    global _pricing_cache, _cache_timestamp
    
    current_time = time.time()
    # Use cached data if valid
    if _pricing_cache is not None and (current_time - _cache_timestamp) < _CACHE_DURATION:
        return _pricing_cache

    try:
        response = requests.get(PRICING_CSV_URL, timeout=5)
        response.raise_for_status()  # Raise error if request fails

        # Read CSV content from response
        csv_file = io.StringIO(response.text)
        reader = csv.DictReader(csv_file)

        pricing = {}
        for row in reader:
            key = (row["Model Name"], row["Model Date"], row["Token Type"])
            pricing[key] = {
                "input_price": float(row["Input Price"]),
                # Check for empty string; if empty, set to None
                "cached_input_price": float(row["Cached Input Price"]) if row["Cached Input Price"].strip() else None,
                "output_price": float(row["Output Price"])
            }

        # Update cache
        _pricing_cache = pricing
        _cache_timestamp = current_time

        return pricing
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch pricing data: {e}")