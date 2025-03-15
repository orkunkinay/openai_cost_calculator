# OpenAI Cost Calculator

**OpenAI Cost Calculator** is a Python library that estimates your OpenAI API usage costs based on token usage data. It supports both text and audio tokens. This makes it easy to integrate cost tracking into your applications using the OpenAI service.

**Sample Usage**
```python
from openai_cost_calculator import CostEstimator

# Initialize the cost estimator
estimator = CostEstimator()

# Simulate an OpenAI response
response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0
)

# Estimate cost
try:
    cost_breakdown = estimator.estimate_cost(response)
    print("Cost Breakdown:", cost_breakdown)
except Exception as e:
    print(f"Error estimating cost: {e}")
```

## Installation

```bash
pip install openai_cost_calculator
