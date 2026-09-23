"""Provider-agnostic pricing domain: data model, validation, lookup and arithmetic.

Nothing in this package knows about any specific provider.  Provider quirks
(identifier formats, default regions, peak hours) live in
:mod:`openai_cost_calculator.providers`; fetching and parsing upstream prices
lives in :mod:`openai_cost_calculator.sync`.
"""

from .catalog import BUNDLED_DATA_DIR, ModelIndex, PricingCatalog, bundled_catalog, compact_key
from .costing import USAGE_FIELDS, Cost, LineItem, Usage, price_usage
from .dimensions import DIMENSIONS, Dimension, get_dimension
from .errors import (
    AmbiguousModelError,
    CatalogValidationError,
    MissingRateError,
    PricingError,
    PricingUnavailableError,
    UnknownModelError,
    UnknownProviderError,
    UsageError,
)
from .model import (
    CONDITION_KEYS,
    DEFAULT_CONDITIONS,
    ModelPricing,
    PriceSet,
    ProviderPricing,
    Source,
)
from .selection import RequestConditions, Selection, select_price_set

__all__ = [
    "BUNDLED_DATA_DIR",
    "CONDITION_KEYS",
    "DEFAULT_CONDITIONS",
    "DIMENSIONS",
    "USAGE_FIELDS",
    "AmbiguousModelError",
    "CatalogValidationError",
    "Cost",
    "Dimension",
    "LineItem",
    "MissingRateError",
    "ModelIndex",
    "ModelPricing",
    "PriceSet",
    "PricingCatalog",
    "PricingError",
    "PricingUnavailableError",
    "ProviderPricing",
    "RequestConditions",
    "Selection",
    "Source",
    "UnknownModelError",
    "UnknownProviderError",
    "Usage",
    "UsageError",
    "bundled_catalog",
    "compact_key",
    "get_dimension",
    "price_usage",
    "select_price_set",
]
