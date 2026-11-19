"""Unsupervised anomaly detection on self-checkout scan sequences (synthetic-data demo)."""

from .ensemble import AnomalyEnsemble, CheckoutAnomalyPipeline, combine_normalised_scores
from .features import CorrelationFilter, FeatureExtractor
from .parsing import Visit, read_visits, write_visits
from .synthetic import FRAUD_TYPES, GeneratorConfig, VisitGenerator

__all__ = [
    "AnomalyEnsemble", "CheckoutAnomalyPipeline", "combine_normalised_scores",
    "CorrelationFilter", "FeatureExtractor", "Visit", "read_visits", "write_visits",
    "FRAUD_TYPES", "GeneratorConfig", "VisitGenerator",
]
__version__ = "0.1.0"
