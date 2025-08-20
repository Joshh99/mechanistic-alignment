"""
Mini SAE Package for Mechanistic Interpretability

This package implements a Sparse Autoencoder (SAE) for analyzing transformer activations
and discovering interpretable features.

Main Components:
- MiniSAE: Core SAE model with encoder/decoder architecture
- SAEConfig: Configuration class for SAE parameters
- SAETrainer: Training infrastructure with loss tracking and sparsity management
- SAEAnalyzer: Analysis tools for feature discovery and visualization
- ActivationCollector: Pipeline for collecting transformer activations
"""

from .model import MiniSAE, SAEConfig, ActivationCollector
from .training import SAETrainer, SAEAnalyzer

__version__ = "0.1.0"

__all__ = [
    "MiniSAE",
    "SAEConfig", 
    "SAETrainer",
    "SAEAnalyzer",
    "ActivationCollector"
]