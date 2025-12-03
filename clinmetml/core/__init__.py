"""
Core modules for ClinMetML package.

This module contains the main classes for data processing, feature selection,
model training, and interpretability analysis.
"""

from .data_cleaner import DataCleaner
from .feature_selector import FeatureSelector
from .multicollinearity_reducer import MulticollinearityReducer
from .rfe_selector import FeatureRefinementSelector, RFESelector
from .model_trainer import (
    ModelTrainingConfig, DataProcessor, SamplingManager, 
    ParameterOptimizer, MetricsCalculator, TraditionalMLTrainer,
    DeepLearningTrainer, CrossValidationPipeline
)
from .interpretability_analyzer import (
    InterpretabilityConfig, DataLoader, ModelLoader,
    SHAPAnalyzer, VisualizationManager, InterpretabilityPipeline
)
from .model_persistence import ModelPersistenceManager, DataSplitManager

# Import main functions from modules without classes
from . import feature_selector
from . import multicollinearity_reducer

__all__ = [
    'DataCleaner',
    'FeatureSelector',
    'MulticollinearityReducer',
    'FeatureRefinementSelector',
    'RFESelector',
    'ModelTrainingConfig',
    'DataProcessor',
    'SamplingManager', 
    'ParameterOptimizer',
    'MetricsCalculator',
    'TraditionalMLTrainer',
    'DeepLearningTrainer',
    'CrossValidationPipeline',
    'InterpretabilityConfig',
    'DataLoader',
    'ModelLoader',
    'SHAPAnalyzer',
    'VisualizationManager',
    'InterpretabilityPipeline',
    'ModelPersistenceManager',
    'DataSplitManager',
    'feature_selector',
    'multicollinearity_reducer',
]
