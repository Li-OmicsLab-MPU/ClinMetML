"""
ClinMetML: Automated Metabolomics Biomarker Discovery and Predictive Modeling

A comprehensive Python package for automated metabolomics data analysis,
biomarker discovery, and predictive model construction.

Main Components:
- Data cleaning and preprocessing
- Feature selection and dimensionality reduction  
- Multicollinearity reduction
- Automated model training and evaluation
- Model interpretability analysis
"""

__version__ = "0.1.0"
__author__ = "Junrong Li"


# Import main classes and functions for easy access
from .core.data_cleaner import DataCleaner
from .core.rfe_selector import FeatureRefinementSelector
from .core.model_trainer import CrossValidationPipeline
from .core.interpretability_analyzer import InterpretabilityPipeline
from .pipeline.auto_pipeline import ClinMetMLPipeline

# Import modules for function access
from .core import feature_selector, multicollinearity_reducer

# Main API exports
__all__ = [
    'DataCleaner',
    'FeatureRefinementSelector',
    'CrossValidationPipeline',
    'InterpretabilityPipeline',
    'ClinMetMLPipeline',
    'feature_selector',
    'multicollinearity_reducer',
]

# Package metadata
PACKAGE_INFO = {
    'name': 'clinmetml',
    'version': __version__,
    'description': 'Automated Metabolomics Biomarker Discovery and Predictive Modeling',
    'author': __author__,
    
    'url': 'https://github.com/your-username/clinmetml',
    'license': 'MIT',
}
