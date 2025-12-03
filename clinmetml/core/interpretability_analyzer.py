#!/usr/bin/env python3
"""
Model Interpretability Analysis Pipeline

This script provides comprehensive SHAP-based interpretability analysis for various machine learning models
trained on tabular biomarker data. It supports different model types and provides multiple visualization outputs.

Features:
- Support for multiple ML algorithms (Random Forest, XGBoost, LightGBM, etc.)
- Automatic SHAP explainer selection based on model type
- Comprehensive visualization suite (summary plots, waterfall plots, force plots, etc.)
- Feature importance analysis and ranking
- Batch processing capabilities

Author: Generated from Jupyter notebook
Date: 2025
"""

import os
import sys

# ClinMetML path management
from ..utils.paths import get_interpretability_dir, get_model_training_dir
import argparse
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
import shap

# Machine learning models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Sampling methods
from imblearn.under_sampling import NearMiss

# Import model persistence module
from .model_persistence import ModelPersistenceManager, DataSplitManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class InterpretabilityConfig:
    """Configuration class for interpretability analysis."""
    
    def __init__(self):
        # Input/Output paths
        self.data_path = None
        self.model_path = None
        self.output_dir = get_interpretability_dir()
        self.model_name_for_files = None  # Used for file naming prefix
        
        # Analysis configuration
        self.target_column = None  # Will be set from user arguments
        self.model_name = None     # Specific model to analyze, if None uses best model
        self.sample_indices = [2, 5, 10]  # Samples to analyze in detail
        self.max_display_features = 15
        self.interaction_features = None  # User-defined features for interaction plots, None means use top features
        
        # Sampling configuration
        self.apply_sampling = True
        self.sampling_method = "nearmiss"
        
        # Visualization configuration
        self.figure_dpi = 300
        self.figure_format = "png"
        self.save_plots = True
        self.show_plots = False
        
        # SHAP configuration
        self.shap_sample_size = 1000  # For background dataset
        self.class_to_analyze = 1  # Which class to focus on for binary classification


class DataLoader:
    """Handles data loading and preprocessing for interpretability analysis."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for analysis."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        X = df.drop(self.config.target_column, axis=1)
        y = df[self.config.target_column]
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {dict(y.value_counts())}")
        
        return X, y
    
    def apply_sampling_if_needed(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply sampling method if configured."""
        if not self.config.apply_sampling:
            return X, y
        
        logger.info(f"Applying {self.config.sampling_method} sampling")
        
        if self.config.sampling_method == "nearmiss":
            sampler = NearMiss(version=1)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            logger.info(f"Original distribution: {dict(y.value_counts())}")
            logger.info(f"Resampled distribution: {dict(pd.Series(y_resampled).value_counts())}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        
        return X, y


class ModelLoader:
    """Handles loading of trained models with full context."""
    
    @staticmethod
    def load_model(model_path: str, specific_model_name: str = None, data_path: str = None) -> Dict[str, Any]:
        """
        Load a trained model from file with full context, or create from scratch if needed.
        
        Args:
            model_path: Path to model file (.pkl, .json, or model directory)
            specific_model_name: Optional specific model name to load
            data_path: Path to data file (needed for creating models from scratch)
            
        Returns:
            Dictionary containing model and context information
        """
        logger.info(f"Loading model from {model_path}")
        if specific_model_name:
            logger.info(f"Specific model requested: {specific_model_name}")
        
        # If specific_model_name is provided but model_path doesn't exist or doesn't contain the model,
        # create the model from scratch
        if specific_model_name and not os.path.exists(model_path):
            logger.warning(f"Model path {model_path} does not exist")
            logger.info(f"Creating {specific_model_name} model from scratch...")
            return ModelLoader.create_model_from_scratch(specific_model_name, data_path)
        
        # Determine the type of model path
        if os.path.isdir(model_path):
            # If specific_model_name is provided and model_path is a directory, 
            # try to find the specific model in that directory
            if specific_model_name:
                specific_model_path = os.path.join(model_path, specific_model_name)
                if os.path.isdir(specific_model_path):
                    return ModelLoader._load_from_directory(specific_model_path)
                else:
                    logger.warning(f"Specific model directory {specific_model_path} not found")
                    logger.info(f"Creating {specific_model_name} model from scratch...")
                    return ModelLoader.create_model_from_scratch(specific_model_name, data_path)
            else:
                # Direct model directory
                return ModelLoader._load_from_directory(model_path)
        elif model_path.endswith('.pkl'):
            # Legacy pickle file
            return ModelLoader._load_legacy_pickle(model_path)
        elif model_path.endswith('.json'):
            # JSON metadata - try to find saved model or fallback to recreation
            try:
                return ModelLoader._load_from_json_metadata(model_path, specific_model_name)
            except (FileNotFoundError, ValueError) as e:
                if specific_model_name:
                    logger.warning(f"Failed to load from JSON metadata: {e}")
                    logger.info(f"Creating {specific_model_name} model from scratch...")
                    return ModelLoader.create_model_from_scratch(specific_model_name, data_path)
                else:
                    raise e
        else:
            if specific_model_name:
                logger.warning(f"Unsupported model path format: {model_path}")
                logger.info(f"Creating {specific_model_name} model from scratch...")
                return ModelLoader.create_model_from_scratch(specific_model_name, data_path)
            else:
                raise ValueError(f"Unsupported model path format: {model_path}")
    
    @staticmethod
    def _load_from_directory(model_dir: str) -> Dict[str, Any]:
        """Load model from a saved model directory."""
        # Extract model name from directory path
        model_name = os.path.basename(model_dir)
        output_dir = os.path.dirname(os.path.dirname(model_dir))  # Go up two levels to get output_dir
        
        persistence_manager = ModelPersistenceManager(output_dir)
        model_context = persistence_manager.load_model_with_context(model_name)
        
        return {
            'model': model_context['model'],
            'metadata': model_context['metadata'],
            'split_indices': model_context['split_indices'],
            'needs_retraining': False,
            'model_name': model_name
        }
    
    @staticmethod
    def _load_legacy_pickle(pickle_path: str) -> Dict[str, Any]:
        """Load model from legacy pickle file."""
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded legacy model: {type(model).__name__}")
        
        return {
            'model': model,
            'metadata': None,
            'split_indices': None,
            'needs_retraining': False,
            'model_name': type(model).__name__.lower()
        }
    
    @staticmethod
    def _load_from_json_metadata(json_path: str, specific_model_name: str = None) -> Dict[str, Any]:
        """Load model from JSON metadata, trying saved model first."""
        # First, try to find a saved model
        output_dir = os.path.dirname(os.path.dirname(json_path))  # Go up to output_dir
        persistence_manager = ModelPersistenceManager(output_dir)
        
        # If specific model name is provided, try to load it directly
        if specific_model_name:
            logger.info(f"Looking for specific model: {specific_model_name}")
            try:
                model_context = persistence_manager.load_model_with_context(specific_model_name)
                logger.info(f"✅ Found saved {specific_model_name} model, loading directly")
                return {
                    'model': model_context['model'],
                    'metadata': model_context['metadata'],
                    'split_indices': model_context['split_indices'],
                    'needs_retraining': False,
                    'model_name': specific_model_name
                }
            except FileNotFoundError:
                logger.warning(f"Specified model '{specific_model_name}' not found, falling back to best model")
        
        # Check if we have a saved best model
        try:
            best_model_context = persistence_manager.load_model_with_context('best_model')
            logger.info("✅ Found saved best model, loading directly")
            return {
                'model': best_model_context['model'],
                'metadata': best_model_context['metadata'],
                'split_indices': best_model_context['split_indices'],
                'needs_retraining': False,
                'model_name': best_model_context['metadata']['model_name']
            }
        except FileNotFoundError:
            logger.info("No saved best model found, checking individual models...")
        
        # Try to load individual saved models
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        best_model_name = metadata.get('best_model')
        if not best_model_name:
            raise ValueError("No 'best_model' found in metadata")
        
        try:
            model_context = persistence_manager.load_model_with_context(best_model_name)
            logger.info(f"✅ Found saved {best_model_name} model, loading directly")
            return {
                'model': model_context['model'],
                'metadata': model_context['metadata'],
                'split_indices': model_context['split_indices'],
                'needs_retraining': False,
                'model_name': best_model_name
            }
        except FileNotFoundError:
            logger.warning(f"No saved {best_model_name} model found, will need to recreate and retrain")
        
        # Fallback: recreate model (needs retraining)
        return ModelLoader._recreate_model_from_metadata(metadata, best_model_name)
    
    @staticmethod
    def _recreate_model_from_metadata(metadata: Dict, model_name: str) -> Dict[str, Any]:
        """Recreate model from metadata (requires retraining)."""
        logger.info(f"Recreating {model_name} model from metadata...")
        
        # Create model based on name
        model = ModelLoader._create_model_by_name(model_name)
        
        return {
            'model': model,
            'metadata': metadata,
            'split_indices': None,
            'needs_retraining': True,
            'model_name': model_name
        }
    
    @staticmethod
    def _create_model_by_name(model_name: str):
        """Create a new model instance by name with default parameters."""
        logger.info(f"Creating new {model_name} model instance...")
        
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1
                )
            except ImportError:
                raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=-1
                )
            except ImportError:
                raise ImportError("LightGBM is not installed. Please install it with: pip install lightgbm")
        elif model_name == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_name == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel='rbf',
                probability=True,  # Important for SHAP
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}. Supported models: random_forest, xgboost, lightgbm, gradient_boosting, decision_tree, logistic_regression, svm")
    
    @staticmethod
    def create_model_from_scratch(model_name: str, data_path: str = None) -> Dict[str, Any]:
        """Create a completely new model when no existing model or metadata is found."""
        logger.info(f"Creating {model_name} model from scratch...")
        
        model = ModelLoader._create_model_by_name(model_name)
        
        # Create minimal metadata
        metadata = {
            'model_name': model_name,
            'created_from_scratch': True,
            'training_data': {
                'data_path': data_path,
                'sampling_method': 'nearmiss'  # Default sampling method
            }
        }
        
        return {
            'model': model,
            'metadata': metadata,
            'split_indices': None,
            'needs_retraining': True,
            'model_name': model_name
        }
    
    @staticmethod
    def get_model_type(model) -> str:
        """Determine the type of the loaded model."""
        model_type = type(model).__name__.lower()
        
        # Check for pytorch-tabular models
        if 'tabularmodel' in model_type:
            return 'deep_learning'
        elif 'randomforest' in model_type:
            return 'tree'
        elif 'xgb' in model_type or 'lightgbm' in model_type or 'lgbm' in model_type:
            return 'tree'
        elif 'gradientboosting' in model_type:
            return 'tree'
        elif 'decisiontree' in model_type:
            return 'tree'
        elif 'linear' in model_type or 'logistic' in model_type:
            return 'linear'
        elif 'svm' in model_type:
            return 'kernel'
        else:
            return 'unknown'


class SHAPAnalyzer:
    """Main class for SHAP-based interpretability analysis."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explainer = None
        self.shap_values = None
        self.shap_values_class_0 = None
        self.shap_values_class_1 = None
        
    def create_explainer(self, model, X_background: pd.DataFrame, model_type: str = None):
        """Create appropriate SHAP explainer based on model type."""
        if model_type is None:
            model_type = ModelLoader.get_model_type(model)
        
        logger.info(f"Creating SHAP explainer for model type: {model_type}")
        
        # Sample background data if too large
        if len(X_background) > self.config.shap_sample_size:
            background_sample = X_background.sample(n=self.config.shap_sample_size, random_state=42)
        else:
            background_sample = X_background
        
        if model_type == 'tree':
            # Tree-based models (Random Forest, XGBoost, LightGBM, etc.)
            try:
                self.explainer = shap.TreeExplainer(model)
                logger.info("Using TreeExplainer")
            except (ValueError, TypeError) as e:
                # Handle XGBoost compatibility issues with SHAP
                if 'could not convert string to float' in str(e) or 'base_score' in str(e):
                    logger.warning(f"TreeExplainer failed for XGBoost model: {e}")
                    logger.info("Falling back to KernelExplainer for XGBoost model")
                    try:
                        self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                        logger.info("Using KernelExplainer (XGBoost fallback)")
                    except AttributeError as ke:
                        if "feature_names_in_" in str(ke):
                            logger.warning(f"KernelExplainer also failed: {ke}")
                            logger.info("Trying alternative approach with lambda wrapper")
                            # Create a wrapper function to avoid attribute issues
                            def model_predict(X):
                                return model.predict_proba(X)
                            self.explainer = shap.KernelExplainer(model_predict, background_sample.values)
                            logger.info("Using KernelExplainer with wrapper function")
                        else:
                            raise ke
                else:
                    raise e
        elif model_type == 'linear':
            # Linear models
            self.explainer = shap.LinearExplainer(model, background_sample)
            logger.info("Using LinearExplainer")
        elif model_type == 'deep_learning':
            # Deep learning models (pytorch-tabular)
            try:
                # For pytorch-tabular models, use KernelExplainer with predict_proba
                if hasattr(model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                    logger.info("Using KernelExplainer for deep learning model")
                else:
                    # Fallback to predict method
                    self.explainer = shap.KernelExplainer(model.predict, background_sample)
                    logger.info("Using KernelExplainer (predict) for deep learning model")
            except Exception as e:
                logger.warning(f"Failed to create explainer for DL model: {e}")
                # Fallback to default
                self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
                logger.info("Using KernelExplainer (fallback)")
        elif model_type == 'kernel':
            # Kernel-based models (SVM, etc.)
            self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            logger.info("Using KernelExplainer")
        else:
            # Default to KernelExplainer for unknown models
            self.explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            logger.info("Using KernelExplainer (default)")
    
    def calculate_shap_values(self, X_data: pd.DataFrame):
        """Calculate SHAP values for the given data."""
        logger.info("Calculating SHAP values...")
        
        # Calculate SHAP values
        shap_values_explanation = self.explainer(X_data)
        
        # Handle different SHAP value formats
        if hasattr(shap_values_explanation, 'values'):
            # New SHAP format (Explanation object)
            if len(shap_values_explanation.values.shape) == 3:
                # Multi-class output
                self.shap_values_class_0 = shap_values_explanation[:, :, 0]
                self.shap_values_class_1 = shap_values_explanation[:, :, 1]
            else:
                # Binary classification with single output
                self.shap_values_class_1 = shap_values_explanation
                self.shap_values_class_0 = None
        else:
            # Old SHAP format (numpy array)
            if isinstance(shap_values_explanation, list):
                self.shap_values_class_0 = shap_values_explanation[0]
                self.shap_values_class_1 = shap_values_explanation[1]
            else:
                self.shap_values_class_1 = shap_values_explanation
                self.shap_values_class_0 = None
        
        self.shap_values = shap_values_explanation
        logger.info("SHAP values calculated successfully")


class VisualizationManager:
    """Manages all visualization outputs for interpretability analysis."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_filename_with_prefix(self, base_filename: str) -> str:
        """Generate filename with model name prefix."""
        if self.config.model_name_for_files:
            return f"{self.config.model_name_for_files}_{base_filename}"
        return base_filename
    
    def create_feature_importance_plot(self, shap_values, X_data: pd.DataFrame, 
                                     class_name: str = "Class_1"):
        """Create feature importance bar plot."""
        logger.info(f"Creating feature importance plot for {class_name}")
        
        # Calculate mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            importance_values = np.abs(shap_values.values).mean(axis=0)
        else:
            importance_values = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X_data.columns,
            'Importance': importance_values
        }).sort_values('Importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8), dpi=self.config.figure_dpi)
        colors = sns.color_palette("viridis", n_colors=len(importance_df))
        
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title(f'Feature Importance - {class_name}', fontsize=14)
        plt.tight_layout()
        
        if self.config.save_plots:
            filename = self._get_filename_with_prefix(f'feature_importance_{class_name.lower()}.{self.config.figure_format}')
            plt.savefig(self.output_dir / filename, 
                       dpi=self.config.figure_dpi, bbox_inches='tight')
        
        if self.config.show_plots:
            plt.show()
        else:
            plt.close()
    
    def create_summary_plot(self, shap_values, X_data: pd.DataFrame, class_name: str = "Class_1"):
        """Create SHAP summary plot."""
        logger.info(f"Creating summary plot for {class_name}")
        
        plt.figure(figsize=(10, 8), dpi=self.config.figure_dpi)
        
        try:
            shap.summary_plot(shap_values, X_data, show=False, 
                            max_display=self.config.max_display_features)
            plt.title(f'SHAP Summary Plot - {class_name}', fontsize=14)
            
            if self.config.save_plots:
                filename = self._get_filename_with_prefix(f'summary_plot_{class_name.lower()}.{self.config.figure_format}')
                plt.savefig(self.output_dir / filename, 
                           dpi=self.config.figure_dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to create summary plot: {e}")
            plt.close()
    
    def create_waterfall_plots(self, shap_values, X_data: pd.DataFrame, 
                             sample_indices: List[int], class_name: str = "Class_1"):
        """Create waterfall plots for specific samples."""
        logger.info(f"Creating waterfall plots for {class_name}")
        
        for idx in sample_indices:
            if idx >= len(X_data):
                logger.warning(f"Sample index {idx} out of range, skipping")
                continue
            
            try:
                plt.figure(figsize=(10, 6), dpi=self.config.figure_dpi)
                shap.plots.waterfall(shap_values[idx], show=False, 
                                   max_display=self.config.max_display_features)
                plt.title(f'Waterfall Plot - Sample {idx} - {class_name}', fontsize=14)
                plt.tight_layout()
                
                if self.config.save_plots:
                    filename = self._get_filename_with_prefix(f'waterfall_sample_{idx}_{class_name.lower()}.{self.config.figure_format}')
                    plt.savefig(self.output_dir / filename, 
                               dpi=self.config.figure_dpi, bbox_inches='tight')
                
                if self.config.show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                logger.warning(f"Failed to create waterfall plot for sample {idx}: {e}")
                if plt.get_fignums():
                    plt.close()
    
    def create_force_plots(self, explainer, shap_values, X_data: pd.DataFrame, 
                          sample_indices: List[int], class_name: str = "Class_1"):
        """Create force plots for specific samples."""
        logger.info(f"Creating force plots for {class_name}")
        
        for idx in sample_indices:
            if idx >= len(X_data):
                logger.warning(f"Sample index {idx} out of range, skipping")
                continue
            
            try:
                plt.figure(figsize=(12, 4), dpi=self.config.figure_dpi)
                
                # Get base value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, (list, np.ndarray)):
                        base_value = explainer.expected_value[self.config.class_to_analyze]
                    else:
                        base_value = explainer.expected_value
                else:
                    base_value = 0.0
                
                # Try different force plot methods for compatibility
                try:
                    # Method 1: Use new SHAP API with Explanation object
                    if hasattr(shap_values, 'values'):
                        shap.plots.force(shap_values[idx], matplotlib=True, show=False)
                    else:
                        # Method 2: Use legacy force_plot function
                        shap.force_plot(
                            base_value,
                            shap_values[idx] if hasattr(shap_values[idx], '__len__') else shap_values[idx].values,
                            X_data.iloc[idx].values,
                            feature_names=list(X_data.columns),
                            matplotlib=True,
                            show=False
                        )
                except:
                    # Method 3: Create manual force plot using bar chart
                    logger.info(f"Creating manual force plot for sample {idx}")
                    
                    # Get SHAP values for this sample
                    if hasattr(shap_values, 'values'):
                        sample_shap = shap_values[idx].values
                    else:
                        sample_shap = shap_values[idx]
                    
                    # Create horizontal bar chart
                    feature_names = list(X_data.columns)
                    colors = ['red' if val > 0 else 'blue' for val in sample_shap]
                    
                    plt.barh(range(len(feature_names)), sample_shap, color=colors, alpha=0.7)
                    plt.yticks(range(len(feature_names)), feature_names)
                    plt.xlabel('SHAP Value')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add base value annotation
                    plt.text(0.02, 0.98, f'Base value: {base_value:.3f}', 
                            transform=plt.gca().transAxes, verticalalignment='top')
                
                plt.title(f'Force Plot - Sample {idx} - {class_name}', fontsize=14)
                plt.tight_layout()
                
                if self.config.save_plots:
                    filename = self._get_filename_with_prefix(f'force_plot_sample_{idx}_{class_name.lower()}.{self.config.figure_format}')
                    plt.savefig(self.output_dir / filename, 
                               dpi=self.config.figure_dpi, bbox_inches='tight')
                
                if self.config.show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                logger.warning(f"Failed to create force plot for sample {idx}: {e}")
                if plt.get_fignums():
                    plt.close()
    
    def create_feature_interaction_plot(self, shap_values, X_data: pd.DataFrame, 
                                      feature_name: str, class_name: str = "Class_1"):
        """Create feature interaction plot with LOWESS curve."""
        logger.info(f"Creating interaction plot for feature: {feature_name}")
        
        if feature_name not in X_data.columns:
            logger.warning(f"Feature {feature_name} not found in data")
            return
        
        try:
            # Extract SHAP values for the specific feature
            feature_idx = list(X_data.columns).index(feature_name)
            
            if hasattr(shap_values, 'values'):
                feature_shap_values = shap_values.values[:, feature_idx]
            else:
                feature_shap_values = shap_values[:, feature_idx]
            
            feature_values = X_data[feature_name].values
            
            # Create plot
            plt.figure(figsize=(8, 6), dpi=self.config.figure_dpi)
            plt.scatter(feature_values, feature_shap_values, s=20, alpha=0.7, label='SHAP values')
            
            # Add LOWESS curve if statsmodels is available
            try:
                import statsmodels.api as sm
                lowess_result = sm.nonparametric.lowess(feature_shap_values, feature_values, frac=0.3)
                plt.plot(lowess_result[:, 0], lowess_result[:, 1], color='red', linewidth=2, label='LOWESS Curve')
            except ImportError:
                logger.warning("statsmodels not available, skipping LOWESS curve")
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='SHAP = 0')
            
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel(f'SHAP value for {feature_name}', fontsize=12)
            plt.title(f'Feature Interaction: {feature_name} - {class_name}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if self.config.save_plots:
                safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
                filename = self._get_filename_with_prefix(f'interaction_{safe_feature_name}_{class_name.lower()}.{self.config.figure_format}')
                plt.savefig(self.output_dir / filename, 
                           dpi=self.config.figure_dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.warning(f"Failed to create interaction plot for {feature_name}: {e}")
            if plt.get_fignums():
                plt.close()


class InterpretabilityPipeline:
    """Main pipeline for model interpretability analysis."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.shap_analyzer = SHAPAnalyzer(config)
        self.viz_manager = VisualizationManager(config)
    
    def run_analysis(self, data_path: str, model_path: str):
        """Run complete interpretability analysis."""
        logger.info("=" * 60)
        logger.info("STARTING MODEL INTERPRETABILITY ANALYSIS")
        logger.info("=" * 60)
        
        # Load data
        X, y = self.data_loader.load_data(data_path)
        
        # Load model with context
        model_context = ModelLoader.load_model(model_path, self.config.model_name, data_path)
        model = model_context['model']
        model_name = model_context['model_name']
        needs_retraining = model_context['needs_retraining']
        split_indices = model_context['split_indices']
        metadata = model_context['metadata']
        
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Needs retraining: {needs_retraining}")
        
        # Set model name for file naming
        self.config.model_name_for_files = model_name
        logger.info(f"Files will be prefixed with: {model_name}_")
        
        # Apply sampling if needed (use same method as training if available)
        if metadata and 'training_data' in metadata:
            training_sampling = metadata['training_data'].get('sampling_method')
            if training_sampling and training_sampling != 'none':
                logger.info(f"Using same sampling method as training: {training_sampling}")
                self.config.sampling_method = training_sampling
        
        X_processed, y_processed = self.data_loader.apply_sampling_if_needed(X, y)
        
        # Handle data splitting and model training
        if needs_retraining:
            logger.info(f"Training {model_name} model...")
            
            # Use same split as original training if available
            if split_indices:
                logger.info("Restoring original train-test split from saved indices...")
                X_train, X_test, y_train, y_test = DataSplitManager.restore_split_from_indices(
                    X_processed, y_processed, split_indices
                )
            else:
                logger.info("Creating new train-test split...")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
                )
            
            # Train the model
            logger.info(f"Training {model_name} on {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Evaluate the model
            from sklearn.metrics import accuracy_score, roc_auc_score
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained successfully - Test Accuracy: {accuracy:.4f}")
            
            if y_pred_proba is not None:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    logger.info(f"Test AUC: {auc:.4f}")
                except ValueError:
                    logger.warning("Could not calculate AUC (possibly single class in test set)")
            
            # Use training data for SHAP analysis
            X_for_shap = X_train.reset_index(drop=True)
        else:
            logger.info("Using pre-trained model directly")
            
            # If we have split indices, use the training data for consistency
            if split_indices:
                logger.info("Using original training data for SHAP analysis...")
                X_train, X_test, y_train, y_test = DataSplitManager.restore_split_from_indices(
                    X_processed, y_processed, split_indices
                )
                X_for_shap = X_train.reset_index(drop=True)
            else:
                # Use full processed data
                X_for_shap = X_processed
        
        logger.info(f"Using data for SHAP analysis: {X_for_shap.shape}")
        
        # Adjust sample indices if they exceed the data size
        max_idx = len(X_for_shap) - 1
        adjusted_indices = [idx for idx in self.config.sample_indices if idx <= max_idx]
        if len(adjusted_indices) < len(self.config.sample_indices):
            logger.warning(f"Some sample indices exceed data size. Using indices: {adjusted_indices}")
            self.config.sample_indices = adjusted_indices
        
        model_type = ModelLoader.get_model_type(model)
        
        # Create SHAP explainer
        self.shap_analyzer.create_explainer(model, X_for_shap, model_type)
        
        # Calculate SHAP values
        self.shap_analyzer.calculate_shap_values(X_for_shap)
        
        # Generate visualizations
        self._generate_all_visualizations(X_for_shap)
        
        # Save analysis results
        self._save_analysis_results(X_for_shap)
        
        logger.info("=" * 60)
        logger.info("INTERPRETABILITY ANALYSIS COMPLETED")
        logger.info(f"Results saved to: {self.config.output_dir}")
        logger.info("=" * 60)
    
    def _generate_all_visualizations(self, X_data: pd.DataFrame):
        """Generate all visualization outputs."""
        logger.info("Generating visualizations...")
        
        # Feature importance plots
        if self.shap_analyzer.shap_values_class_1 is not None:
            self.viz_manager.create_feature_importance_plot(
                self.shap_analyzer.shap_values_class_1, X_data, "Class_1"
            )
            
            # Summary plot
            self.viz_manager.create_summary_plot(
                self.shap_analyzer.shap_values_class_1, X_data, "Class_1"
            )
            
            # Waterfall plots
            self.viz_manager.create_waterfall_plots(
                self.shap_analyzer.shap_values_class_1, X_data, 
                self.config.sample_indices, "Class_1"
            )
            
            # Force plots
            self.viz_manager.create_force_plots(
                self.shap_analyzer.explainer, self.shap_analyzer.shap_values_class_1, 
                X_data, self.config.sample_indices, "Class_1"
            )
            
            # Feature interaction plots
            if self.config.interaction_features:
                # Use user-defined features
                features_to_plot = []
                for feature in self.config.interaction_features:
                    if feature in X_data.columns:
                        features_to_plot.append(feature)
                    else:
                        logger.warning(f"User-defined interaction feature '{feature}' not found in data columns")
                
                if not features_to_plot:
                    logger.warning("No valid user-defined interaction features found, falling back to top features")
                    features_to_plot = self._get_top_features(X_data, n_features=3)
            else:
                # Use top features by default
                features_to_plot = self._get_top_features(X_data, n_features=3)
            
            for feature in features_to_plot:
                self.viz_manager.create_feature_interaction_plot(
                    self.shap_analyzer.shap_values_class_1, X_data, feature, "Class_1"
                )
        
        # Generate plots for class 0 if available
        if self.shap_analyzer.shap_values_class_0 is not None:
            self.viz_manager.create_feature_importance_plot(
                self.shap_analyzer.shap_values_class_0, X_data, "Class_0"
            )
    
    def _get_top_features(self, X_data: pd.DataFrame, n_features: int = 5) -> List[str]:
        """Get top N most important features."""
        if self.shap_analyzer.shap_values_class_1 is None:
            return list(X_data.columns[:n_features])
        
        # Calculate feature importance
        if hasattr(self.shap_analyzer.shap_values_class_1, 'values'):
            importance_values = np.abs(self.shap_analyzer.shap_values_class_1.values).mean(axis=0)
        else:
            importance_values = np.abs(self.shap_analyzer.shap_values_class_1).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(importance_values)[-n_features:][::-1]
        return [X_data.columns[i] for i in top_indices]
    
    def _save_analysis_results(self, X_data: pd.DataFrame):
        """Save analysis results to files."""
        logger.info("Saving analysis results...")
        
        # Save feature importance
        if self.shap_analyzer.shap_values_class_1 is not None:
            if hasattr(self.shap_analyzer.shap_values_class_1, 'values'):
                importance_values = np.abs(self.shap_analyzer.shap_values_class_1.values).mean(axis=0)
            else:
                importance_values = np.abs(self.shap_analyzer.shap_values_class_1).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'Feature': X_data.columns,
                'Importance': importance_values
            }).sort_values('Importance', ascending=False)
            
            # Generate filename with model prefix
            importance_filename = self.viz_manager._get_filename_with_prefix('feature_importance.csv')
            importance_df.to_csv(self.config.output_dir + '/' + importance_filename, index=False)
        
        # Save SHAP values
        if hasattr(self.shap_analyzer.shap_values_class_1, 'values'):
            shap_df = pd.DataFrame(
                self.shap_analyzer.shap_values_class_1.values, 
                columns=X_data.columns
            )
        else:
            shap_df = pd.DataFrame(
                self.shap_analyzer.shap_values_class_1, 
                columns=X_data.columns
            )
        
        # Generate filename with model prefix
        shap_filename = self.viz_manager._get_filename_with_prefix('shap_values.csv')
        shap_df.to_csv(self.config.output_dir + '/' + shap_filename, index=False)


def main():
    """Main function to run interpretability analysis."""
    parser = argparse.ArgumentParser(description='Model Interpretability Analysis')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to the data CSV file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model (.pkl), model metadata (.json), or saved model directory (default: auto-detected)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: auto-managed)')
    parser.add_argument('--target_column', type=str, required=True,
                       help='Name of the target column')
    parser.add_argument('--sample_indices', type=str, default='2,5,10',
                       help='Comma-separated sample indices to analyze')
    parser.add_argument('--apply_sampling', action='store_true',
                       help='Apply NearMiss sampling')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to analyze (e.g., random_forest, xgboost, ft_transformer). If not specified, uses best model.')
    parser.add_argument('--interaction_features', type=str, default=None,
                       help='Comma-separated list of features for interaction plots (e.g., "feature1,feature2"). If not specified, uses top 3 most important features.')
    
    args = parser.parse_args()
    
    # Create configuration
    config = InterpretabilityConfig()
    config.data_path = args.data_path
    
    # Set default model path if not provided
    if args.model_path is None:
        config.model_path = os.path.join(get_model_training_dir(), "saved_models")
    else:
        config.model_path = args.model_path
    
    # Use path manager for output directory
    config.output_dir = args.output_dir if args.output_dir is not None else get_interpretability_dir()
    config.target_column = args.target_column
    config.model_name = args.model_name
    config.sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
    config.apply_sampling = args.apply_sampling
    config.show_plots = args.show_plots
    
    # Parse interaction features
    if args.interaction_features:
        config.interaction_features = [x.strip() for x in args.interaction_features.split(',')]
        logger.info(f"User-defined interaction features: {config.interaction_features}")
    else:
        config.interaction_features = None
        logger.info("Will use top 3 most important features for interaction plots")
    
    logger.info(f"Output directory forced to: {config.output_dir}")
    
    # Run analysis
    pipeline = InterpretabilityPipeline(config)
    pipeline.run_analysis(args.data_path, args.model_path)


if __name__ == "__main__":
    main()
