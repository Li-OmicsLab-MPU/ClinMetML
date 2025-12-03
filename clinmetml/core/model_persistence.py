#!/usr/bin/env python3
"""
Model Persistence Module

This module handles saving and loading of trained models along with their training configurations,
ensuring reproducibility and consistency between training and interpretability analysis.

Features:
- Save complete model state (model object + training config + data splits)
- Load models with full context restoration
- Ensure reproducible data splits using saved random states
- Support for multiple model types and configurations

Author: Generated for tabular biomarker pipeline
Date: 2025
"""

import os
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ModelPersistenceManager:
    """Manages saving and loading of trained models with full context."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "saved_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_with_context(self, 
                               model: Any,
                               model_name: str,
                               model_config: Dict,
                               training_data: Dict,
                               performance_metrics: Dict,
                               data_split_info: Dict) -> str:
        """
        Save a trained model with complete context for reproducibility.
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'random_forest', 'xgboost')
            model_config: Model configuration and hyperparameters
            training_data: Information about training data
            performance_metrics: Model performance metrics
            data_split_info: Information about data splits (indices, random_state, etc.)
            
        Returns:
            Path to saved model directory
        """
        # Create model-specific directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model object
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'model_config': model_config,
            'training_data': training_data,
            'performance_metrics': performance_metrics,
            'data_split_info': data_split_info,
            'save_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save data split indices if available
        if 'train_indices' in data_split_info and 'test_indices' in data_split_info:
            split_indices = {
                'train_indices': data_split_info['train_indices'],
                'test_indices': data_split_info['test_indices']
            }
            indices_path = model_dir / "split_indices.json"
            with open(indices_path, 'w') as f:
                json.dump(split_indices, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_dir}")
        return str(model_dir)
    
    def load_model_with_context(self, model_name: str) -> Dict[str, Any]:
        """
        Load a saved model with complete context.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary containing model, metadata, and context information
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load model object
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load split indices if available
        split_indices = None
        indices_path = model_dir / "split_indices.json"
        if indices_path.exists():
            with open(indices_path, 'r') as f:
                split_indices = json.load(f)
        
        logger.info(f"Model loaded from: {model_dir}")
        
        return {
            'model': model,
            'metadata': metadata,
            'split_indices': split_indices,
            'model_dir': str(model_dir)
        }
    
    def list_saved_models(self) -> List[str]:
        """List all saved models."""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pkl").exists():
                models.append(model_dir.name)
        
        return models
    
    def get_best_model_info(self) -> Optional[Dict]:
        """Get information about the best model from best_model.json."""
        best_model_path = self.output_dir / "metrics" / "best_model.json"
        
        if not best_model_path.exists():
            logger.warning(f"Best model info not found: {best_model_path}")
            return None
        
        with open(best_model_path, 'r') as f:
            return json.load(f)
    
    def save_best_model(self, 
                       models_dict: Dict[str, Any],
                       training_results: Dict[str, Any],
                       data_context: Dict[str, Any]) -> str:
        """
        Save the best model based on best_model.json information.
        
        Args:
            models_dict: Dictionary of trained models {model_name: model_object}
            training_results: Training results and metrics
            data_context: Data processing and split information
            
        Returns:
            Path to saved best model
        """
        # Get best model info
        best_model_info = self.get_best_model_info()
        if not best_model_info:
            raise ValueError("Best model information not available")
        
        best_model_name = best_model_info['best_model']
        
        if best_model_name not in models_dict:
            raise ValueError(f"Best model '{best_model_name}' not found in models dictionary")
        
        best_model = models_dict[best_model_name]
        
        # Prepare model configuration
        model_config = {
            'model_parameters': self._extract_model_parameters(best_model),
            'selection_criteria': best_model_info['selection_criteria'],
            'training_config': data_context.get('training_config', {})
        }
        
        # Prepare training data info
        training_data_info = self._build_training_data_info(data_context)
        
        # Get performance metrics
        performance_metrics = best_model_info['scores']
        
        # Get data split info
        data_split_info = data_context.get('data_split_info', {})
        
        # Save the best model
        model_path = self.save_model_with_context(
            model=best_model,
            model_name=best_model_name,
            model_config=model_config,
            training_data=training_data_info,
            performance_metrics=performance_metrics,
            data_split_info=data_split_info
        )
        
        # Also save as 'best_model' for easy access
        best_model_link = self.models_dir / "best_model"
        if best_model_link.exists():
            if best_model_link.is_symlink():
                best_model_link.unlink()
            else:
                import shutil
                shutil.rmtree(best_model_link)
        
        # Create symlink or copy
        try:
            best_model_link.symlink_to(best_model_name, target_is_directory=True)
            logger.info(f"Created symlink: best_model -> {best_model_name}")
        except OSError as e:
            logger.warning(f"Symlink creation failed: {e}, trying copy...")
            # Fallback: copy directory if symlink fails
            import shutil
            try:
                shutil.copytree(self.models_dir / best_model_name, best_model_link)
                logger.info(f"Created copy: best_model (from {best_model_name})")
            except Exception as copy_error:
                logger.error(f"Failed to create best_model directory: {copy_error}")
                raise
        
        logger.info(f"Best model '{best_model_name}' saved and linked as 'best_model'")
        return model_path
    
    def _build_training_data_info(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a standardized training_data_info dict from a data context.

        This helper centralizes how we extract shapes, feature names,
        target column, sampling method and class distribution so that the
        same structure is used consistently whenever models are saved
        with context.
        """

        return {
            'original_shape': data_context.get('original_shape'),
            'processed_shape': data_context.get('processed_shape'),
            'feature_names': data_context.get('feature_names', []),
            'target_column': data_context.get('target_column'),
            'sampling_method': data_context.get('sampling_method'),
            'class_distribution': data_context.get('class_distribution', {}),
        }
    
    def _extract_model_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract model parameters for saving, filtering out null values."""
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Filter out None/null values to reduce storage size
                filtered_params = {k: v for k, v in params.items() if v is not None}
                return filtered_params
            else:
                # For models without get_params, extract common attributes
                params = {}
                for attr in dir(model):
                    if not attr.startswith('_') and not callable(getattr(model, attr)):
                        try:
                            value = getattr(model, attr)
                            if isinstance(value, (int, float, str, bool, list, dict)) and value is not None:
                                params[attr] = value
                        except:
                            continue
                return params
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")
            return {}


class DataSplitManager:
    """Manages consistent data splitting for training and interpretability analysis."""
    
    @staticmethod
    def create_reproducible_split(X: pd.DataFrame, 
                                y: pd.Series,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Create a reproducible train-test split and return split information.
        
        Args:
            X: Feature data
            y: Target data
            test_size: Proportion of test data
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split
            
        Returns:
            X_train, X_test, y_train, y_test, split_info
        """
        # Perform split
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Create split information
        split_info = {
            'test_size': test_size,
            'random_state': random_state,
            'stratify': stratify,
            'train_indices': X_train.index.tolist(),
            'test_indices': X_test.index.tolist(),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'original_shape': X.shape
        }
        
        return X_train, X_test, y_train, y_test, split_info
    
    @staticmethod
    def restore_split_from_indices(X: pd.DataFrame,
                                  y: pd.Series,
                                  split_indices: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Restore train-test split using saved indices.
        
        Args:
            X: Feature data
            y: Target data
            split_indices: Dictionary containing train and test indices
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        train_indices = split_indices['train_indices']
        test_indices = split_indices['test_indices']
        
        # Restore splits using indices
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)
        
        return X_train, X_test, y_train, y_test


# Utility functions for backward compatibility
def save_model_with_full_context(output_dir: str,
                                model: Any,
                                model_name: str,
                                **context) -> str:
    """Convenience function for saving models with context."""
    manager = ModelPersistenceManager(output_dir)
    return manager.save_model_with_context(model, model_name, **context)


def load_model_with_full_context(output_dir: str, model_name: str) -> Dict[str, Any]:
    """Convenience function for loading models with context."""
    manager = ModelPersistenceManager(output_dir)
    return manager.load_model_with_context(model_name)
