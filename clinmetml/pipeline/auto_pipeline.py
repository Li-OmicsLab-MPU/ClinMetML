"""
ClinMetMLPipeline: Main pipeline class for automated metabolomics analysis.

This module provides a unified interface for the entire ClinMetML workflow,
from data cleaning to model interpretability analysis.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path

# Import core modules
from ..core.data_cleaner import DataCleaner
from ..core.feature_selector import FeatureSelector
from ..core.multicollinearity_reducer import MulticollinearityReducer
from ..core.rfe_selector import RFESelector
from ..core.model_trainer import ModelTrainingConfig, CrossValidationPipeline
from ..core.interpretability_analyzer import InterpretabilityConfig, InterpretabilityPipeline


class ClinMetMLPipeline:
    """
    Main pipeline class for automated metabolomics biomarker discovery and modeling.
    
    This class orchestrates the entire workflow:
    1. Data cleaning and preprocessing
    2. Feature selection
    3. Multicollinearity reduction
    4. RFE-based feature refinement
    5. Model training and evaluation
    6. Model interpretability analysis
    
    Parameters
    ----------
    output_dir : str, optional
        Directory to save all outputs. Default is 'clinmetml_outputs'
    random_state : int, optional
        Random state for reproducibility. Default is 42
    verbose : bool, optional
        Whether to print detailed progress information. Default is True
    """
    
    def __init__(
        self,
        output_dir: str = "clinmetml_outputs",
        random_state: int = 42,
        verbose: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_cleaner = None
        self.feature_selector = None
        self.multicollinearity_reducer = None
        self.rfe_selector = None
        self.model_trainer = None
        self.interpretability_analyzer = None
        
        # Store intermediate results
        self.results = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "clinmetml_pipeline.log"
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(
        self,
        data: Union[str, pd.DataFrame],
        target_column: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete ClinMetML pipeline.
        
        Parameters
        ----------
        data : str or pd.DataFrame
            Input data file path or DataFrame
        target_column : str
            Name of the target column
        **kwargs : dict
            Additional parameters for pipeline components
            
        Returns
        -------
        dict
            Dictionary containing all pipeline results
        """
        self.logger.info("Starting ClinMetML pipeline...")
        
        # Load data if needed
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        # Step 1: Data Cleaning
        self.logger.info("Step 1: Data cleaning and preprocessing...")
        cleaned_data = self.run_data_cleaning(
            data, 
            **kwargs.get('data_cleaning', {})
        )
        
        # Step 2: Feature Selection
        self.logger.info("Step 2: Feature selection...")
        selected_data = self.run_feature_selection(
            cleaned_data,
            target_column,
            **kwargs.get('feature_selection', {})
        )
        
        # Step 3: Multicollinearity Reduction
        self.logger.info("Step 3: Multicollinearity reduction...")
        reduced_data = self.run_multicollinearity_reduction(
            selected_data,
            target_column,
            **kwargs.get('multicollinearity_reduction', {})
        )
        
        # Step 4: RFE Selection
        self.logger.info("Step 4: RFE-based feature refinement...")
        refined_data = self.run_rfe_selection(
            reduced_data,
            target_column,
            **kwargs.get('rfe_selection', {})
        )
        
        # Step 5: Model Training
        self.logger.info("Step 5: Model training and evaluation...")
        model_results = self.run_model_training(
            refined_data,
            target_column,
            **kwargs.get('model_training', {})
        )
        
        # Step 6: Model Interpretability
        self.logger.info("Step 6: Model interpretability analysis...")
        interpretability_results = self.run_interpretability_analysis(
            refined_data,
            target_column,
            model_results,
            **kwargs.get('interpretability', {})
        )
        
        # Compile final results
        final_results = {
            'cleaned_data': cleaned_data,
            'selected_data': selected_data,
            'reduced_data': reduced_data,
            'refined_data': refined_data,
            'model_results': model_results,
            'interpretability_results': interpretability_results,
            'pipeline_config': kwargs
        }
        
        self.results = final_results
        self.logger.info("ClinMetML pipeline completed successfully!")
        
        return final_results
    
    def run_data_cleaning(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run data cleaning step."""
        self.data_cleaner = DataCleaner()
        
        # Default parameters
        params = {
            'imputation_method': 'knn',
            'filter_missing_threshold': 0.2,
            'normalization': 'standard',
            'log_transform': True,
            'autonorm': True,
            **kwargs
        }
        
        cleaned_data = self.data_cleaner.clean_pipeline(data, **params)
        
        # Save cleaned data
        output_file = self.output_dir / "cleaned_data.csv"
        cleaned_data.to_csv(output_file, index=False)
        
        return cleaned_data
    
    def run_feature_selection(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        **kwargs
    ) -> pd.DataFrame:
        """Run feature selection step."""
        self.feature_selector = FeatureSelector()
        
        # Default parameters
        params = {
            'output_dir': str(self.output_dir / "feature_selection_outputs"),
            **kwargs
        }
        
        selected_data = self.feature_selector.select_features(
            data, target_column, **params
        )
        
        return selected_data
    
    def run_multicollinearity_reduction(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs
    ) -> pd.DataFrame:
        """Run multicollinearity reduction step."""
        self.multicollinearity_reducer = MulticollinearityReducer()
        
        # Default parameters  
        params = {
            'output_dir': str(self.output_dir / "multicollinearity_outputs"),
            'vif_threshold': 5.0,
            **kwargs
        }
        
        reduced_data = self.multicollinearity_reducer.reduce_multicollinearity(
            data, target_column, **params
        )
        
        return reduced_data
    
    def run_rfe_selection(
        self,
        data: pd.DataFrame, 
        target_column: str,
        **kwargs
    ) -> pd.DataFrame:
        """Run RFE selection step."""
        self.rfe_selector = RFESelector()
        
        # Default parameters
        params = {
            'output_dir': str(self.output_dir / "rfe_outputs"),
            'n_features_to_select': 20,
            **kwargs
        }
        
        refined_data = self.rfe_selector.select_features_rfe(
            data, target_column, **params
        )
        
        return refined_data
    
    def run_model_training(
        self,
        data: pd.DataFrame,
        target_column: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Run model training step."""
        # Persist current data to CSV for the CrossValidationPipeline
        output_dir = Path(kwargs.get('output_dir', self.output_dir / "model_outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        data_path = output_dir / "model_training_input.csv"
        data.to_csv(data_path, index=False)

        # Map simple pipeline params to ModelTrainingConfig
        config = ModelTrainingConfig()
        config.data_path = str(data_path)
        config.target_column = target_column
        config.output_dir = str(output_dir)

        # Map model short codes to internal names
        model_codes = kwargs.get('models', ['rf', 'xgb', 'lgb'])
        model_map = {
            'rf': 'random_forest',
            'random_forest': 'random_forest',
            'xgb': 'xgboost',
            'xgboost': 'xgboost',
            'lgb': 'lightgbm',
            'lightgbm': 'lightgbm',
        }
        config.models_to_train = [
            model_map[m] for m in model_codes if m in model_map
        ]

        dl_cfg = kwargs.get('deep_learning_models', kwargs.get('dl_models'))
        if dl_cfg is not None:
            # Ensure we work with a list of strings
            if isinstance(dl_cfg, (list, tuple)):
                allowed_dl = set(config.deep_learning_models)
                selected_dl = [m for m in dl_cfg if m in allowed_dl]
                if selected_dl:
                    config.deep_learning_models = selected_dl

        # Cross-validation and sampling settings
        config.n_folds = kwargs.get('cv_folds', config.n_folds)
        config.test_size = kwargs.get('test_size', config.test_size)

        # Sampling strategy:
        # 1) If caller explicitly provides sampling_method in model_training config,
        #    honor it directly (e.g. "nearmiss", "smote", "smote_nearmiss").
        # 2) Otherwise, fall back to the existing balance_classes-based behavior
        #    (auto -> internal recommendation, none -> no sampling).
        explicit_sampling = kwargs.get('sampling_method', None)
        if explicit_sampling is not None:
            config.sampling_method = explicit_sampling
        else:
            if kwargs.get('balance_classes', True):
                config.sampling_method = 'auto'
            else:
                config.sampling_method = 'none'

        # Construct and run the cross-validation pipeline
        self.model_trainer = CrossValidationPipeline(config)
        model_results = self.model_trainer.run_pipeline(config.data_path)

        return model_results
    
    def run_interpretability_analysis(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Run interpretability analysis step."""
        # Persist current data to CSV for the InterpretabilityPipeline
        output_dir = Path(kwargs.get('output_dir', self.output_dir / "interpretability_outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        data_path = output_dir / "interpretability_input.csv"
        data.to_csv(data_path, index=False)

        # Model training output directory should match the one used in run_model_training
        model_output_dir = Path(kwargs.get('model_output_dir', self.output_dir / "model_outputs"))

        # For interpretability, point to the best_model.json metadata so that
        # ModelLoader can resolve the correct saved model directory via
        # ModelPersistenceManager, instead of passing the root saved_models dir
        # (which would lead to paths like saved_models/saved_models).
        model_path = model_output_dir / "metrics" / "best_model.json"

        # Configure interpretability settings
        config = InterpretabilityConfig()
        config.data_path = str(data_path)
        config.model_path = str(model_path)
        config.output_dir = str(output_dir)
        config.target_column = target_column

        # Optional overrides from kwargs (e.g., model_name, sample_indices, apply_sampling, show_plots)
        if 'model_name' in kwargs:
            config.model_name = kwargs['model_name']
        if 'sample_indices' in kwargs:
            config.sample_indices = kwargs['sample_indices']
        if 'apply_sampling' in kwargs:
            config.apply_sampling = kwargs['apply_sampling']
        if 'show_plots' in kwargs:
            config.show_plots = kwargs['show_plots']
        if 'interaction_features' in kwargs:
            config.interaction_features = kwargs['interaction_features']

        # Run interpretability pipeline (file-based API)
        self.interpretability_analyzer = InterpretabilityPipeline(config)
        self.interpretability_analyzer.run_analysis(str(data_path), str(model_path))

        # The current pipeline mainly writes results to disk; return a simple summary dict
        return {
            'data_path': str(data_path),
            'model_path': str(model_path),
            'output_dir': str(output_dir),
        }
    
    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Get the best performing model from the pipeline results."""
        if 'model_results' not in self.results:
            return None
            
        model_results = self.results['model_results']
        
        # Find best model based on validation score
        best_model = None
        best_score = -np.inf
        
        for model_name, model_info in model_results.items():
            if isinstance(model_info, dict) and 'validation_score' in model_info:
                score = model_info['validation_score']
                if score > best_score:
                    best_score = score
                    best_model = {
                        'name': model_name,
                        'score': score,
                        'model': model_info
                    }
        
        return best_model
    
    def get_top_features(self, n_features: int = 10) -> Optional[List[str]]:
        """Get top N important features from the best model."""
        best_model = self.get_best_model()
        
        if not best_model or 'interpretability_results' not in self.results:
            return None
            
        interpretability_results = self.results['interpretability_results']
        model_name = best_model['name']
        
        if model_name in interpretability_results:
            feature_importance = interpretability_results[model_name].get('feature_importance', {})
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                return [feature for feature, _ in sorted_features[:n_features]]
        
        return None
    
    def save_summary_report(self, filename: str = "pipeline_summary.txt"):
        """Save a summary report of the pipeline results."""
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("ClinMetML Pipeline Summary Report\n")
            f.write("=" * 40 + "\n\n")
            
            if 'cleaned_data' in self.results:
                cleaned_data = self.results['cleaned_data']
                f.write(f"Data Shape After Cleaning: {cleaned_data.shape}\n")
            
            if 'refined_data' in self.results:
                refined_data = self.results['refined_data']
                f.write(f"Final Feature Count: {refined_data.shape[1] - 1}\n")
            
            best_model = self.get_best_model()
            if best_model:
                f.write(f"Best Model: {best_model['name']}\n")
                f.write(f"Best Score: {best_model['score']:.4f}\n")
            
            top_features = self.get_top_features()
            if top_features:
                f.write(f"\nTop 10 Features:\n")
                for i, feature in enumerate(top_features, 1):
                    f.write(f"{i}. {feature}\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
