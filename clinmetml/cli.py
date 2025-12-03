"""
Command Line Interface for ClinMetML.

This module provides a command-line interface for running ClinMetML pipelines
without writing Python code.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd

from .pipeline.auto_pipeline import ClinMetMLPipeline
from . import __version__


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="ClinMetML: Automated Metabolomics Biomarker Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  clinmetml --data data.csv --target disease_status --output results/
  
  # With custom parameters
  clinmetml --data data.csv --target disease_status --output results/ \\
             --imputation knn --normalization standard --models rf xgb lgb
  
  # Using configuration file
  clinmetml --data data.csv --target disease_status --config config.json
        """
    )
    
    # Version
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'ClinMetML {__version__}'
    )
    
    # Required arguments
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to input CSV file containing metabolomics data'
    )
    
    parser.add_argument(
        '--target', 
        type=str, 
        required=True,
        help='Name of the target column in the data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', 
        type=str, 
        default='clinmetml_results',
        help='Output directory for results (default: clinmetml_results)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to JSON configuration file with pipeline parameters'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    # Data cleaning parameters
    cleaning_group = parser.add_argument_group('Data Cleaning Options')
    cleaning_group.add_argument(
        '--imputation', 
        choices=['knn', 'bpca', 'ppca', 'svd', 'cart', 'lasso'],
        default='knn',
        help='Missing value imputation method (default: knn)'
    )
    
    cleaning_group.add_argument(
        '--normalization', 
        choices=['standard', 'minmax', 'robust', 'autonorm'],
        default='standard',
        help='Data normalization method (default: standard)'
    )
    
    cleaning_group.add_argument(
        '--missing-threshold', 
        type=float, 
        default=0.2,
        help='Threshold for removing features with missing values (default: 0.2)'
    )
    
    # Feature selection parameters
    feature_group = parser.add_argument_group('Feature Selection Options')
    feature_group.add_argument(
        '--feature-method', 
        choices=['univariate', 'rfe', 'forward'],
        default='univariate',
        help='Feature selection method (default: univariate)'
    )
    
    feature_group.add_argument(
        '--k-best', 
        type=int, 
        default=100,
        help='Number of best features to select (default: 100)'
    )
    
    feature_group.add_argument(
        '--vif-threshold', 
        type=float, 
        default=5.0,
        help='VIF threshold for multicollinearity reduction (default: 5.0)'
    )
    
    feature_group.add_argument(
        '--final-features', 
        type=int, 
        default=20,
        help='Final number of features after RFE (default: 20)'
    )
    
    # Model training parameters
    model_group = parser.add_argument_group('Model Training Options')
    model_group.add_argument(
        '--models', 
        nargs='+',
        choices=['rf', 'xgb', 'lgb', 'svm', 'lr'],
        default=['rf', 'xgb', 'lgb'],
        help='Machine learning models to train (default: rf xgb lgb)'
    )
    
    model_group.add_argument(
        '--cv-folds', 
        type=int, 
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    model_group.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    model_group.add_argument(
        '--balance-classes', 
        action='store_true',
        help='Enable class balancing for imbalanced datasets'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--shap-analysis', 
        action='store_true',
        help='Enable SHAP interpretability analysis'
    )
    
    analysis_group.add_argument(
        '--top-features', 
        type=int, 
        default=10,
        help='Number of top features to display (default: 10)'
    )
    
    return parser


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def validate_inputs(args):
    """Validate input arguments."""
    # Check if data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file '{args.data}' not found.")
        sys.exit(1)
    
    # Check if data file is readable
    try:
        data = pd.read_csv(args.data, nrows=1)
        if args.target not in data.columns:
            print(f"Error: Target column '{args.target}' not found in data.")
            print(f"Available columns: {list(data.columns)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)
    
    # Validate numeric parameters
    if not 0 < args.missing_threshold <= 1:
        print("Error: missing-threshold must be between 0 and 1.")
        sys.exit(1)
    
    if not 0 < args.test_size < 1:
        print("Error: test-size must be between 0 and 1.")
        sys.exit(1)
    
    if args.cv_folds < 2:
        print("Error: cv-folds must be at least 2.")
        sys.exit(1)


def build_pipeline_config(args):
    """Build pipeline configuration from command line arguments."""
    config = {
        'data_cleaning': {
            'imputation_method': args.imputation,
            'filter_missing_threshold': args.missing_threshold,
            'normalization_method': args.normalization,
            'log_transform': True,
            'handle_outliers': True
        },
        'feature_selection': {
            'method': args.feature_method,
            'k_best': args.k_best,
            'score_func': 'f_classif'
        },
        'multicollinearity_reduction': {
            'vif_threshold': args.vif_threshold,
            'correlation_threshold': 0.9
        },
        'rfe_selection': {
            'n_features_to_select': args.final_features,
            'estimator': 'rf'
        },
        'model_training': {
            'models': args.models,
            'cv_folds': args.cv_folds,
            'test_size': args.test_size,
            'balance_classes': args.balance_classes,
            'hyperparameter_tuning': True
        },
        'interpretability': {
            'shap_analysis': args.shap_analysis,
            'feature_importance': True,
            'generate_plots': True,
            'plot_top_features': args.top_features
        }
    }
    
    return config


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = build_pipeline_config(args)
    
    # Print startup information
    print(f"🧬 ClinMetML v{__version__}")
    print("=" * 50)
    print(f"📊 Data file: {args.data}")
    print(f"🎯 Target column: {args.target}")
    print(f"📁 Output directory: {args.output}")
    print(f"🔢 Random state: {args.random_state}")
    
    # Load data
    print(f"\n📈 Loading data...")
    try:
        data = pd.read_csv(args.data)
        print(f"Data shape: {data.shape}")
        print(f"Target distribution: {data[args.target].value_counts().to_dict()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Initialize pipeline
    print(f"\n🚀 Initializing ClinMetML pipeline...")
    pipeline = ClinMetMLPipeline(
        output_dir=args.output,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # Run pipeline
    try:
        print(f"\n⚙️ Running ClinMetML pipeline...")
        results = pipeline.run_full_pipeline(
            data=data,
            target_column=args.target,
            **config
        )
        
        # Display results
        print(f"\n📊 Pipeline Results:")
        print("-" * 30)
        
        # Data processing summary
        print(f"Original data: {data.shape}")
        print(f"Final refined data: {results['refined_data'].shape}")
        
        # Model performance
        print(f"\n🏆 Model Performance:")
        model_results = results['model_results']
        for model_name, model_info in model_results.items():
            if isinstance(model_info, dict) and 'validation_score' in model_info:
                score = model_info['validation_score']
                print(f"  {model_name}: {score:.4f}")
        
        # Best model
        best_model = pipeline.get_best_model()
        if best_model:
            print(f"\n🥇 Best Model: {best_model['name']} (Score: {best_model['score']:.4f})")
        
        # Top features
        top_features = pipeline.get_top_features(n_features=args.top_features)
        if top_features:
            print(f"\n🎯 Top {args.top_features} Biomarkers:")
            for i, feature in enumerate(top_features, 1):
                print(f"  {i}. {feature}")
        
        # Generate summary
        pipeline.save_summary_report("cli_analysis_summary.txt")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
