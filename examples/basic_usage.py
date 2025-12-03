"""
Basic usage example for ClinMetML package.

This example demonstrates how to use ClinMetML for a complete
metabolomics biomarker discovery pipeline.
"""

import pandas as pd
import numpy as np
from clinmetml import ClinMetMLPipeline

def create_sample_data():
    """Create sample metabolomics data for demonstration."""
    np.random.seed(42)
    
    # Create sample data with 1000 samples and 500 metabolites
    n_samples = 1000
    n_features = 500
    
    # Generate feature names (metabolite IDs)
    feature_names = [f"metabolite_{i:03d}" for i in range(n_features)]
    
    # Generate random metabolomics data
    data = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_features))
    
    # Add some missing values (realistic for metabolomics data)
    missing_mask = np.random.random((n_samples, n_features)) < 0.1
    data[missing_mask] = np.nan
    
    # Create target variable (disease status: 0=healthy, 1=disease)
    # Make some features more predictive
    predictive_features = np.random.choice(n_features, size=20, replace=False)
    target = np.zeros(n_samples)
    
    for i, feature_idx in enumerate(predictive_features):
        # Higher values of certain metabolites associated with disease
        effect_size = np.random.uniform(0.5, 2.0)
        target += data[:, feature_idx] * effect_size
    
    # Add noise and convert to binary
    target += np.random.normal(0, 1, n_samples)
    target = (target > np.median(target)).astype(int)
    
    # Create DataFrame with an ID column and disease target
    df = pd.DataFrame(data, columns=feature_names)
    df.insert(0, 'id', np.arange(n_samples))
    df['disease'] = target
    
    return df

def main():
    """Run the basic ClinMetML pipeline example."""
    print("🧬 ClinMetML Basic Usage Example")
    print("=" * 40)
    
    # Step 1: Create or load sample data
    print("\n📊 Creating sample metabolomics data...")
    data = create_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"Target distribution: {data['disease_status'].value_counts().to_dict()}")
    
    # Step 2: Initialize ClinMetML pipeline
    print("\n🚀 Initializing ClinMetML pipeline...")
    pipeline = ClinMetMLPipeline(
        output_dir="clinmetml_basic_example",
        random_state=42,
        verbose=True
    )
    
    # Step 3: Configure pipeline parameters
    pipeline_config = {
        'data_cleaning': {
            'imputation_method': 'knn',
            'filter_missing_threshold': 0.3,
            'normalization': 'standard',
            'log_transform': True,
            'autonorm': True,
            'target_column': 'disease',
            'id_column': 'id',
        },
        'feature_selection': {
            'method': ['randomforest', 'elasticnet', 'relief', 'lightgbm', 'fcbf', 'mrmr'],
            'k_best': 40,
            'id_column': 'id',
            'balance_threshold': 0.3,
            'n_iterations': 3,
            'sample_ratio': 0.75,
            'match_ratio': 3,
            'random_state': 123,
            'match_cols': ['age', 'sex'],  # For real data, replace with your covariates
        },
        'multicollinearity_reduction': {
            'vif_threshold': 5.0,
            'correlation_threshold': 0.9,
            'distance_cutoff': 1.2,
            'method': 'spearman',
            'visualize': False,
            'save_plots': False,
        },
        'rfe_selection': {
            'n_features_to_select': 20,
            'estimator': 'rf',
            'test_size': 0.3,
            'random_state': 42,
            'sampling_strategy': 'auto',
            'id_column': 'id',
        },
        'model_training': {
            'models': ['rf', 'xgb', 'lgb'],
            'cv_folds': 5,
            'test_size': 0.3,
            'balance_classes': True,
            'sampling_method': 'nearmiss',
            'id_column': 'id',
        },
        'interpretability': {
            'shap_analysis': True,
            'feature_importance': True,
            'generate_plots': True,
        }
    }
    
    # Step 4: Run the complete pipeline
    print("\n⚙️ Running complete ClinMetML pipeline...")
    results = pipeline.run_full_pipeline(
        data=data,
        target_column='disease',
        **pipeline_config
    )
    
    # Step 5: Analyze results
    print("\n📈 Pipeline Results:")
    print("-" * 20)
    
    # Data processing results
    print(f"Original data shape: {data.shape}")
    print(f"After cleaning: {results['cleaned_data'].shape}")
    print(f"After feature selection: {results['selected_data'].shape}")
    print(f"After multicollinearity reduction: {results['reduced_data'].shape}")
    print(f"Final refined data: {results['refined_data'].shape}")
    
    # Model performance results
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
    
    # Top biomarkers
    top_features = pipeline.get_top_features(n_features=10)
    if top_features:
        print(f"\n🎯 Top 10 Biomarkers:")
        for i, feature in enumerate(top_features, 1):
            print(f"  {i}. {feature}")
    
    # Generate summary report
    print(f"\n📄 Generating summary report...")
    pipeline.save_summary_report("basic_example_summary.txt")
    
    print(f"\n✅ Analysis complete! Check 'clinmetml_basic_example' directory for detailed results.")
    
    return results

if __name__ == "__main__":
    results = main()
