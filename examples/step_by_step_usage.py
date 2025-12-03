"""Step-by-step usage example for ClinMetML package.

This example demonstrates how to use individual ClinMetML components
for more granular control over the analysis pipeline, using the same
configuration as in the tests/README (disease/id columns, multiple
feature selection methods, multicollinearity reduction, RFE, model
training, and interpretability).
"""

import pandas as pd
import numpy as np
from pathlib import Path

from clinmetml import (
    DataCleaner,
    FeatureRefinementSelector,
    CrossValidationPipeline,
    InterpretabilityPipeline,
    feature_selector,
    multicollinearity_reducer,
)
from clinmetml.core.model_trainer import ModelTrainingConfig
from clinmetml.core.interpretability_analyzer import InterpretabilityConfig

def load_sample_data() -> pd.DataFrame:
    """Load or create sample data for the example.

    This uses synthetic data with an integer ID column and a binary
    disease outcome column named ``disease``.
    """

    np.random.seed(42)

    n_samples = 500
    n_features = 200

    feature_names = [f"metabolite_{i:03d}" for i in range(n_features)]
    data = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_features))

    # Add missing values
    missing_mask = np.random.random((n_samples, n_features)) < 0.15
    data[missing_mask] = np.nan

    # Create binary disease outcome
    target = np.random.binomial(1, 0.4, n_samples)

    # Make some features predictive
    predictive_indices = [10, 25, 50, 75, 100]
    for idx in predictive_indices:
        data[target == 1, idx] *= 1.5

    df = pd.DataFrame(data, columns=feature_names)
    df.insert(0, "id", np.arange(n_samples))
    df["disease"] = target

    return df

def main():
    """Run step-by-step ClinMetML analysis."""
    print("🔬 ClinMetML Step-by-Step Analysis")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("clinmetml_stepwise_example")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📊 Loading sample data...")
    data = load_sample_data()
    print(f"Initial data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Target distribution: {data['disease'].value_counts().to_dict()}")
    
    # Step 1: Data Cleaning
    print("\n🧹 Step 1: Data Cleaning")
    print("-" * 25)
    
    cleaner = DataCleaner()

    cleaning_params = {
        'imputation_method': 'knn',
        'filter_missing_threshold': 0.3,
        'normalization': 'standard',
        'log_transform': True,
        'autonorm': True,
        'target_column': 'disease',
        'id_column': 'id',
        'quality_plots_dir': str(output_dir / "data_cleaning"),
    }

    cleaned_data = cleaner.clean_pipeline(data, **cleaning_params)
    print(f"After cleaning: {cleaned_data.shape}")
    
    # Save cleaned data
    cleaned_data.to_csv(output_dir / "cleaned_data.csv", index=False)
    
    # Step 2: Feature Selection
    print("\n🎯 Step 2: Feature Selection (auto)")
    print("-" * 35)

    cleaned_path = output_dir / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False)

    fs_output_dir = output_dir / "feature_selection"
    fs_output_dir.mkdir(parents=True, exist_ok=True)

    fs_results = feature_selector.run_auto_feature_selection(
        data_path=str(cleaned_path),
        target_col="disease",
        methods=["randomforest", "elasticnet", "relief", "lightgbm", "fcbf", "mrmr"],
        top_k=40,
        output_dir=str(fs_output_dir),
        id_col="id",
        balance_threshold=0.3,
        n_iterations=3,
        sample_ratio=0.75,
        match_ratio=3,
        random_state=123,
        match_cols=["age", "sex"],  # For real data, replace with your covariates
    )

    final_feature_path = fs_output_dir / "final_feature_dataset.csv"
    selected_data = pd.read_csv(final_feature_path)
    print(f"After feature selection: {selected_data.shape}")
    
    # Step 3: Multicollinearity Reduction
    print("\n📊 Step 3: Multicollinearity Reduction")
    print("-" * 35)
    
    # Multicollinearity reduction using the functional API
    mc_output_dir = output_dir / "multicollinearity"
    mc_output_dir.mkdir(parents=True, exist_ok=True)

    reduced_data, corr_matrix, analysis_results = multicollinearity_reducer.remove_collinearity(
        data=selected_data,
        target_col="disease",
        correlation_threshold=0.9,
        distance_cutoff=1.2,
        method="spearman",
        visualize=False,
        save_plots=False,
        output_dir=str(mc_output_dir),
    )
    print(f"After multicollinearity reduction: {reduced_data.shape}")
    
    # Step 4: RFE Selection
    print("\n🔍 Step 4: RFE Feature Selection")
    print("-" * 30)
    
    rfe_output_dir = output_dir / "rfe_selection"
    rfe_output_dir.mkdir(parents=True, exist_ok=True)

    rfe_selector = FeatureRefinementSelector(
        target_column="disease",
        algorithm="random_forest",
        test_size=0.3,
        random_state=42,
        sampling_strategy="auto",
        output_dir=str(rfe_output_dir),
        n_features_to_select=20,
        id_column="id",
    )

    rfe_selector.run_complete_analysis(
        data_path=str(mc_output_dir / "refined_dataset.csv"),
        highlight_features=None,
        save_selected_data=True,
        selected_data_filename="rfe_selected_data.csv",
    )

    refined_data = pd.read_csv(rfe_output_dir / "rfe_selected_data.csv")
    print(f"After RFE selection: {refined_data.shape}")
    
    # Save refined data
    refined_data.to_csv(output_dir / "refined_data.csv", index=False)
    
    # Step 5: Model Training
    print("\n🤖 Step 5: Model Training (CrossValidationPipeline)")
    print("-" * 45)

    mt_output_dir = output_dir / "model_training"
    mt_output_dir.mkdir(parents=True, exist_ok=True)

    mt_config = ModelTrainingConfig(
        data_path=str(rfe_output_dir / "rfe_selected_data.csv"),
        target_column="disease",
        models_to_train=["random_forest", "xgboost", "lightgbm"],
        sampling_method="nearmiss",
        n_trials=1,
        n_folds=2,
        id_column="id",
        output_dir=str(mt_output_dir),
    )

    cv_pipeline = CrossValidationPipeline(mt_config)
    model_results = cv_pipeline.run_pipeline(mt_config.data_path)
    
    # Display model performance
    print("\n📈 Model Performance:")
    for model_name, model_info in model_results.items():
        if isinstance(model_info, dict) and 'validation_score' in model_info:
            score = model_info['validation_score']
            print(f"  {model_name}: {score:.4f}")
    
    # Step 6: Model Interpretability
    print("\n🔍 Step 6: Model Interpretability Analysis")
    print("-" * 40)

    interpret_output_dir = output_dir / "interpretability"
    interpret_output_dir.mkdir(parents=True, exist_ok=True)

    iconfig = InterpretabilityConfig()
    iconfig.data_path = str(interpret_output_dir / "interpretability_input.csv")
    iconfig.model_path = str(mt_output_dir / "metrics" / "best_model.json")
    iconfig.output_dir = str(interpret_output_dir)
    iconfig.target_column = "disease"

    # Persist refined data for interpretability pipeline
    refined_data.to_csv(iconfig.data_path, index=False)

    ipipeline = InterpretabilityPipeline(iconfig)
    ipipeline.run_analysis(iconfig.data_path, iconfig.model_path)

    interpretability_results = {
        'config': iconfig,
    }
    
    # Display top features from best model
    print("\n🏆 Feature Importance (Best Model):")
    
    # Find best model
    best_model_name = None
    best_score = -1
    for model_name, model_info in model_results.items():
        if isinstance(model_info, dict) and 'validation_score' in model_info:
            if model_info['validation_score'] > best_score:
                best_score = model_info['validation_score']
                best_model_name = model_name
    
    if best_model_name and best_model_name in interpretability_results:
        feature_importance = interpretability_results[best_model_name].get('feature_importance', {})
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {i}. {feature}: {importance:.4f}")
    
    # Step 7: Generate Summary Report
    print("\n📄 Step 7: Generate Summary Report")
    print("-" * 35)
    
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ClinMetML Step-by-Step Analysis Summary\n")
        f.write("=" * 45 + "\n\n")
        
        f.write(f"Initial data shape: {data.shape}\n")
        f.write(f"After cleaning: {cleaned_data.shape}\n")
        f.write(f"After feature selection: {selected_data.shape}\n")
        f.write(f"After multicollinearity reduction: {reduced_data.shape}\n")
        f.write(f"Final refined data: {refined_data.shape}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for model_name, model_info in model_results.items():
            if isinstance(model_info, dict) and 'validation_score' in model_info:
                score = model_info['validation_score']
                f.write(f"{model_name}: {score:.4f}\n")
        
        if best_model_name:
            f.write(f"\nBest Model: {best_model_name} (Score: {best_score:.4f})\n")
        
        if feature_importance:
            f.write(f"\nTop 10 Features:\n")
            for i, (feature, importance) in enumerate(sorted_features, 1):
                f.write(f"{i}. {feature}: {importance:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")
    print(f"\n✅ Step-by-step analysis complete!")
    print(f"📁 All results saved in: {output_dir}")
    
    return {
        'cleaned_data': cleaned_data,
        'selected_data': selected_data,
        'reduced_data': reduced_data,
        'refined_data': refined_data,
        'model_results': model_results,
        'interpretability_results': interpretability_results
    }

if __name__ == "__main__":
    results = main()
