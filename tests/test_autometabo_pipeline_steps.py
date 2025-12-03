"""Comprehensive test script for ClinMetML.

This script is intended for quick end-to-end sanity checking that all
core components of ClinMetML can run without errors on a small
synthetic dataset, including:

1. DataCleaner
2. FeatureSelector
3. MulticollinearityReducer
4. RFESelector
5. ModelTrainer
6. InterpretabilityAnalyzer
7. ClinMetMLPipeline.run_full_pipeline

Run:
    cd /home/ljr/tabular_biomarker/ClinMetML
    python -m tests.test_clinmetml_pipeline_steps

The script only prints shapes and key metrics so that it runs relatively
fast and is easy to inspect.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import components
from clinmetml.core.data_cleaner import DataCleaner
from clinmetml.core import feature_selector, multicollinearity_reducer
from clinmetml.core.rfe_selector import FeatureRefinementSelector
from clinmetml.core.model_trainer import ModelTrainingConfig, CrossValidationPipeline
from clinmetml.core.interpretability_analyzer import InterpretabilityConfig
from clinmetml.core.interpretability_analyzer import InterpretabilityPipeline
from clinmetml.pipeline.auto_pipeline import ClinMetMLPipeline


def load_mafld_mdd_woman_data() -> pd.DataFrame:
    """Load the demo dataset for testing.

    The file is expected at ClinMetML/demo_data/MAFLD_MDD_woman.csv
    relative to this test module.
    """
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "demo_data" / "data1.csv"
    df = pd.read_csv(data_path)
    return df


def run_stepwise_tests(output_dir: Path):
    print("\n===== STEPWISE COMPONENT TESTS =====")
    output_dir.mkdir(exist_ok=True)

    data = load_mafld_mdd_woman_data()
    print(f"Initial data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Target distribution: {data['disease'].value_counts().to_dict()}")

    # 1. Data cleaning
    print("\n[1] Data Cleaning ...")
    cleaner = DataCleaner()
    cleaning_params = {
        "imputation_method": "knn",
        "filter_missing_threshold": 0.3,
        "normalization": "standard",
        "log_transform": True,
        "autonorm": True,
        "quality_plots_dir": str(output_dir / "data_quality_plots"),
        "target_column": "disease",
        "id_column": "id",
    }
    cleaned_data = cleaner.clean_pipeline(data, **cleaning_params)
    print(f"After cleaning: {cleaned_data.shape}")
    cleaned_path = output_dir / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False)

    # 2. Feature selection (auto: balanced vs resampling)
    print("\n[2] Feature Selection (auto) ...")
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
        match_cols=["age", "gender"],
    )
    print("Feature selection result files:")
    for method, path in fs_results.items():
        print(f"  {method}: {path}")

    # 3. Multicollinearity reduction (on final feature dataset)
    print("\n[3] Multicollinearity Reduction ...")
    mc_output_dir = output_dir / "multicollinearity"
    mc_output_dir.mkdir(parents=True, exist_ok=True)

    # Load final feature dataset produced by run_auto_feature_selection
    final_feature_path = fs_output_dir / "final_feature_dataset.csv"
    if not final_feature_path.exists():
        raise FileNotFoundError(f"Final feature dataset not found at {final_feature_path}. "
                                "Please check run_auto_feature_selection outputs.")
    final_feature_data = pd.read_csv(final_feature_path)
    print(f"Final feature dataset shape: {final_feature_data.shape}")

    refined_no_collinearity, corr_matrix, analysis_results = multicollinearity_reducer.remove_collinearity(
        data=final_feature_data,
        target_col="disease",
        correlation_threshold=0.9,
        distance_cutoff=1.2,
        method="spearman",
        visualize=False,
        save_plots=False,
        output_dir=str(mc_output_dir),
    )
    print(f"After multicollinearity reduction: {refined_no_collinearity.shape}")

    # 使用内置工具在原始数据基础上去除共线性特征，同时保留目标列
    refined_dataset_path = multicollinearity_reducer.save_refined_dataset(
        original_data_path=str(final_feature_path),
        removed_features=analysis_results["removed_features"],
        output_dir=str(mc_output_dir),
        target_col="disease",
        remove_covariates=False,
        covariates=None,
    )
    no_collinearity_path = Path(refined_dataset_path)

    # 4. RFE feature refinement
    print("\n[4] RFE Feature Refinement ...")
    rfe_output_dir = output_dir / "rfe_selection"
    rfe_output_dir.mkdir(parents=True, exist_ok=True)
    rfe_selector = FeatureRefinementSelector(
        target_column="disease",
        algorithm="random_forest",
        test_size=0.3,
        random_state=42,
        sampling_strategy="auto",
        output_dir=str(rfe_output_dir),
        n_features_to_select=10,
        id_column="id",
    )
    rfe_selector.run_complete_analysis(
        data_path=str(no_collinearity_path),
        highlight_features=None,
        save_selected_data=True,
        selected_data_filename="rfe_selected_data.csv",
    )
    rfe_selected_path = rfe_output_dir / "rfe_selected_data.csv"
    print(f"RFE selected data saved to: {rfe_selected_path}")

    # 5. Model training (cross-validation pipeline)
    print("\n[5] Model Training (CrossValidationPipeline) ...")
    mt_output_dir = output_dir / "model_training"
    mt_output_dir.mkdir(parents=True, exist_ok=True)

    mt_config = ModelTrainingConfig(
        data_path=str(rfe_selected_path),
        target_column="disease",
        output_dir=str(mt_output_dir),
        models_to_train=["random_forest", "xgboost"],
        deep_learning_models=["ft_transformer", "gandalf"],
        sampling_method="nearmiss",
        n_trials=1,
        n_folds=2,
        id_column="id",
        threshold_limits=(0.1, 0.9),  # 这里设置 DCA 的阈值范围
    )
    cv_pipeline = CrossValidationPipeline(mt_config)
    _ = cv_pipeline.run_pipeline(mt_config.data_path)
    print("Model training pipeline completed. Check model_training outputs for details.")

    # 6. Model interpretability (stepwise interpretability test)
    print("\n[6] Model Interpretability (InterpretabilityPipeline) ...")
    interpret_output_dir = output_dir / "interpretability"
    interpret_output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare interpretability input data: reuse the RFE-selected dataset that
    # still contains the target column 'disease'. This avoids dropping the
    # target (which happens in refined_no_collinearity) and matches the data
    # used for final model training.
    interpret_data_path = rfe_selected_path

    # Configure interpretability settings to mirror step-by-step example and README
    iconfig = InterpretabilityConfig()
    iconfig.data_path = str(interpret_data_path)
    iconfig.model_path = str(mt_output_dir / "metrics" / "best_model.json")
    iconfig.output_dir = str(interpret_output_dir)
    iconfig.target_column = "disease"

    ipipeline = InterpretabilityPipeline(iconfig)
    ipipeline.run_analysis(iconfig.data_path, iconfig.model_path)
    print("Interpretability analysis completed. Check interpretability outputs for SHAP results and plots.")

    return {
        "cleaned_data": cleaned_data,
        "no_collinearity_data": refined_no_collinearity,
        "rfe_selected_path": str(rfe_selected_path),
    }


def run_full_pipeline_test(data: pd.DataFrame, output_dir: Path):
    print("\n===== FULL PIPELINE TEST (ClinMetMLPipeline) =====")

    # Ensure parent directory exists before passing nested path to ClinMetMLPipeline
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ClinMetMLPipeline(
        output_dir=str(output_dir / "clinmetml_pipeline_test"),
        random_state=42,
        verbose=True,
    )

    pipeline_config = {
        "data_cleaning": {
            "imputation_method": "knn",
            "filter_missing_threshold": 0.3,
            "normalization": "standard",
            "log_transform": True,
            "autonorm": True,
            "target_column": "disease",
            "id_column": "id",
            "quality_plots_dir": str(output_dir / "data_quality_plots"),
        },
        "feature_selection": {
            "method": ["randomforest", "elasticnet", "relief", "lightgbm", "fcbf", "mrmr"],
            "k_best": 30,
            "score_func": "f_classif",
            "id_column": "id",
            "match_cols": ["age", "gender"], 
        },
            
        
        "multicollinearity_reduction": {
            # Match stepwise: remove_collinearity(data, target_col, correlation_threshold=0.9,
            # distance_cutoff=1.2, method="spearman", visualize=False, save_plots=False,...)
            "vif_threshold": 5.0,
            "correlation_threshold": 0.9,
            "distance_cutoff": 1.2,
            "method": "spearman",
            "visualize": False,
            "save_plots": False,
        },
        "rfe_selection": {
            # Match stepwise RFE: n_features_to_select=10, algorithm="random_forest",
            # test_size=0.3, random_state=42, sampling_strategy="auto", id_column="id"
            "n_features_to_select": 10,
            "estimator": "rf",
            "test_size": 0.3,
            "random_state": 42,
            "sampling_strategy": "auto",
            "id_column": "id",
        },
        "model_training": {
            # Match stepwise cross-validation settings as closely as possible
            "models": ["rf", "xgb", "lgb"],
            "cv_folds": 2,
            "test_size": 0.3,
            "balance_classes": True,
            "id_column": "id",
            "sampling_method": "nearmiss",
            "deep_learning_models":["ft_transformer", "gandalf"],
        },
        "interpretability": {
            "shap_analysis": True,
            "feature_importance": True,
            "generate_plots": True,
        },
    }

    results = pipeline.run_full_pipeline(
        data=data,
        target_column="disease",
        **pipeline_config,
    )

    print("\nFull pipeline data shapes:")
    print(f"  cleaned_data: {results['cleaned_data'].shape}")
    print(f"  selected_data: {results['selected_data'].shape}")
    print(f"  reduced_data: {results['reduced_data'].shape}")
    print(f"  refined_data: {results['refined_data'].shape}")

    print("\nFull pipeline model validation scores:")
    for model_name, model_info in results["model_results"].items():
        if isinstance(model_info, dict) and "validation_score" in model_info:
            print(f"  {model_name}: {model_info['validation_score']:.4f}")

    best_model = pipeline.get_best_model()
    if best_model:
        print(
            f"Best model from full pipeline: {best_model['name']} "
            f"(score={best_model['score']:.4f})"
        )

    top_features = pipeline.get_top_features(n_features=10)
    if top_features:
        print("Top 10 features from full pipeline:")
        for i, feat in enumerate(top_features, 1):
            print(f"  {i}. {feat}")

    pipeline.save_summary_report("pipeline_test_summary.txt")

    return results


def main():
    base_output_dir = Path("clinmetml_tests")
    base_output_dir.mkdir(exist_ok=True)

    # Stepwise tests (using MAFLD_MDD_woman.csv)
    stepwise_results = run_stepwise_tests(base_output_dir / "stepwise")

    # Full pipeline test using the same real dataset
    data_for_full = load_mafld_mdd_woman_data()
    _ = run_full_pipeline_test(data_for_full, base_output_dir / "full_pipeline")

    print("\nAll tests finished. Check the 'clinmetml_tests' directory for outputs.")
    # Stepwise tests are currently disabled; main does not need to return their results.
    return None


if __name__ == "__main__":
    main()
