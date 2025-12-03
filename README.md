# ClinMetML 🧬

**An end-to-end automated framework for streamlining metabolomics biomarker discovery and robust clinical predictive modeling.**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/clinmetml.svg)](https://badge.fury.io/py/clinmetml)


ClinMetML is a flexible Python framework for metabolomics that prioritizes clinical utility. It features a rigorous multi-stage feature refinement pipeline—integrating ensemble selection, RFE ranking, forward addition, and collinearity checks—to identify the smallest, most predictive biomarker subset. With built-in support for imbalanced learning, tabular Deep Learning, SHAP, and Decision Curve Analysis (DCA), ClinMetML streamlines the development of robust and interpretable diagnostic models.

## 🚀 Key Features

- **🧹 Rigorous Data Cleaning**: End-to-end preprocessing with data quality reports, flexible missing-value imputation (KNN, BPCA, PPCA, SVD, CART-PMM, Lasso), missingness-based feature filtering, log2 transform, and robust Autonorm scaling.
- **🎯 Robust Multi-Stage Feature Selection**: Imbalance-aware resampling with covariate-based nearest-neighbor matching, ensemble feature selection across multiple algorithms (Random Forest, LightGBM, ElasticNet, FCBF, Relief, mRMR), multi-iteration robustness analysis and feature voting, and creation of a final consensus feature set.
- **📊 Multicollinearity & Panel Refinement**: Hierarchical clustering on correlation structure to remove redundant predictors, plus RFE-based ranking and forward feature addition with AUC monitoring to identify the smallest, non-redundant biomarker panel that preserves performance.
- **🤖 Flexible Model Training**: Cross-validated training of diverse models, including Random Forest, XGBoost, LightGBM and a suite of deep learning models for tabular data (via `pytorch-tabular`), with configurable sampling strategies (SMOTE/ADASYN/NearMiss/hybrids) and comprehensive metrics (ROC-AUC, F1, MCC, a custom BPS composite score, calibration).
- **� Transparent Interpretability**: SHAP-based global and local explanations for both tree-based and deep learning models, including importance plots, summary plots, waterfall/force plots, and interaction analyses.
- **⚕️ Clinical Utility (DCA)**: Built-in Decision Curve Analysis to compute net benefit curves for the best model across risk thresholds, directly comparing "treat-all"/"treat-none" strategies and quantifying clinical usefulness.
- **📈 Reproducible Outputs & Reporting**: Structured output directories for each step (cleaning, feature selection, multicollinearity reduction, RFE, model training, interpretability), plus summary files and logs to support auditing and manuscript preparation.

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install clinmetml
```

### From Source
```bash
git clone https://github.com/Li-OmicsLab-MPU/ClinMetML.git
cd clinmetml
pip install -e .
```

### With Optional Dependencies
```bash
# For deep learning support
pip install clinmetml[deep-learning]

# For development
pip install clinmetml[dev]

# Install all optional dependencies
pip install clinmetml[all]
```

> To enable deep learning models in ClinMetML, make sure `pytorch-tabular` is installed in your environment:
> `pip install pytorch-tabular` or `pip install "clinmetml[deep-learning]"`.

## Quick Start

### Simple Pipeline Usage

```python
import pandas as pd
from clinmetml import ClinMetMLPipeline

# Load your metabolomics data
data = pd.read_csv("your_metabolomics_data.csv")

# Initialize the pipeline
pipeline = ClinMetMLPipeline(
    output_dir="my_analysis_results",
    random_state=42,
    verbose=True
)

# Run the complete analysis pipeline (configuration as used in our paper)
results = pipeline.run_full_pipeline(
    data=data,
    target_column="disease",
    data_cleaning={
        # Match the stepwise tests / paper configuration
        'imputation_method': 'knn',
        'filter_missing_threshold': 0.3,
        'normalization': 'standard',
        'log_transform': True,
        'autonorm': True,
        'target_column': 'disease',
        'id_column': 'id',
    },
    feature_selection={
        # Use multiple feature selection methods as in the paper/tests
        'method': ['randomforest', 'elasticnet', 'relief', 'lightgbm', 'fcbf', 'mrmr'],
        'k_best': 40,
        'id_column': 'id',
        'balance_threshold': 0.3,
        'n_iterations': 3,
        'sample_ratio': 0.75,
        'match_ratio': 3,
        'random_state': 123,
        'match_cols': ['age', 'sex'],
    },
    multicollinearity_reduction={
        # Match stepwise multicollinearity reduction settings
        'vif_threshold': 5.0,
        'correlation_threshold': 0.9,
        'distance_cutoff': 1.2,
        'method': 'spearman',
        'visualize': False,
        'save_plots': False,
    },
    model_training={
        # Match the model training setup used in the tests/paper
        'models': ['rf', 'xgb', 'lgb'],
        'deep_learning_models': ['ft_transformer', 'gandalf'],
        'cv_folds': 5,
        'test_size': 0.3,
        'balance_classes': True,
        'sampling_method': 'nearmiss',
        'id_column': 'id',
        # Optional: control the DCA risk threshold range (low, high)
        'threshold_limits': (0.1, 0.9),
    },
    interpretability={
        # Enable SHAP-based interpretability analysis after training
        'shap_analysis': True,
        'feature_importance': True,
        'generate_plots': True,
    },
)

# Get the best model
best_model = pipeline.get_best_model()
print(f"Best model: {best_model['name']} (Score: {best_model['score']:.4f})")

# Get top biomarkers
top_features = pipeline.get_top_features(n_features=10)
print("Top 10 biomarkers:", top_features)
```

### Step-by-Step Usage

```python
import pandas as pd

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


# Step 1: Data Cleaning
data = pd.read_csv("your_metabolomics_data.csv")

cleaner = DataCleaner()
cleaned_data = cleaner.clean_pipeline(
    data,
    imputation_method="knn",
    filter_missing_threshold=0.3,
    normalization="standard",
    log_transform=True,
    autonorm=True,
    target_column="disease",
    id_column="id",
)

# Step 2: Feature Selection (using the feature_selector module)
cleaned_path = "cleaned_data.csv"
cleaned_data.to_csv(cleaned_path, index=False)

fs_results = feature_selector.run_auto_feature_selection(
    data_path=cleaned_path,
    target_col="disease",
    methods=["randomforest", "elasticnet", "relief", "lightgbm", "fcbf", "mrmr"],
    top_k=40,
    output_dir="feature_selection_outputs",
    id_col="id",
    balance_threshold=0.3,
    n_iterations=3,
    sample_ratio=0.75,
    match_ratio=3,
    random_state=123,
    match_cols=["age", "sex"],
)

final_feature_path = "feature_selection_outputs/final_feature_dataset.csv"
selected_data = pd.read_csv(final_feature_path)

# Step 3: Reduce Multicollinearity
reduced_data, corr_matrix, analysis_results = multicollinearity_reducer.remove_collinearity(
    data=selected_data,
    target_col="disease",
    correlation_threshold=0.9,
    distance_cutoff=1.2,
    method="spearman",
    visualize=False,
    save_plots=False,
    output_dir="multicollinearity_outputs",
)

# Step 4: RFE Selection
rfe_selector = FeatureRefinementSelector(
    target_column="disease",
    algorithm="random_forest",
    test_size=0.3,
    random_state=42,
    sampling_strategy="nearmiss",
    output_dir="rfe_outputs",
    n_features_to_select=20,
    id_column="id",
)

rfe_selector.run_complete_analysis(
    data_path="multicollinearity_outputs/refined_dataset.csv",
    highlight_features=None,
    save_selected_data=True,
    selected_data_filename="rfe_selected_data.csv",
)

refined_data = pd.read_csv("rfe_outputs/rfe_selected_data.csv")

# Step 5: Model Training (CrossValidationPipeline)
mt_config = ModelTrainingConfig(
    data_path="rfe_outputs/rfe_selected_data.csv",
    target_column="disease",
    models_to_train=["random_forest", "xgboost", "lightgbm"],
    deep_learning_models=["ft_transformer", "gandalf"],
    sampling_method="nearmiss",
    n_trials=1,
    n_folds=2,
    id_column="id",
)

cv_pipeline = CrossValidationPipeline(mt_config)
model_results = cv_pipeline.run_pipeline(mt_config.data_path)

# Step 6: Model Interpretability
iconfig = InterpretabilityConfig()
iconfig.data_path = "interpretability_outputs/interpretability_input.csv"
iconfig.model_path = "model_outputs/metrics/best_model.json"
iconfig.target_column = "disease"

ipipeline = InterpretabilityPipeline(iconfig)
ipipeline.run_analysis(iconfig.data_path, iconfig.model_path)
```

## 📊 Supported Algorithms

### Data Cleaning Methods
- **Missing Value Imputation**: KNN, BPCA, PPCA, SVD, CART-PMM, Lasso
- **Normalization**: Standard, MinMax, Robust, Log2, Autonorm
- **Outlier Detection**: IQR-based detection and visualization

### Feature Selection Methods
- **Tree-based Importance**: Random Forest and LightGBM feature importance
- **Regularized Regression**: ElasticNet-based feature ranking
- **Filter Methods**: FCBF and mRMR (if the corresponding packages are installed)
- **Relief-based Methods**: ReliefF 
 - **RFE-based Refinement**: RFE ranking with forward feature addition and AUC-based refinement (via `rfe_selector`)

### Multicollinearity Reduction
- **Correlation- and Clustering-based Filtering**: Spearman/Pearson correlation matrix with hierarchical clustering to group features and remove redundant predictors

### Machine Learning Models
- **Traditional ML**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Decision Tree
- **Deep Learning**: TabNet, FT-Transformer, GANDALF, NODE (with pytorch-tabular)


### Model Interpretability
- **SHAP-based Analysis**: Tree/linear/deep models with automatic explainer selection
- **Global Explanations**: Feature importance bar plots and SHAP summary plots
- **Local Explanations**: Waterfall and force plots for individual samples
- **Feature Interaction Analysis**: SHAP-based interaction and effect visualization

## ⚕️ Decision Curve Analysis (DCA)

ClinMetML provides built-in **Decision Curve Analysis (DCA)** to evaluate the
clinical usefulness of the best-performing model over a user-defined risk
threshold range (e.g. 0.1–0.9).

- Computes net benefit curves for the model and compares them to "treat-all" and
  "treat-none" strategies.
- Saves DCA CSVs and plots under the model training outputs (e.g.
  `model_training/dca/` in stepwise tests or `model_outputs/dca/` in the full
  pipeline).

## 📁 Output Structure

ClinMetML generates a comprehensive set of outputs:

```
clinmetml_outputs/
├── cleaned_data.csv                    # Cleaned dataset
├── feature_selection_outputs/          # Feature selection results
├── multicollinearity_outputs/          # Multicollinearity analysis
├── rfe_outputs/                        # RFE selection results
├── model_outputs/                      # Trained models and metrics
├── interpretability_outputs/           # SHAP analysis and plots
├── pipeline_summary.txt                # Summary report
└── clinmetml_pipeline.log            # Detailed logs
```

## 🔧 Configuration Options

### Data Cleaning Parameters
```python
data_cleaning_params = {
    'imputation_method': 'knn',           # 'knn', 'bpca', 'ppca', 'svd'
    'filter_missing_threshold': 0.2,      # Remove features with >20% missing
    'normalization': 'standard',          # 'standard', 'minmax', 'robust'
    'log_transform': True,                # Apply log2 transformation
    'autonorm': True,                     # Robust normalization using MAD
    'handle_outliers': True,              # Detect and handle outliers
    'target_column': 'disease',           # Target/outcome column
    'id_column': 'id',                    # ID column (excluded from modeling)
    # 'quality_plots_dir': 'data_cleaning_outputs',  # Optional QC plots output
}
```

### Feature Selection Parameters
```python
feature_selection_params = {
    'methods': ['randomforest', 'elasticnet', 'relief', 'lightgbm', 'fcbf', 'mrmr'],
    'top_k': 40,                         # Number of top features to keep
    'id_col': 'id',                      # ID column to exclude
    'balance_threshold': 0.3,            # Class balance threshold for resampling
    'n_iterations': 3,                   # Number of resampling iterations
    'sample_ratio': 0.75,                # Fraction of samples per iteration
    'match_ratio': 3,                    # Case-control matching ratio
    'random_state': 123,                 # Random seed
    'match_cols': ['age', 'sex'],        # Covariates for matching (real data)
}
```


### Multicollinearity Reduction Parameters
```python
multicollinearity_params = {
    'correlation_threshold': 0.9,   # Correlation threshold for redundancy
    'distance_cutoff': 1.2,         # Hierarchical clustering distance cutoff
    'method': 'spearman',           # 'spearman' or 'pearson'
    'clustering_method': 'ward',    # Linkage method for clustering
    'drop_columns': None,           # Extra columns to drop before analysis
    'remove_covariates': False,     # Whether to remove covariate columns
    'covariates': None,             # e.g. ['BMI', 'age', 'sex']
    'visualize': False,             # Whether to show plots
    'save_plots': False,            # Whether to save plots to disk
    # 'output_dir': 'multicollinearity_outputs',  # Optional custom output dir
    # 'vif_threshold': 5.0,         # Accepted for config compatibility (currently unused)
}
```

### RFE Selection Parameters
```python
rfe_params = {
    'estimator': 'rf',              # 'rf', 'logistic', 'svm', 'gb', 'et', 'dt', ...
    'n_features_to_select': 20,     # Number of top features to keep
    'test_size': 0.3,               # Test split proportion
    'random_state': 42,             # Random seed
    'sampling_strategy': 'auto',    # 'auto', 'balanced', 'imbalanced'
    'id_column': 'id',              # ID column to exclude from modeling
    # 'output_dir': 'rfe_outputs',  # Optional custom output dir
}
```
### Model Training Parameters
```python
model_training_params = {
    'models': ['rf', 'xgb', 'lgb'],       # Traditional ML models
    'deep_learning_models': ['ft_transformer', 'gandalf'],  # Optional deep learning models
    'cv_folds': 5,                        # Cross-validation folds
    'test_size': 0.2,                     # Test set proportion
    'balance_classes': True,              # Handle class imbalance
    'sampling_method': 'nearmiss',        # Resampling strategy
    'id_column': 'id',                    # ID column in the dataset
}
```

### Interpretability Parameters
```python
interpretability_params = {
    'data_path': 'interpretability_outputs/interpretability_input.csv',  # Input data
    'model_path': 'model_outputs/metrics/best_model.json',               # Trained model metadata
    'output_dir': 'interpretability_outputs',                            # Where to save plots
    'target_column': 'disease',                                          # Target column in data
    'model_name': None,              # Specific model to analyze; None = best model
    'model_name_for_files': None,    # Optional prefix for output filenames
    'sample_indices': [2, 5, 10],    # Samples for detailed plots
    'max_display_features': 15,      # Max features in SHAP plots
    'interaction_features': None,    # Features for interaction plots
    'apply_sampling': True,          # Whether to apply class balancing
    'sampling_method': 'nearmiss',   # Currently supports 'nearmiss'
    'figure_dpi': 300,               # Plot resolution
    'figure_format': 'png',          # Plot file format
    'save_plots': True,              # Save plots or not
    'show_plots': False,             # Show plots interactively
    'shap_sample_size': 1000,        # Background sample size for SHAP
    'class_to_analyze': 1,           # Focus class for binary classification
}
```


## 🤝 Contributing

We welcome contributions via pull requests and issue reports on GitHub. Feel free to open an issue for bugs, feature requests, or questions, and submit PRs that improve the code, tests, or documentation.

### Development Setup
```bash
git clone https://github.com/Li-OmicsLab-MPU/ClinMetML.git
cd clinmetml
pip install -e .[dev]
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Li-OmicsLab-MPU/ClinMetML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Li-OmicsLab-MPU/ClinMetML/discussions)


## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [SHAP](https://shap.readthedocs.io/)
- Inspired by [MetaboAnalyst](https://www.metaboanalyst.ca/home.xhtml) and [AutoGluon](https://auto.gluon.ai/)

## 📈 Citation

If you use ClinMetML in your research, please cite:

```bibtex
@software{clinmetml2024,
  title={ClinMetML: Automated Metabolomics Biomarker Discovery and Predictive Modeling},
  author={Junrong Li},
  year={2025},
  url={https://github.com/Li-OmicsLab-MPU/ClinMetML}
}
```

---

**ClinMetML** - Making metabolomics analysis accessible to everyone! 🧬✨
