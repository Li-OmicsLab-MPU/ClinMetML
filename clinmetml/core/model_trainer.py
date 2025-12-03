#!/usr/bin/env python3
"""
Tabular Biomarker Model Training Pipeline

This script implements a comprehensive machine learning and deep learning pipeline
for training models on tabular biomarker data after recursive feature elimination.

Features:
- Multiple sampling methods (SMOTE, ADASYN, NearMiss, etc.)
- Deep learning models (FT-Transformer, GANDALF, AutoInt, NODE, TabNet, etc.)
- Traditional ML models (Random Forest, XGBoost, LightGBM, etc.)
- Cross-validation with comprehensive evaluation metrics
- Configurable parameters and model selection

Author: Generated from Jupyter notebook
Date: 2025
"""

import os
import sys

# ClinMetML path management
from ..utils.paths import get_model_training_dir, get_rfe_dir
import argparse
import warnings
import logging

# 设置环境变量防止分布式训练
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU
os.environ['WORLD_SIZE'] = '1'           # 单进程
os.environ['RANK'] = '0'                 # 主进程
os.environ['LOCAL_RANK'] = '0'           # 本地主进程
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import model persistence module
from .model_persistence import ModelPersistenceManager, DataSplitManager
from .dca import compute_dca_curves, plot_dca_curves
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, 
    precision_score, roc_curve, brier_score_loss, confusion_matrix,
    matthews_corrcoef
)

# Sampling methods
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import NearMiss, TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek

# Traditional ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# Deep learning models (pytorch-tabular)
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import (
        FTTransformerConfig, GANDALFConfig, AutoIntConfig, 
        NodeConfig, TabNetModelConfig, TabTransformerConfig, DANetConfig
    )
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    PYTORCH_TABULAR_AVAILABLE = False
    warnings.warn("pytorch-tabular not available. Deep learning models will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
@dataclass
class ModelTrainingConfig:
    """Configuration class for model training parameters."""

    # Data configuration
    data_path: Optional[str] = None
    target_column: Optional[str] = None  # Will be set from user parameters or auto-detected
    id_column: Optional[str] = None      # ID column to be removed before training
    categorical_columns: List[str] = field(default_factory=list)
    continuous_columns: List[str] = field(default_factory=list)

    # Sampling configuration
    sampling_method: str = "smote"  # Options: smote, adasyn, nearmiss, tomek, etc.
    sampling_params: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    n_trials: int = 5
    n_folds: int = 10
    test_size: float = 0.2
    random_state: int = 42

    # Model configuration
    models_to_train: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "lightgbm"]
    )
    deep_learning_models: List[str] = field(
        default_factory=lambda: ["ft_transformer", "gandalf", "tabnet"]
    )

    # Deep learning specific
    batch_size: int = 512        # 减小batch size提高训练稳定性
    max_epochs: int = 50         # 减少epoch数量加快训练
    learning_rate: float = 1e-3

    # Parameter optimization configuration
    enable_grid_search: bool = False
    grid_search_cv: int = 3
    grid_search_scoring: str = 'roc_auc'
    grid_search_n_jobs: int = -1

    # Output configuration
    output_dir: str = field(default_factory=get_model_training_dir)  # Use path manager
    save_predictions: bool = True
    save_metrics: bool = True
    save_models: bool = True  # New: Save trained models

    # Model selection configuration
    best_model_metric: str = "bps"

    # DCA configuration: threshold range for decision curve analysis
    threshold_limits: Tuple[float, float] = (0.05, 1.0)


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data shape: {df.shape}")
        
        # Remove ID column if specified
        if self.config.id_column:
            if self.config.id_column in df.columns:
                logger.info(f"Removing ID column: {self.config.id_column}")
                df = df.drop(columns=[self.config.id_column])
                logger.info(f"Data shape after removing ID column: {df.shape}")
            else:
                logger.warning(f"ID column '{self.config.id_column}' not found in data. Available columns: {list(df.columns)}")
        
        return df
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        if self.config.target_column is None:
            # Auto-detect target column (assume it's the first column or has specific patterns)
            potential_targets = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['target', 'label', 'dep_', 'class', 'outcome'])]
            if potential_targets:
                self.config.target_column = potential_targets[0]
                logger.info(f"Auto-detected target column: {self.config.target_column}")
            else:
                # Default to first column if no pattern matches
                self.config.target_column = df.columns[0]
                logger.warning(f"No target column specified. Using first column: {self.config.target_column}")
        
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data. Available columns: {list(df.columns)}")
        
        X = df.drop(self.config.target_column, axis=1)
        y = df[self.config.target_column]
        return X, y
    
    def auto_detect_column_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Automatically detect categorical and continuous columns."""
        categorical_cols = []
        continuous_cols = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category'] or X[col].nunique() < 10:
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
        
        logger.info(f"Detected {len(categorical_cols)} categorical and {len(continuous_cols)} continuous columns")
        return categorical_cols, continuous_cols
    
    def analyze_class_distribution(self, y: pd.Series) -> str:
        """
        Analyze class distribution and recommend sampling strategy.
        
        Args:
            y: Target variable
            
        Returns:
            Recommended sampling method
        """
        # Calculate class distribution
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        
        logger.info("=" * 60)
        logger.info("CLASS DISTRIBUTION ANALYSIS")
        logger.info("=" * 60)
        
        # Display class distribution
        for class_label, count in class_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"Class {class_label}: {count:,} samples ({percentage:.2f}%)")
        
        # Calculate imbalance ratio (majority class / minority class)
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        logger.info(f"Total samples: {total_samples:,}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Determine sampling strategy based on imbalance ratio and sample sizes
        if imbalance_ratio <= 1.5:
            # Balanced or slightly imbalanced
            recommended_method = "none"
            reason = "Dataset is relatively balanced"
        elif imbalance_ratio <= 3.0:
            # Moderately imbalanced
            if min_count < 100:
                recommended_method = "smote"
                reason = "Moderate imbalance with small minority class - use oversampling"
            else:
                recommended_method = "nearmiss"
                reason = "Moderate imbalance with sufficient samples - use undersampling"
        elif imbalance_ratio <= 10.0:
            # Highly imbalanced
            if min_count < 50:
                recommended_method = "smote"
                reason = "High imbalance with very small minority class - use oversampling"
            elif min_count < 200:
                recommended_method = "smote_nearmiss"
                reason = "High imbalance - use combined oversampling and undersampling"
            else:
                recommended_method = "nearmiss"
                reason = "High imbalance with sufficient samples - use undersampling"
        else:
            # Extremely imbalanced
            if min_count < 30:
                recommended_method = "adasyn"
                reason = "Extreme imbalance with very small minority class - use adaptive oversampling"
            elif min_count < 100:
                recommended_method = "adasyn_nearmiss"
                reason = "Extreme imbalance - use adaptive combined sampling"
            else:
                recommended_method = "smote_nearmiss"
                reason = "Extreme imbalance - use combined oversampling and undersampling"
        
        logger.info("-" * 60)
        logger.info(f"📊 RECOMMENDATION: {recommended_method.upper()}")
        logger.info(f"💡 Reason: {reason}")
        logger.info("=" * 60)
        
        return recommended_method


class SamplingManager:
    """Manages different sampling strategies for imbalanced datasets."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        
    def apply_sampling(self, X: pd.DataFrame, y: pd.Series, method: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply the specified sampling method."""
        if method is None:
            method = self.config.sampling_method
            
        logger.info(f"Applying {method} sampling")
        
        if method == "smote":
            sampler = SMOTE(random_state=self.config.random_state)
        elif method == "adasyn":
            sampler = ADASYN(random_state=self.config.random_state)
        elif method == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=self.config.random_state)
        elif method == "nearmiss":
            sampler = NearMiss(version=1)
        elif method == "tomek":
            sampler = TomekLinks()
        elif method == "random_undersample":
            sampler = RandomUnderSampler(random_state=self.config.random_state)
        elif method == "smote_tomek":
            sampler = SMOTETomek(random_state=self.config.random_state)
        elif method == "adasyn_nearmiss":
            return self._apply_combined_sampling(X, y, "adasyn", "nearmiss")
        elif method == "smote_nearmiss":
            return self._apply_combined_sampling(X, y, "smote", "nearmiss")
        elif method == "adasyn_tomek":
            return self._apply_combined_sampling(X, y, "adasyn", "tomek")
        elif method == "none":
            return X, y
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logger.info(f"Original class distribution: {dict(y.value_counts())}")
        logger.info(f"Resampled class distribution: {dict(pd.Series(y_resampled).value_counts())}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def _apply_combined_sampling(self, X: pd.DataFrame, y: pd.Series, 
                               oversample_method: str, undersample_method: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply combined over-sampling and under-sampling methods."""
        logger.info(f"Applying combined sampling: {oversample_method} + {undersample_method}")
        
        # 第一步：过采样
        if oversample_method == "smote":
            oversampler = SMOTE(random_state=self.config.random_state)
        elif oversample_method == "adasyn":
            oversampler = ADASYN(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown oversampling method: {oversample_method}")
        
        X_oversampled, y_oversampled = oversampler.fit_resample(X, y)
        logger.info(f"After {oversample_method}: {dict(pd.Series(y_oversampled).value_counts())}")
        
        # 第二步：欠采样
        if undersample_method == "nearmiss":
            undersampler = NearMiss(version=1)
        elif undersample_method == "tomek":
            undersampler = TomekLinks()
        else:
            raise ValueError(f"Unknown undersampling method: {undersample_method}")
        
        X_final, y_final = undersampler.fit_resample(X_oversampled, y_oversampled)
        
        logger.info(f"Original class distribution: {dict(y.value_counts())}")
        logger.info(f"Final class distribution: {dict(pd.Series(y_final).value_counts())}")
        
        return pd.DataFrame(X_final, columns=X.columns), pd.Series(y_final)


class ParameterOptimizer:
    """Handles hyperparameter optimization using grid search."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        
    def get_parameter_grids(self) -> Dict[str, Dict]:
        """Define parameter grids for different models."""
        return {
            'random_forest': {
                'n_estimators': [10, 50, 100, 200, 400],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['log2', 'sqrt']
            },
            'decision_tree': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_leaf': range(1, 50, 5),
                'max_depth': [None, 4, 8, 12, 16],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': range(20, 81, 10),
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [200, 300, 500],
                'min_samples_leaf': [20, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 6],
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'max_depth': [4, 6, 8],
                'num_leaves': [20, 30, 40],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
    
    def optimize_parameters(self, model, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series) -> Dict[str, Any]:
        """Optimize parameters for a given model using grid search."""
        if not self.config.enable_grid_search:
            return {}
        
        param_grids = self.get_parameter_grids()
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}, skipping optimization")
            return {}
        
        logger.info(f"Starting grid search for {model_name}...")
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=self.config.grid_search_cv,
            scoring=self.config.grid_search_scoring,
            n_jobs=self.config.grid_search_n_jobs,
            verbose=1
        )
        
        try:
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best {self.config.grid_search_scoring} score: {grid_search.best_score_:.4f}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_
            }
            
        except Exception as e:
            logger.error(f"Grid search failed for {model_name}: {str(e)}")
            return {}
    
    def get_optimized_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series):
        """Get an optimized model with best parameters."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        # Create base model
        if model_name == "random_forest":
            base_model = RandomForestClassifier(random_state=self.config.random_state)
        elif model_name == "decision_tree":
            base_model = DecisionTreeClassifier(random_state=self.config.random_state)
        elif model_name == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=self.config.random_state)
        elif model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
                base_model = XGBClassifier(
                    random_state=self.config.random_state,
                    n_jobs=1,
                    verbosity=0
                )
            except ImportError:
                logger.warning("XGBoost not available")
                return None
        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb
                base_model = lgb.LGBMClassifier(
                    random_state=self.config.random_state,
                    verbosity=-1,
                    n_jobs=1
                )
            except ImportError:
                logger.warning("LightGBM not available")
                return None
        else:
            logger.warning(f"Unknown model: {model_name}")
            return None
        
        # Optimize parameters
        optimization_result = self.optimize_parameters(base_model, model_name, X_train, y_train)
        
        if optimization_result and 'best_estimator' in optimization_result:
            return optimization_result['best_estimator']
        else:
            return base_model


class MetricsCalculator:
    """Calculates comprehensive evaluation metrics."""
    
    @staticmethod
    def confusion_matrix_components(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix components."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            metrics['log_loss'] = MetricsCalculator._log_loss(y_true, y_pred_proba)
            
            # Calculate BPS 2.0 and related metrics
            bps_metrics = MetricsCalculator.calculate_refined_bps(y_true, y_pred_proba)
            metrics.update(bps_metrics)
        
        # Confusion matrix derived metrics
        tp, tn, fp, fn = MetricsCalculator.confusion_matrix_components(y_true, y_pred)
        
        # Matthews Correlation Coefficient
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = numerator / denominator if denominator != 0 else 0
        
        # PPV (Positive Predictive Value)
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # NPV (Negative Predictive Value)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Sensitivity (True Positive Rate)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def _log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, eps: float = 1e-15) -> float:
        """Calculate log loss."""
        # 确保概率值在有效范围内
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
        
        # 处理可能的NaN或无穷值
        if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
            return float('nan')
        
        try:
            loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
            return loss if not np.isnan(loss) else float('nan')
        except:
            return float('nan')
    
    @staticmethod
    def calculate_refined_bps(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        计算精炼平衡性能分数 (Refined Balanced Performance Score, BPS 2.0)。

        该指标以一种稳健的方式结合了模型的区分能力 (Discrimination) 和校准能力 (Calibration)。
        它遵循了BPS 2.0的设计原则，包括使用精简的指标集、相对分数（缩放分数）以及结构化聚合。

        Args:
            y_true (np.ndarray): 真实的二进制标签 (0 或 1)。
            y_pred_proba (np.ndarray): 模型预测的阳性类别（类别1）的概率。
            threshold (float): 用于计算马修斯相关系数(MCC)的分类阈值，默认为0.5。

        Returns:
            Dict[str, float]: 一个包含 BPS 2.0 及其所有组成部分的字典，方便进行详细分析。
        """
        # 确保输入为numpy数组，以进行向量化计算
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        # 对预测概率进行数值稳定性处理，避免log(0)的情况
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

        try:
            # --- 1. 计算区分能力分数 (Discrimination Score, DS) ---
            
            # 1.1 AUC-ROC: 衡量模型的整体排序能力，阈值无关。
            auc = roc_auc_score(y_true, y_pred_proba)

            # 1.2 马修斯相关系数 (MCC): 最均衡的阈值依赖指标之一。
            # 首先根据阈值将概率转换为二进制预测
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_binary)
            
            # 计算F1分数和Recall
            f1 = f1_score(y_true, y_pred_binary)
            recall = recall_score(y_true, y_pred_binary)
            
            # 将MCC从[-1, 1]的范围归一化到[0, 1]，以便进行几何平均
            mcc_norm = (mcc + 1) / 2

            # 1.3 计算区分能力分数 (DS): 使用AUC和归一化MCC的几何平均值
            ds = np.sqrt(auc * mcc_norm)

            # --- 2. 计算校准能力分数 (Calibration Score, CS) ---
            
            # 计算数据的患病率（阳性样本的比例），这是计算基线模型的关键
            prevalence = np.mean(y_true)

            # 处理患病率为0或1的极端情况，此时基线分数为0，会导致除零错误。
            # 在这种情况下，校准问题是平凡的，我们可以认为校准是完美的（得分为1）。
            if prevalence == 0 or prevalence == 1:
                sbs = 1.0
                sll = 1.0
                bs = 0.0  # Brier Score 在这种情况下为0
                ll = 0.0  # Log Loss 在这种情况下为0
            else:
                # 2.1 缩放布里尔分数 (Scaled Brier Score, SBS)
                # 计算模型的Brier分数
                bs = brier_score_loss(y_true, y_pred_proba)
                # 计算基线模型（只会预测平均患病率）的Brier分数
                bs_ref = prevalence * (1 - prevalence)
                # 计算SBS，表示模型相对于基线模型的改进程度
                sbs = 1 - (bs / bs_ref)

                # 2.2 缩放对数损失 (Scaled Log-Loss, SLL)
                # 手动计算Log-Loss以确保数值稳定性
                ll = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
                # 计算基线模型的Log-Loss
                ll_ref = - (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence))
                # 计算SLL，表示模型相对于基线模型的改进程度
                sll = 1 - (ll / ll_ref)

            # 2.3 计算校准能力分数 (CS): 使用SBS和SLL的几何平均值
            # 使用max(0, ...)来确保比基线模型更差的校准能力不会获得分数（避免负数开方）
            cs = np.sqrt(max(0, sbs) * max(0, sll))

            # --- 3. 计算最终的 BPS 2.0 ---
            # BPS 2.0 是区分能力分数(DS)和校准能力分数(CS)的几何平均值
            bps = np.sqrt(ds * cs)

            # 返回一个包含所有计算结果的字典，便于透明化和报告
            return {
                'bps': bps,
                'discrimination_score': ds,
                'calibration_score': cs,
                'auc': auc,
                'mcc': mcc,
                'mcc_norm': mcc_norm,
                'f1_score': f1,
                'recall': recall,
                'scaled_brier_score': sbs,
                'scaled_log_loss': sll,
                'brier_score': bs,
                'log_loss': ll
            }
        except Exception as e:
            logger.warning(f"Error calculating BPS 2.0: {e}")
            # 返回NaN值作为错误情况的默认值
            return {
                'bps': float('nan'),
                'discrimination_score': float('nan'),
                'calibration_score': float('nan'),
                'auc': float('nan'),
                'mcc': float('nan'),
                'mcc_norm': float('nan'),
                'f1_score': float('nan'),
                'recall': float('nan'),
                'scaled_brier_score': float('nan'),
                'scaled_log_loss': float('nan'),
                'brier_score': float('nan'),
                'log_loss': float('nan')
            }


class TraditionalMLTrainer:
    """Trainer for traditional machine learning models."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.models = self._initialize_models()
        self.parameter_optimizer = ParameterOptimizer(config)
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize traditional ML models."""
        models = {}
        
        if "random_forest" in self.config.models_to_train:
            models["random_forest"] = RandomForestClassifier(random_state=self.config.random_state)
        
        if "decision_tree" in self.config.models_to_train:
            models["decision_tree"] = DecisionTreeClassifier(
                criterion='gini', min_samples_leaf=41, splitter='best', 
                random_state=self.config.random_state
            )
        
        if "gradient_boosting" in self.config.models_to_train:
            models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=60, max_depth=7, min_samples_split=100,
                random_state=self.config.random_state
            )
        
        if "xgboost" in self.config.models_to_train:
            models["xgboost"] = XGBClassifier(
                max_depth=6, 
                min_child_weight=1, 
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=1,  # 限制并行度避免资源竞争
                verbosity=0  # 减少输出
            )
        
        if "lightgbm" in self.config.models_to_train:
            models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                verbosity=-1, 
                random_state=self.config.random_state,
                n_jobs=1  # 限制并行度
            )
        
        return models
    
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series, 
                          model_name: str) -> Dict[str, float]:
        """Train and evaluate a single model."""
        # Use optimized model if grid search is enabled
        if self.config.enable_grid_search:
            logger.info(f"    Using parameter optimization for {model_name}...")
            model = self.parameter_optimizer.get_optimized_model(model_name, X_train, y_train)
            if model is None:
                logger.warning(f"    Failed to optimize {model_name}, using default parameters")
                model = self.models[model_name]
        else:
            model = self.models[model_name]
        
        # Train model with progress logging
        logger.info(f"    Training {model_name} on {len(X_train)} samples...")
        try:
            if not self.config.enable_grid_search:  # Only fit if not already fitted during grid search
                model.fit(X_train, y_train)
            logger.info(f"    {model_name} training completed successfully")
        except Exception as e:
            logger.error(f"    {model_name} training failed: {str(e)}")
            raise
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return metrics, y_pred, y_pred_proba


class DeepLearningTrainer:
    """Trainer for deep learning models using pytorch-tabular."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model_configs = self._initialize_model_configs()
    
    def _initialize_model_configs(self) -> Dict[str, Any]:
        """Initialize deep learning model configurations."""
        if not PYTORCH_TABULAR_AVAILABLE:
            return {}
        
        configs = {}
        
        if "ft_transformer" in self.config.deep_learning_models:
            configs["ft_transformer"] = FTTransformerConfig(task="classification")
        
        if "gandalf" in self.config.deep_learning_models:
            configs["gandalf"] = GANDALFConfig(
                task="classification",
                gflu_stages=6,
                gflu_feature_init_sparsity=0.3,
                gflu_dropout=0.0,
                learning_rate=self.config.learning_rate
            )
        
        if "autoint" in self.config.deep_learning_models:
            configs["autoint"] = AutoIntConfig(task="classification")
        
        if "node" in self.config.deep_learning_models:
            configs["node"] = NodeConfig(
                task="classification",
                depth=10,
                num_trees=50,
                learning_rate=self.config.learning_rate
            )
        
        if "tabnet" in self.config.deep_learning_models:
            configs["tabnet"] = TabNetModelConfig(
                task="classification",
                learning_rate=self.config.learning_rate
            )
        
        if "tab_transformer" in self.config.deep_learning_models:
            configs["tab_transformer"] = TabTransformerConfig(
                task="classification",
                input_embed_dim=8,
                num_attn_blocks=1,
                num_heads=2,
                learning_rate=self.config.learning_rate
            )
        
        if "danet" in self.config.deep_learning_models:
            configs["danet"] = DANetConfig(
                task="classification",
                n_layers=1,
                virtual_batch_size=16,
                learning_rate=self.config.learning_rate
            )
        
        return configs
    
    def create_tabular_model(self, model_name: str, categorical_cols: List[str], 
                           continuous_cols: List[str]):
        """Create a TabularModel instance."""
        if not PYTORCH_TABULAR_AVAILABLE:
            raise RuntimeError("pytorch-tabular is not available")
        
        data_config = DataConfig(
            target=[self.config.target_column],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )
        
        trainer_config = TrainerConfig(
            batch_size=self.config.batch_size,
            max_epochs=self.config.max_epochs,
            accelerator='cpu',    # 强制使用CPU避免GPU分布式训练问题
            devices=1,           # 使用单设备
            progress_bar='none', # 完全禁用进度条
            checkpoints=None,    # 禁用检查点
            early_stopping=None, # 禁用早停
            trainer_kwargs=dict(
                enable_model_summary=False,
                enable_progress_bar=False,
                enable_checkpointing=False
            )
        )
        
        optimizer_config = OptimizerConfig()
        model_config = self.model_configs[model_name]
        
        return TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False
        )


class CrossValidationPipeline:
    """Main cross-validation training pipeline."""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.sampling_manager = SamplingManager(config)
        self.ml_trainer = TraditionalMLTrainer(config)
        self.dl_trainer = DeepLearningTrainer(config) if PYTORCH_TABULAR_AVAILABLE else None
        
        # Initialize model persistence manager
        self.model_persistence = ModelPersistenceManager(config.output_dir) if config.save_models else None
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{config.output_dir}/predictions").mkdir(parents=True, exist_ok=True)
        Path(f"{config.output_dir}/metrics").mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting model training pipeline")
        
        # Load and prepare data
        df = self.data_processor.load_data(data_path)
        X, y = self.data_processor.prepare_features_target(df)
        
        # Auto-detect column types if not specified
        if not self.config.categorical_columns and not self.config.continuous_columns:
            cat_cols, cont_cols = self.data_processor.auto_detect_column_types(X)
            self.config.categorical_columns = cat_cols
            self.config.continuous_columns = cont_cols
        
        # Auto-detect sampling method if set to 'auto'
        if self.config.sampling_method == "auto":
            recommended_method = self.data_processor.analyze_class_distribution(y)
            logger.info(f"Auto-detected sampling method: {recommended_method}")
            # Update the sampling method in config
            self.config.sampling_method = recommended_method
        
        # Apply sampling
        X_resampled, y_resampled = self.sampling_manager.apply_sampling(X, y)
        
        # Combine resampled data
        data_resampled = pd.concat([y_resampled, X_resampled], axis=1)
        
        # Run cross-validation for each model
        all_results = {}
        
        # Traditional ML models
        for model_name in self.config.models_to_train:
            if model_name in self.ml_trainer.models:
                logger.info(f"Training {model_name}")
                results = self._run_cv_for_ml_model(data_resampled, model_name)
                all_results[model_name] = results
        
        # Deep learning models
        if self.dl_trainer and PYTORCH_TABULAR_AVAILABLE:
            for model_name in self.config.deep_learning_models:
                if model_name in self.dl_trainer.model_configs:
                    logger.info(f"Training {model_name}")
                    results = self._run_cv_for_dl_model(data_resampled, model_name)
                    all_results[model_name] = results
        
        # Save results
        self._save_results(all_results)
        
        # Train and save final models if enabled
        if not getattr(self.config, 'skip_final_training', False):
            if self.config.save_models and self.model_persistence:
                logger.info("Training and saving final models...")
                self._train_and_save_final_models(X, y, X_resampled, y_resampled, all_results)
        else:
            logger.info("⚡ Skipping final model training as requested")
        
        logger.info("Pipeline completed successfully")
        return all_results
    
    def _run_cv_for_ml_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Run cross-validation for a traditional ML model."""
        all_metrics = {
            'train': {metric: [] for metric in ['accuracy', 'roc_auc', 'f1_score', 'recall', 
                                              'precision', 'mcc', 'ppv', 'npv', 'sensitivity', 
                                              'specificity', 'brier_score', 'log_loss', 'bps',
                                              'discrimination_score', 'calibration_score', 'scaled_brier_score',
                                              'scaled_log_loss', 'mcc_norm']},
            'test': {metric: [] for metric in ['accuracy', 'roc_auc', 'f1_score', 'recall', 
                                             'precision', 'mcc', 'ppv', 'npv', 'sensitivity', 
                                             'specificity', 'brier_score', 'log_loss', 'bps',
                                             'discrimination_score', 'calibration_score', 'scaled_brier_score',
                                             'scaled_log_loss', 'mcc_norm']}
        }
        
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
        
        for trial in range(self.config.n_trials):
            logger.info(f"  Trial {trial + 1}/{self.config.n_trials}")
            
            # Split data
            train_data, test_data = train_test_split(
                data, test_size=self.config.test_size, 
                random_state=self.config.random_state + trial
            )
            
            trial_train_metrics = {metric: [] for metric in all_metrics['train'].keys()}
            trial_test_metrics = {metric: [] for metric in all_metrics['test'].keys()}
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
                # Prepare fold data
                train_fold = train_data.iloc[train_idx]
                val_fold = train_data.iloc[val_idx]
                
                X_train = train_fold.drop(self.config.target_column, axis=1)
                y_train = train_fold[self.config.target_column]
                X_val = val_fold.drop(self.config.target_column, axis=1)
                y_val = val_fold[self.config.target_column]
                
                # Train and evaluate on validation set
                val_metrics, y_pred_val, y_pred_proba_val = self.ml_trainer.train_and_evaluate(
                    X_train, y_train, X_val, y_val, model_name
                )
                
                # Store validation metrics
                for metric, value in val_metrics.items():
                    if metric in trial_train_metrics:
                        trial_train_metrics[metric].append(value)
                
                # Evaluate on test set
                X_test = test_data.drop(self.config.target_column, axis=1)
                y_test = test_data[self.config.target_column]
                
                test_metrics, y_pred_test, y_pred_proba_test = self.ml_trainer.train_and_evaluate(
                    X_train, y_train, X_test, y_test, model_name
                )
                
                # Store test metrics
                for metric, value in test_metrics.items():
                    if metric in trial_test_metrics:
                        trial_test_metrics[metric].append(value)
                
                # Save predictions if requested
                if self.config.save_predictions:
                    self._save_predictions(
                        y_val, y_pred_val, y_pred_proba_val, 
                        f"{model_name}_trial{trial}_fold{fold}_train"
                    )
                    self._save_predictions(
                        y_test, y_pred_test, y_pred_proba_test, 
                        f"{model_name}_trial{trial}_fold{fold}_test"
                    )
            
            # Aggregate trial metrics
            for metric in all_metrics['train'].keys():
                if trial_train_metrics[metric]:
                    all_metrics['train'][metric].extend(trial_train_metrics[metric])
                if trial_test_metrics[metric]:
                    all_metrics['test'][metric].extend(trial_test_metrics[metric])
        
        return all_metrics
    
    def _run_cv_for_dl_model(self, data: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Run cross-validation for a deep learning model."""
        if not PYTORCH_TABULAR_AVAILABLE:
            logger.warning(f"Skipping {model_name} - pytorch-tabular not available")
            return {}
        
        all_metrics = {
            'train': {metric: [] for metric in ['accuracy', 'roc_auc', 'f1_score', 'recall', 
                                              'precision', 'mcc', 'ppv', 'npv', 'sensitivity', 
                                              'specificity', 'brier_score', 'log_loss', 'bps',
                                              'discrimination_score', 'calibration_score', 'scaled_brier_score',
                                              'scaled_log_loss', 'mcc_norm']},
            'test': {metric: [] for metric in ['accuracy', 'roc_auc', 'f1_score', 'recall', 
                                             'precision', 'mcc', 'ppv', 'npv', 'sensitivity', 
                                             'specificity', 'brier_score', 'log_loss', 'bps',
                                             'discrimination_score', 'calibration_score', 'scaled_brier_score',
                                             'scaled_log_loss', 'mcc_norm']}
        }
        
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
        
        for trial in range(self.config.n_trials):
            logger.info(f"  Trial {trial + 1}/{self.config.n_trials}")
            
            # Split data
            train_data, test_data = train_test_split(
                data, test_size=self.config.test_size, 
                random_state=self.config.random_state + trial
            )
            
            trial_train_metrics = {metric: [] for metric in all_metrics['train'].keys()}
            trial_test_metrics = {metric: [] for metric in all_metrics['test'].keys()}
            
            # Create tabular model for this trial
            tabular_model = self.dl_trainer.create_tabular_model(
                model_name, self.config.categorical_columns, self.config.continuous_columns
            )
            
            datamodule = None
            model = None
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
                logger.info(f"    Training {model_name} fold {fold + 1}/{self.config.n_folds}...")
                
                # Prepare fold data
                train_fold = train_data.iloc[train_idx]
                val_fold = train_data.iloc[val_idx]
                
                try:
                    if datamodule is None:
                        # Initialize datamodule and model in the first fold
                        datamodule = tabular_model.prepare_dataloader(
                            train=train_fold, validation=val_fold, seed=42
                        )
                        model = tabular_model.prepare_model(datamodule)
                    else:
                        # Creates a copy of the datamodule with same transformers but different data
                        datamodule = datamodule.copy(train=train_fold, validation=val_fold)
                    
                    # Train the model
                    tabular_model.train(model, datamodule)
                    
                    # Predict on validation set
                    pred_df = tabular_model.predict(val_fold)
                    y_val_true = val_fold[self.config.target_column]
                    
                    # Calculate validation metrics
                    val_metrics = self._calculate_dl_metrics(y_val_true, pred_df)
                    for metric, value in val_metrics.items():
                        if metric in trial_train_metrics:
                            trial_train_metrics[metric].append(value)
                    
                    # Predict on test set
                    test_pred_df = tabular_model.predict(test_data)
                    y_test_true = test_data[self.config.target_column]
                    
                    # Calculate test metrics
                    test_metrics = self._calculate_dl_metrics(y_test_true, test_pred_df)
                    for metric, value in test_metrics.items():
                        if metric in trial_test_metrics:
                            trial_test_metrics[metric].append(value)
                    
                    # Save predictions if requested
                    if self.config.save_predictions:
                        self._save_dl_predictions(
                            y_val_true, pred_df, 
                            f"{model_name}_trial{trial}_fold{fold}_train"
                        )
                        self._save_dl_predictions(
                            y_test_true, test_pred_df, 
                            f"{model_name}_trial{trial}_fold{fold}_test"
                        )
                    
                    # Reset model weights for next fold
                    tabular_model.model.reset_weights()
                    
                    logger.info(f"    {model_name} fold {fold + 1} completed successfully")
                    
                except Exception as e:
                    logger.error(f"    {model_name} fold {fold + 1} failed: {str(e)}")
                    continue
            
            # Aggregate trial metrics
            for metric in all_metrics['train'].keys():
                if trial_train_metrics[metric]:
                    all_metrics['train'][metric].extend(trial_train_metrics[metric])
                if trial_test_metrics[metric]:
                    all_metrics['test'][metric].extend(trial_test_metrics[metric])
        
        return all_metrics
    
    def _calculate_dl_metrics(self, y_true: pd.Series, pred_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for deep learning model predictions."""
        target_col = self.config.target_column
        pred_col = f"{target_col}_prediction"
        prob_col = f"{target_col}_1_probability"
        
        y_pred = pred_df[pred_col].values
        y_pred_proba = pred_df[prob_col].values if prob_col in pred_df.columns else None
        
        return MetricsCalculator.calculate_metrics(y_true.values, y_pred, y_pred_proba)
    
    def _save_dl_predictions(self, y_true: pd.Series, pred_df: pd.DataFrame, filename: str):
        """Save deep learning model predictions to file."""
        if not self.config.save_predictions:
            return
        
        target_col = self.config.target_column
        pred_col = f"{target_col}_prediction"
        prob_col = f"{target_col}_1_probability"
        
        save_df = pd.DataFrame({
            'y_true': y_true.values,
            'y_pred': pred_df[pred_col].values,
            'y_pred_proba': pred_df[prob_col].values if prob_col in pred_df.columns else np.nan
        })
        
        save_df.to_csv(f"{self.config.output_dir}/predictions/{filename}.csv", index=False)
    
    def _save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray, filename: str):
        """Save predictions to file."""
        if not self.config.save_predictions:
            return
        
        pred_df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba if y_pred_proba is not None else np.nan
        })
        
        pred_df.to_csv(f"{self.config.output_dir}/predictions/{filename}.csv", index=False)
    
    def _generate_model_summary(self, summary_results: Dict[str, Any]):
        """Generate a comprehensive model summary table."""
        if not summary_results:
            logger.warning("No results to summarize")
            return
        
        # Define key metrics for summary
        key_metrics = ['accuracy', 'roc_auc', 'f1_score', 'recall', 'precision', 'mcc', 
                      'ppv', 'npv', 'sensitivity', 'specificity',
                      'bps', 'discrimination_score', 'calibration_score', 
                      'brier_score', 'log_loss']
        
        # Create summary table
        summary_data = []
        for model_name, model_results in summary_results.items():
            if 'test' in model_results:
                row = {'Model': model_name}
                test_results = model_results['test']
                
                for metric in key_metrics:
                    if metric in test_results:
                        mean_val = test_results[metric]['mean']
                        std_val = test_results[metric]['std']
                        row[f'{metric}_mean'] = mean_val
                        row[f'{metric}_std'] = std_val
                        row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        row[f'{metric}_mean'] = np.nan
                        row[f'{metric}_std'] = np.nan
                        row[metric] = "N/A"
                
                summary_data.append(row)
        
        if not summary_data:
            logger.warning("No test results found for summary")
            return
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        summary_df.to_csv(f"{self.config.output_dir}/metrics/model_summary.csv", index=False)
        
        # Display summary table
        logger.info("=" * 80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        
        # Display key metrics in a formatted table
        display_metrics = ['accuracy', 'roc_auc', 'f1_score', 'mcc', 'ppv', 'npv', 'sensitivity', 'specificity', 'bps']
        
        # Print header
        header = f"{'Model':<15}"
        for metric in display_metrics:
            header += f"{metric.upper():<12}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print model results
        for _, row in summary_df.iterrows():
            line = f"{row['Model']:<15}"
            for metric in display_metrics:
                if f'{metric}_mean' in row and not pd.isna(row[f'{metric}_mean']):
                    line += f"{row[f'{metric}_mean']:<12.4f}"
                else:
                    line += f"{'N/A':<12}"
            logger.info(line)
        
        logger.info("=" * 80)
        logger.info(f"Full summary saved to: {self.config.output_dir}/metrics/model_summary.csv")
    
    def select_best_model_by_metric(self, primary_metric: str = 'bps', 
                                   secondary_metrics: List[str] = None,
                                   summary_results: Dict[str, Any] = None,
                                   results_path: str = None) -> Tuple[str, Dict]:
        """
        Select the best model based on specified metrics.
        
        Args:
            primary_metric: Primary metric for model selection (default: 'bps')
            secondary_metrics: List of secondary metrics for tie-breaking
            summary_results: Pre-loaded summary results dict (if None, will load from file)
            results_path: Path to results JSON file (if None, uses default location)
        
        Returns:
            Tuple of (best_model_name, best_model_scores)
        """
        # Load results if not provided
        if summary_results is None:
            if results_path is None:
                results_path = f"{self.config.output_dir}/metrics/summary_results.json"
            
            try:
                with open(results_path, 'r') as f:
                    summary_results = json.load(f)
            except FileNotFoundError:
                logger.error(f"Results file not found: {results_path}")
                return None, None
            except Exception as e:
                logger.error(f"Error loading results: {e}")
                return None, None
        
        # Validate input
        if not summary_results:
            logger.warning("No results available for model selection")
            return None, None
        
        if secondary_metrics is None:
            secondary_metrics = ['roc_auc', 'f1_score', 'mcc']
        
        logger.info(f"Selecting best model using {primary_metric} as primary metric")
        
        # Extract test results for comparison
        model_scores = {}
        for model_name, model_results in summary_results.items():
            if 'test' in model_results and primary_metric in model_results['test']:
                test_results = model_results['test']
                primary_score = test_results[primary_metric]['mean']
                
                # Calculate composite score including secondary metrics
                secondary_scores = []
                for metric in secondary_metrics:
                    if metric in test_results:
                        secondary_scores.append(test_results[metric]['mean'])
                
                # Composite score: 70% primary + 30% average of secondary metrics
                if secondary_scores:
                    composite_score = 0.7 * primary_score + 0.3 * np.mean(secondary_scores)
                else:
                    composite_score = primary_score
                
                model_scores[model_name] = {
                    'primary_score': primary_score,
                    'composite_score': composite_score,
                    'secondary_scores': secondary_scores
                }
        
        if not model_scores:
            logger.warning(f"No models found with {primary_metric} metric")
            return None, None
        
        # Find best model
        best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['composite_score'])
        best_scores = model_scores[best_model]
        
        # Log best model selection
        logger.info("=" * 80)
        logger.info("BEST MODEL SELECTION")
        logger.info("=" * 80)
        logger.info(f"Primary Metric: {primary_metric.upper()}")
        logger.info(f"Secondary Metrics: {', '.join([m.upper() for m in secondary_metrics])}")
        logger.info("")
        logger.info(f"🏆 BEST MODEL: {best_model.upper()}")
        logger.info(f"   {primary_metric.upper()}: {best_scores['primary_score']:.4f}")
        logger.info(f"   Composite Score: {best_scores['composite_score']:.4f}")
        
        if best_scores['secondary_scores']:
            logger.info("   Secondary Metrics:")
            for i, metric in enumerate(secondary_metrics):
                if i < len(best_scores['secondary_scores']):
                    logger.info(f"     {metric.upper()}: {best_scores['secondary_scores'][i]:.4f}")
        
        # Show ranking of all models
        logger.info("")
        logger.info("📊 MODEL RANKING:")
        sorted_models = sorted(model_scores.items(), 
                             key=lambda x: x[1]['composite_score'], 
                             reverse=True)
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            logger.info(f"   {i}. {model_name.upper():<15} "
                       f"({primary_metric.upper()}: {scores['primary_score']:.4f}, "
                       f"Composite: {scores['composite_score']:.4f})")
        
        # Save best model info
        best_model_info = {
            'best_model': best_model,
            'selection_criteria': {
                'primary_metric': primary_metric,
                'secondary_metrics': secondary_metrics
            },
            'scores': best_scores,
            'ranking': [(model, scores) for model, scores in sorted_models]
        }
        
        with open(f"{self.config.output_dir}/metrics/best_model.json", 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        logger.info("=" * 80)
        logger.info(f"Best model info saved to: {self.config.output_dir}/metrics/best_model.json")
        
        return best_model, best_scores

    def _train_and_save_final_models(self, X_original: pd.DataFrame, y_original: pd.Series,
                                   X_resampled: pd.DataFrame, y_resampled: pd.Series,
                                   cv_results: Dict[str, Any]):
        """Train final models on full data and save them with context."""
        logger.info("Training final models on full resampled data...")
        
        # Create reproducible train-test split
        X_train, X_test, y_train, y_test, split_info = DataSplitManager.create_reproducible_split(
            X_resampled, y_resampled, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=True
        )
        
        # Prepare data context
        data_context = self._prepare_data_context(X_original, y_original, X_resampled, y_resampled, split_info)
        
        # Dictionary to store trained models
        trained_models = {}
        
        # 1) Train and save traditional ML models
        for model_name in self.config.models_to_train:
            if model_name in self.ml_trainer.models and model_name in cv_results:
                logger.info(f"  Training final {model_name} model...")

                model = self._create_and_train_model(model_name, X_train, y_train)
                trained_models[model_name] = model

                # Save the model with its context
                self._save_individual_model(model, model_name, cv_results, data_context, split_info)

        # 1b) Train and save deep learning models
        if self.dl_trainer and PYTORCH_TABULAR_AVAILABLE:
            for model_name in self.config.deep_learning_models:
                if model_name in cv_results:
                    logger.info(f"  Training final {model_name} deep learning model...")
                    dl_model = self._create_and_train_dl_model(model_name, X_train, y_train, data_context)
                    if dl_model is not None:
                        trained_models[model_name] = dl_model
                        self._save_individual_dl_model(dl_model, model_name, cv_results, data_context, split_info)

        # 2) Select the best model for DCA computation (only consider traditional ML models)
        if not trained_models:
            logger.warning("No trained models available for DCA computation.")
            return

        best_model_name = None
        best_score = -np.inf

        for model_name, results in cv_results.items():
            # Only consider traditional ML models for DCA computation
            if model_name not in trained_models or model_name not in self.ml_trainer.models:
                continue
            try:
                test_metrics = results.get("test", {})
                bps_list = test_metrics.get("bps", [])
                if not bps_list:
                    continue
                mean_bps = float(np.mean(bps_list))
                if mean_bps > best_score:
                    best_score = mean_bps
                    best_model_name = model_name
            except Exception as e:
                logger.warning(f"Failed to evaluate CV BPS for {model_name}: {e}")

        if best_model_name is None:
            logger.warning("Could not determine best model for DCA; skipping DCA computation.")
            return

        logger.info(f"Selected best model for DCA: {best_model_name} (mean BPS={best_score:.4f})")

        best_model = trained_models[best_model_name]
        if not hasattr(best_model, "predict_proba"):
            logger.warning(f"Best model {best_model_name} has no predict_proba; skipping DCA.")
            return

        dca_dir = Path(self.config.output_dir) / "dca"
        dca_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute DCA for the best model
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_proba = best_model.predict_proba(X_test)[:, 1]

            thresholds, train_curves, train_all = compute_dca_curves(
                y_train,
                {best_model_name: y_train_proba},
            )
            train_df = pd.DataFrame({
                "threshold": thresholds,
                f"{best_model_name}_net_benefit": train_curves[best_model_name],
                "treat_all": train_all,
            })
            train_df.to_csv(dca_dir / f"{best_model_name}_train_dca.csv", index=False)

            thresholds, test_curves, test_all = compute_dca_curves(
                y_test,
                {best_model_name: y_test_proba},
            )
            test_df = pd.DataFrame({
                "threshold": thresholds,
                f"{best_model_name}_net_benefit": test_curves[best_model_name],
                "treat_all": test_all,
            })
            test_df.to_csv(dca_dir / f"{best_model_name}_test_dca.csv", index=False)

            plot_dca_curves(
                y_train,
                {best_model_name: y_train_proba},
                f"DCA - {best_model_name} (train)",
                output_path=str(dca_dir / f"{best_model_name}_train_dca.png"),
                threshold_limits=self.config.threshold_limits,
            )
            plot_dca_curves(
                y_test,
                {best_model_name: y_test_proba},
                f"DCA - {best_model_name} (test)",
                output_path=str(dca_dir / f"{best_model_name}_test_dca.png"),
                threshold_limits=self.config.threshold_limits,
            )

        except Exception as e:
            logger.warning(f"Failed to generate DCA for best model {best_model_name}: {e}")

        # Save the overall best model (which may be a deep learning model)
        self._save_best_model(trained_models, cv_results, data_context)

    def _prepare_data_context(self, X_original: pd.DataFrame, y_original: pd.Series,
                              X_resampled: pd.DataFrame, y_resampled: pd.Series, 
                              split_info: Dict) -> Dict:
        """Prepare data context for model saving.

        This method constructs a minimal but self-contained description of the
        data used for training, so that downstream components (model
        persistence, interpretability) have enough context without relying on
        any external pipeline object.
        """
        # Basic shapes
        original_shape = X_original.shape
        processed_shape = X_resampled.shape

        # Class distribution on original target
        class_counts = y_original.value_counts().to_dict()

        data_context = {
            "original_shape": original_shape,
            "processed_shape": processed_shape,
            "feature_names": list(X_resampled.columns),
            "target_column": self.config.target_column,
            "sampling_method": self.config.sampling_method,
            "class_distribution": class_counts,
            # Attach split info so interpretability can restore the same split
            "data_split_info": split_info,
            # Optionally persist training config for reproducibility
            "training_config": {
                "n_trials": self.config.n_trials,
                "n_folds": self.config.n_folds,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "models_to_train": self.config.models_to_train,
                "sampling_method": self.config.sampling_method,
            },
            # Add categorical and continuous column info if available
            "categorical_columns": self.config.categorical_columns,
            "continuous_columns": self.config.continuous_columns,
        }

        return data_context
    
    def _create_and_train_model(self, model_name: str, X_train: pd.DataFrame,
                                y_train: pd.Series):
        """Create and train a traditional ML model for final saving.

        This is a lightweight helper used only in the final training stage of the
        pipeline. It reuses the existing TraditionalMLTrainer configuration and
        optionally performs grid search if enabled in the config.
        """

        if model_name not in self.ml_trainer.models:
            logger.warning(f"Unknown model '{model_name}' requested for final training.")
            return None

        # If grid search is enabled, obtain an optimized model; otherwise reuse
        # the base model instance from TraditionalMLTrainer.
        if self.config.enable_grid_search:
            logger.info(f"Using parameter optimization for final {model_name} model...")
            optimized_model = self.ml_trainer.parameter_optimizer.get_optimized_model(
                model_name, X_train, y_train
            )
            model = optimized_model if optimized_model is not None else self.ml_trainer.models[model_name]
        else:
            model = self.ml_trainer.models[model_name]

        logger.info(f"Fitting final {model_name} model on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        return model
    
    def _create_and_train_dl_model(self, model_name: str, X_train: pd.DataFrame,
                                   y_train: pd.Series, data_context: Dict):
        """Create and train a deep learning model.

        This helper is currently a thin wrapper around DeepLearningTrainer and
        returns None if deep learning support is not available.
        """
        if not PYTORCH_TABULAR_AVAILABLE or not self.dl_trainer:
            logger.warning("pytorch-tabular not available, skipping DL model training")
            return None

        # Prepare data for pytorch-tabular
        train_data = pd.concat([X_train, y_train], axis=1)

        try:
            tabular_model = self.dl_trainer.create_tabular_model(
                model_name,
                data_context.get('categorical_columns', self.config.categorical_columns),
                data_context.get('continuous_columns', self.config.continuous_columns),
            )
            tabular_model.fit(train=train_data)
            return tabular_model
        except Exception as e:
            logger.error(f"Failed to create/train {model_name}: {e}")
            return None
    
    def _save_individual_model(self, model, model_name: str, cv_results: Dict,
                               data_context: Dict, split_info: Dict):
        """Save an individual traditional ML model with its context.

        This persists the trained model together with configuration,
        performance metrics and data split information so that it can be
        reloaded later for interpretability or deployment.
        """

        if not self.model_persistence:
            logger.warning("ModelPersistenceManager not configured; skipping model save.")
            return

        try:
            # Prepare model configuration for traditional ML models
            model_config = {
                'model_type': 'traditional_ml',
                'model_class': type(model).__name__,
                'model_parameters': self.model_persistence._extract_model_parameters(model),
                'enable_grid_search': bool(self.config.enable_grid_search),
                'cv_performance': cv_results.get(model_name, {}),
            }

            # Prepare training data info (aligned with save_best_model expectations)
            training_data_info = self.model_persistence._build_training_data_info(data_context)

            data_split_info = split_info or {}

            self.model_persistence.save_model_with_context(
                model=model,
                model_name=model_name,
                model_config=model_config,
                training_data=training_data_info,
                performance_metrics=cv_results.get(model_name, {}),
                data_split_info=data_split_info,
            )

            logger.info(f"Saved model '{model_name}' with full training context.")

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
    
    def _save_individual_dl_model(self, model, model_name: str, cv_results: Dict, 
                                 data_context: Dict, split_info: Dict):
        """Save an individual deep learning model with its context."""
        try:
            # Prepare model configuration for DL models
            model_config = {
                'model_type': 'deep_learning',
                'model_class': type(model).__name__,
                'model_parameters': self._extract_dl_model_parameters(model),
                'enable_grid_search': False,  # DL models typically don't use grid search
                'cv_performance': cv_results[model_name],
                'framework': 'pytorch_tabular'
            }
            
            # Save the model using pickle (pytorch-tabular models are pickleable)
            self.model_persistence.save_model_with_context(
                model=model,
                model_name=model_name,
                model_config=model_config,
                training_data=data_context,
                performance_metrics=cv_results[model_name],
                data_split_info=split_info
            )
            
        except Exception as e:
            logger.error(f"Failed to save DL model {model_name}: {e}")
    
    def _extract_dl_model_parameters(self, model) -> Dict:
        """Extract parameters from a deep learning model."""
        try:
            if hasattr(model, 'config'):
                # pytorch-tabular models have config attributes
                config_dict = {}
                for attr_name in ['data_config', 'model_config', 'trainer_config', 'optimizer_config']:
                    if hasattr(model, attr_name):
                        attr = getattr(model, attr_name)
                        if hasattr(attr, '__dict__'):
                            config_dict[attr_name] = {k: v for k, v in attr.__dict__.items() 
                                                    if not k.startswith('_') and not callable(v)}
                return config_dict
            else:
                # Fallback to generic parameter extraction
                return self.model_persistence._extract_model_parameters(model)
        except Exception as e:
            logger.warning(f"Could not extract DL model parameters: {e}")
            return {}
    
    def _save_best_model(self, trained_models: Dict, cv_results: Dict, data_context: Dict):
        """Save the best model based on selection criteria."""
        if trained_models:
            try:
                self.model_persistence.save_best_model(
                    models_dict=trained_models,
                    training_results=cv_results,
                    data_context=data_context
                )
                logger.info("✅ Best model saved successfully")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")

    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        if not self.config.save_metrics:
            return
        
        # Calculate summary statistics
        summary_results = {}
        for model_name, model_results in results.items():
            summary_results[model_name] = {}
            for split in ['train', 'test']:
                if split in model_results:
                    summary_results[model_name][split] = {}
                    for metric, values in model_results[split].items():
                        if values:
                            summary_results[model_name][split][metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values)
                            }
        
        # Save to JSON
        with open(f"{self.config.output_dir}/metrics/summary_results.json", 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Save detailed results
        with open(f"{self.config.output_dir}/metrics/detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_dir}/metrics/")
        
        # Generate model summary and best model selection
        self._generate_model_summary(summary_results)
        self.select_best_model_by_metric(
            primary_metric=self.config.best_model_metric,
            summary_results=summary_results
        )


def create_default_config() -> ModelTrainingConfig:
    """Create default configuration."""
    config = ModelTrainingConfig()
    
    # Set default data path (None means user must specify)
    config.data_path = None
    
    # Default models to train
    config.models_to_train = ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]
    config.deep_learning_models = ["ft_transformer", "gandalf", "tabnet"]
    
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Tabular Biomarker Model Training Pipeline")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the input CSV file (default: auto-detected from RFE output)")
    parser.add_argument("--target_column", type=str, default=None, help="Target column name (if not specified, will be auto-detected)")
    parser.add_argument("--sampling_method", type=str, default="auto", 
                       choices=["auto", "smote", "adasyn", "nearmiss", "tomek", "none", "smote_tomek",
                               "adasyn_nearmiss", "smote_nearmiss", "adasyn_tomek"],
                       help="Sampling method for imbalanced data (auto: automatically determine based on data distribution)")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: auto-managed)")
    parser.add_argument("--models", nargs="*", 
                       default=["random_forest", "xgboost", "lightgbm"],
                       help="Traditional ML models to train")
    parser.add_argument("--dl_models", nargs="*", 
                       default=["ft_transformer", "gandalf", "tabnet"],
                       help="Deep learning models to train")
    parser.add_argument("--enable_grid_search", action="store_true",
                       help="Enable hyperparameter optimization using grid search")
    parser.add_argument("--grid_search_cv", type=int, default=3,
                       help="Number of cross-validation folds for grid search")
    parser.add_argument("--grid_search_scoring", type=str, default="roc_auc",
                       choices=["accuracy", "roc_auc", "f1", "precision", "recall"],
                       help="Scoring metric for grid search")
    parser.add_argument("--best_model_metric", type=str, default="bps",
                       choices=["accuracy", "roc_auc", "f1_score", "mcc", "bps", 
                               "discrimination_score", "calibration_score", "brier_score", "log_loss"],
                       help="Primary metric for best model selection")
    parser.add_argument("--id-col", type=str, default=None,
                       help="Name of the ID column to be removed before training (default: None)")
    parser.add_argument("--save_models", action="store_true", default=True,
                       help="Save trained models for later use (default: True)")
    parser.add_argument("--no_save_models", action="store_true",
                       help="Disable model saving")
    parser.add_argument("--skip_final_training", action="store_true",
                       help="Skip final model training on full data (only do CV evaluation)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    
    # Update config with command line arguments
    # Set default data path if not provided
    if args.data_path is None:
        import glob
        rfe_dir = get_rfe_dir()
        rfe_files = glob.glob(os.path.join(rfe_dir, "rfe_selected_data_*.csv"))
        if rfe_files:
            config.data_path = rfe_files[0]  # Use the first found RFE output file
        else:
            raise FileNotFoundError(f"No RFE output files found in {rfe_dir}")
    else:
        config.data_path = args.data_path
    
    if args.target_column:
        config.target_column = args.target_column
    if getattr(args, 'id_col', None):  # Handle the hyphenated argument name
        config.id_column = getattr(args, 'id_col')
    config.sampling_method = args.sampling_method
    config.n_trials = args.n_trials
    config.n_folds = args.n_folds
    # Use path manager for output directory
    config.output_dir = args.output_dir if args.output_dir is not None else get_model_training_dir()
    config.models_to_train = args.models
    # Wire CLI deep learning model list into config so CrossValidationPipeline
    # can use it via DeepLearningTrainer
    config.deep_learning_models = args.dl_models
    
    # Grid search configuration
    config.enable_grid_search = args.enable_grid_search
    config.grid_search_cv = args.grid_search_cv
    config.grid_search_scoring = args.grid_search_scoring
    
    # Model selection configuration
    config.best_model_metric = args.best_model_metric
    
    # Model saving configuration
    config.save_models = args.save_models and not args.no_save_models
    
    # Final training configuration
    config.skip_final_training = args.skip_final_training
    if config.skip_final_training:
        logger.info("⚡ Skip final training mode enabled - only CV evaluation will be performed")
        config.save_models = False  # 如果跳过最终训练，也跳过模型保存
    
    # Deep learning models configuration
    # Handle special cases for disabling deep learning models
    if args.dl_models and (args.dl_models == ["None"] or args.dl_models == ["none"] or args.dl_models == []):
        config.deep_learning_models = []
        logger.info("Deep learning models disabled")
    else:
        config.deep_learning_models = args.dl_models
    
    # Validate that at least one model type is selected
    if not config.models_to_train and not config.deep_learning_models:
        logger.error("Error: No models selected for training. Please specify at least one traditional ML model (--models) or deep learning model (--dl_models).")
        sys.exit(1)
    
    # Validate data path
    if not config.data_path or not os.path.exists(config.data_path):
        logger.error(f"Data file not found: {config.data_path}")
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = CrossValidationPipeline(config)
        results = pipeline.run_pipeline(config.data_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
