#!/usr/bin/env python3
"""
基于RFE的前向特征选择和refinement脚本
用于对排除共线性后的数据进行特征选择和重要性分析

作者: 
日期: 2025-11-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# ClinMetML path management
from ..utils.paths import get_rfe_dir, get_multicollinearity_dir
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.under_sampling import NearMiss
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import warnings
import os
import json
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')


def create_estimator(algorithm, **params):
    """
    根据算法名称创建估计器
    
    Parameters:
    -----------
    algorithm : str
        算法名称
    **params : dict
        算法参数
        
    Returns:
    --------
    estimator : sklearn estimator
        创建的估计器
    """
    estimators = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'ada_boost': AdaBoostClassifier,
        'extra_trees': ExtraTreesClassifier,
        'ridge': RidgeClassifier,
        'sgd': SGDClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'decision_tree': DecisionTreeClassifier
    }
    
    if algorithm not in estimators:
        raise ValueError(f"不支持的算法: {algorithm}. 支持的算法: {list(estimators.keys())}")
    
    return estimators[algorithm](**params)


def get_feature_importance(estimator, algorithm):
    """
    根据算法类型获取特征重要性
    
    Parameters:
    -----------
    estimator : sklearn estimator
        训练好的估计器
    algorithm : str
        算法名称
        
    Returns:
    --------
    importance : array
        特征重要性数组
    """
    if hasattr(estimator, 'feature_importances_'):
        # 树模型和集成方法
        return estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        # 线性模型
        return np.abs(estimator.coef_[0])
    elif algorithm == 'svm' and hasattr(estimator, 'dual_coef_'):
        # SVM (需要linear kernel)
        if hasattr(estimator, 'coef_'):
            return np.abs(estimator.coef_[0])
        else:
            # 对于非线性SVM，返回均匀重要性
            return np.ones(estimator.n_features_in_) / estimator.n_features_in_
    else:
        # 其他算法返回均匀重要性
        return np.ones(estimator.n_features_in_) / estimator.n_features_in_


class FeatureRefinementSelector:
    """
    基于RFE的特征refinement和前向选择类
    """
    
    def __init__(self, 
                 target_column,
                 algorithm='logistic',
                 test_size=0.3,
                 random_state=42,
                 sampling_strategy='auto',
                 estimator_params=None,
                 output_dir=None,
                 n_features_to_select=None,
                 id_column=None):
        """
        初始化特征选择器
        
        Parameters:
        -----------
        target_column : str
            目标变量列名
        algorithm : str
            机器学习算法名称，支持: 'logistic', 'random_forest', 'gradient_boosting', 
            'ada_boost', 'extra_trees', 'ridge', 'sgd', 'svm', 'knn', 'naive_bayes', 'decision_tree'
        test_size : float
            测试集比例
        random_state : int
            随机种子
        sampling_strategy : str
            采样策略: 'auto'(自动判断), 'balanced'(强制采样), 'imbalanced'(不采样)
        estimator_params : dict
            估计器参数
        output_dir : str
            输出目录
        n_features_to_select : int, optional
            最终要选择的特征数量。如果为None，则使用所有特征。
        id_column : str, optional
            ID列名称，在特征矩阵中将被排除，不参与RFE和建模。
        """
        self.target_column = target_column
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir if output_dir is not None else get_rfe_dir()
        self.n_features_to_select = n_features_to_select
        self.id_column = id_column
        
        # 根据算法设置默认参数
        self.estimator_params = estimator_params or self._get_default_params(algorithm, random_state)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化数据属性
        self.data = None
        self.X = None
        self.y = None
        self.X_resampled = None
        self.y_resampled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rfe_features = None
        self.selection_results = None
        
    def _get_default_params(self, algorithm, random_state):
        """
        根据算法获取默认参数
        """
        default_params = {
            'logistic': {'random_state': random_state, 'max_iter': 1000, 'solver': 'liblinear'},
            'random_forest': {'random_state': random_state, 'n_estimators': 100, 'max_depth': None},
            'gradient_boosting': {'random_state': random_state, 'n_estimators': 100, 'learning_rate': 0.1},
            'ada_boost': {'random_state': random_state, 'n_estimators': 50, 'learning_rate': 1.0},
            'extra_trees': {'random_state': random_state, 'n_estimators': 100, 'max_depth': None},
            'ridge': {'random_state': random_state, 'alpha': 1.0},
            'sgd': {'random_state': random_state, 'max_iter': 1000, 'loss': 'log_loss'},
            'svm': {'random_state': random_state, 'kernel': 'linear', 'probability': True},
            'knn': {'n_neighbors': 5},
            'naive_bayes': {},
            'decision_tree': {'random_state': random_state, 'max_depth': None}
        }
        return default_params.get(algorithm, {'random_state': random_state})
        
    def load_data(self, data_path):
        """
        加载数据
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        """
        print(f"📂 加载数据: {data_path}")
        self.data = pd.read_csv(data_path)
        print(f"数据形状: {self.data.shape}")
        
        # 检查目标列是否存在
        if self.target_column not in self.data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在于数据中")
        
        # 分离特征和目标变量
        feature_cols = [col for col in self.data.columns if col != self.target_column]
        if self.id_column is not None and self.id_column in feature_cols:
            feature_cols = [col for col in feature_cols if col != self.id_column]
            print(f"忽略ID列: {self.id_column}")

        self.X = self.data[feature_cols]
        self.y = self.data[self.target_column]
        
        print(f"特征数量: {self.X.shape[1]}")
        print(f"样本数量: {self.X.shape[0]}")
        print(f"目标变量分布: {self.y.value_counts().to_dict()}")
        
    def apply_undersampling(self):
        """
        根据采样策略应用NearMiss欠采样
        
        采样策略说明:
        - 'auto': 自动判断，不平衡比例>2.0时采样
        - 'balanced': 强制进行采样
        - 'imbalanced': 不进行采样
        """
        print("🔍 分析数据平衡性...")
        
        # 检查类别分布
        class_counts = self.y.value_counts()
        print(f"原始类别分布: {class_counts.to_dict()}")
        
        # 计算不平衡比例
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        print(f"不平衡比例: {imbalance_ratio:.2f}")
        
        # 根据策略决定是否采样
        should_sample = self._should_apply_sampling(imbalance_ratio)
        
        if not should_sample:
            print("⏭️  跳过采样步骤")
            self.X_resampled = self.X.copy()
            self.y_resampled = self.y.copy()
            return
            
        print("🔄 应用NearMiss欠采样...")
        np.random.seed(self.random_state)
        
        try:
            # 应用NearMiss采样
            nr = NearMiss()
            self.X_resampled, self.y_resampled = nr.fit_resample(self.X, self.y)
            
            print(f"采样后数据形状: {self.X_resampled.shape}")
            print(f"采样后类别分布: {pd.Series(self.y_resampled).value_counts().to_dict()}")
            
        except Exception as e:
            print(f"⚠️  采样失败: {str(e)}")
            print("使用原始数据继续分析...")
            self.X_resampled = self.X.copy()
            self.y_resampled = self.y.copy()
    
    def _should_apply_sampling(self, imbalance_ratio):
        """
        根据采样策略和不平衡比例判断是否应该采样
        
        Parameters:
        -----------
        imbalance_ratio : float
            不平衡比例 (最大类别数量 / 最小类别数量)
            
        Returns:
        --------
        bool : 是否应该进行采样
        """
        if self.sampling_strategy == 'balanced':
            print("📋 策略: 强制采样 (balanced)")
            return True
        elif self.sampling_strategy == 'imbalanced':
            print("📋 策略: 不采样 (imbalanced)")
            return False
        elif self.sampling_strategy == 'auto':
            print("📋 策略: 自动判断 (auto)")
            # 自动判断：不平衡比例大于2.0时采样
            if imbalance_ratio >= 2.0:
                print(f"✅ 数据不平衡 (比例: {imbalance_ratio:.2f} >= 2.0)，将进行采样")
                return True
            else:
                print(f"✅ 数据相对平衡 (比例: {imbalance_ratio:.2f} < 2.0)，跳过采样")
                return False
        else:
            print(f"⚠️  未知的采样策略: {self.sampling_strategy}，使用默认策略(auto)")
            return imbalance_ratio >= 2.0
        
    def split_data(self):
        """
        分割训练集和测试集
        """
        print("📊 分割训练集和测试集...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_resampled, 
            self.y_resampled, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y_resampled
        )
        
        print(f"训练集形状: {self.X_train.shape}")
        print(f"测试集形状: {self.X_test.shape}")
        
    def perform_rfe_ranking(self):
        """
        执行RFE特征排名
        """
        print("🔍 执行RFE特征排名...")
        
        # 创建估计器
        estimator = create_estimator(self.algorithm, **self.estimator_params)
        
        # 执行RFE
        rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
        rfe.fit(self.X_train, self.y_train)
        
        # 获取特征排名
        feature_ranking = rfe.ranking_
        
        # 构建特征排名表
        self.rfe_features = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Ranking': feature_ranking
        }).sort_values(by='Ranking')
        
        print(f"RFE排名完成，共 {len(self.rfe_features)} 个特征")
        print("前5个特征:")
        print(self.rfe_features.head())
        
    def perform_forward_selection(self):
        """
        执行前向特征选择分析
        """
        print("🚀 执行前向特征选择分析...")
        
        # 确定要处理的特征数量
        total_features = len(self.rfe_features)
        if self.n_features_to_select is not None:
            n_features = min(self.n_features_to_select, total_features)
            print(f"将按照RFE排名依次选择前 {n_features} 个特征")
        else:
            n_features = total_features
            print(f"将处理所有 {n_features} 个特征")
        
        # 初始化结果DataFrame
        self.selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'ROC'])
        selected_features = []
        
        # 逐步添加特征
        for i in range(n_features):
            # 当前特征
            current_feature = self.rfe_features.iloc[i]['Feature']
            selected_features.append(current_feature)
            
            # 训练模型（仅使用当前选定的特征）
            X_train_subset = self.X_train[selected_features]
            X_test_subset = self.X_test[selected_features]
            
            # 创建并训练模型
            model = create_estimator(self.algorithm, **self.estimator_params)
            model.fit(X_train_subset, self.y_train)
            
            # 获取当前特征的重要性
            feature_importances = get_feature_importance(model, self.algorithm)
            importance = feature_importances[len(selected_features) - 1]
            
            # 预测并计算ROC AUC分数
            y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
            roc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # 保存结果
            new_row = pd.DataFrame({
                'Feature': [current_feature],
                'Importance': [importance],
                'ROC': [roc_score]
            })
            self.selection_results = pd.concat([self.selection_results, new_row], ignore_index=True)
            
            if (i + 1) % 5 == 0 or i == n_features - 1:
                print(f"已处理 {i + 1}/{n_features} 个特征，当前AUC: {roc_score:.4f}")
        
        # 归一化重要性
        importance_sum = self.selection_results['Importance'].sum()
        if importance_sum > 0:
            self.selection_results['Importance_Normalized'] = (
                self.selection_results['Importance'] / importance_sum
            )
        else:
            # 如果重要性全为0，使用均匀分布
            self.selection_results['Importance_Normalized'] = (
                np.ones(len(self.selection_results)) / len(self.selection_results)
            )
        
        print("前向特征选择完成!")
        print(f"最终AUC: {self.selection_results['ROC'].iloc[-1]:.4f}")
        
    def create_visualization(self, highlight_features=None, save_plot=True):
        """
        创建可视化图表
        
        Parameters:
        -----------
        highlight_features : list
            需要高亮显示的特征名称列表
        save_plot : bool
            是否保存图表
        """
        print("📊 创建可视化图表...")
        
        if self.selection_results is None:
            raise ValueError("请先执行前向特征选择")
        
        # 设置高亮特征
        if highlight_features is None:
            highlight_features = []
        
        # 创建颜色渐变 (重要性高的特征颜色深)
        cmap = plt.get_cmap('Blues')
        norm = mcolors.PowerNorm(
            gamma=0.5, 
            vmin=self.selection_results['Importance_Normalized'].min(), 
            vmax=self.selection_results['Importance_Normalized'].max()
        )
        bar_colors = cmap(norm(self.selection_results['Importance_Normalized'].values))
        
        # 开始绘图
        fig, ax1 = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('white')
        
        # 绘制柱状图 (Predictor Importance)
        ax1.bar(
            self.selection_results['Feature'], 
            self.selection_results['Importance_Normalized'], 
            color=bar_colors,
            label='Predictor Importance'
        )
        
        # 设置左侧Y轴 (ax1)
        ax1.set_ylabel('Predictor Importance', fontsize=12)
        
        # 安全地设置Y轴限制
        max_importance = self.selection_results['Importance_Normalized'].max()
        if np.isfinite(max_importance) and max_importance > 0:
            ax1.set_ylim(0, max_importance * 1.1)
        else:
            ax1.set_ylim(0, 1.0)
            
        ax1.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=8))
        ax1.tick_params(axis='y', labelsize=10)
        
        # 设置X轴
        ax1.set_xlabel('Features', fontsize=12)
        ax1.tick_params(axis='x', labelsize=11)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 创建共享X轴的第二个Y轴 (ax2)
        ax2 = ax1.twinx()
        
        # 绘制折线图 (Cumulative AUC)
        ax2.plot(
            self.selection_results['Feature'], 
            self.selection_results['ROC'], 
            color='black', 
            marker='o', 
            linewidth=2,
            markersize=5,
            label='Cumulative AUC'
        )
        
        # 设置右侧Y轴 (ax2)
        ax2.set_ylabel('Cumulative AUC', fontsize=12)
        min_auc = self.selection_results['ROC'].min()
        max_auc = self.selection_results['ROC'].max()
        ax2.set_ylim(min_auc * 0.995, max_auc * 1.005)
        ax2.tick_params(axis='y', labelsize=10)
        
        # 调整X轴标签颜色（高亮特定特征）
        for tick in ax1.get_xticklabels():
            if tick.get_text() in highlight_features:
                tick.set_color('red')
                tick.set_weight('bold')
        
        # 添加网格和标题
        plt.title('RFE-based Forward Feature Selection Results', fontsize=16, pad=20)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.grid(axis='y', linestyle=':', alpha=0.5)
        
        # 优化布局
        fig.tight_layout()
        
        # 保存图表
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'feature_selection_plot_{self.algorithm}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {plot_path}")
        
        plt.show()
        
    def save_results(self, original_data_path=None, save_selected_data=False, selected_data_filename=None):
        """
        保存分析结果和选择的特征数据
        
        Parameters:
        -----------
        original_data_path : str, optional
            原始数据文件路径，用于保存选择的特征数据
        save_selected_data : bool
            是否保存最终选择的特征数据
        selected_data_filename : str, optional
            输出文件名。如果为None，则自动生成文件名
        """
        print("💾 保存分析结果...")
        
        # 保存特征选择结果
        results_path = os.path.join(self.output_dir, f'feature_selection_results_{self.algorithm}.csv')
        self.selection_results.to_csv(results_path, index=False)
        print(f"特征选择结果已保存: {results_path}")
        
        # 保存RFE排名结果
        rfe_path = os.path.join(self.output_dir, f'rfe_ranking_{self.algorithm}.csv')
        self.rfe_features.to_csv(rfe_path, index=False)
        print(f"RFE排名结果已保存: {rfe_path}")
        
        # 保存分析摘要
        summary = {
            'target_column': self.target_column,
            'original_features': self.X.shape[1],
            'original_samples': self.X.shape[0],
            'final_auc': float(self.selection_results['ROC'].iloc[-1]),
            'best_features': self.selection_results.head(10)['Feature'].tolist(),
            'estimator_params': self.estimator_params,
            'sampling_strategy': self.sampling_strategy
        }
        
        summary_path = os.path.join(self.output_dir, f'analysis_summary_{self.algorithm}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"分析摘要已保存: {summary_path}")
        
        # 保存最终选择的特征数据（可选）
        if save_selected_data and original_data_path:
            print("💾 保存最终选择的特征数据...")
            
            if self.selection_results is None:
                raise ValueError("请先执行特征选择分析")
            
            # 读取原始数据
            original_data = pd.read_csv(original_data_path)
            print(f"原始数据形状: {original_data.shape}")
            
            # 获取选择的特征列表
            selected_features = self.selection_results['Feature'].tolist()
            print(f"选择的特征数量: {len(selected_features)}")
            
            # 构建最终列列表（包含目标列）
            final_columns = [self.target_column] + selected_features
            
            # 检查所有列是否存在
            missing_columns = [col for col in final_columns if col not in original_data.columns]
            if missing_columns:
                raise ValueError(f"以下列在原始数据中不存在: {missing_columns}")
            
            # 筛选数据
            selected_data = original_data[final_columns]
            print(f"筛选后数据形状: {selected_data.shape}")
            
            # 生成输出文件名
            if selected_data_filename is None:
                selected_data_filename = f'rfe_selected_data_{self.algorithm}.csv'
            
            output_path = os.path.join(self.output_dir, selected_data_filename)
            
            # 保存数据
            selected_data.to_csv(output_path, index=False)
            print(f"✅ 最终特征数据已保存: {output_path}")
            print(f"包含特征: {selected_features}")
            
            return output_path
        
    def run_complete_analysis(self, data_path, highlight_features=None, save_selected_data=True, selected_data_filename=None):
        """
        运行完整的特征refinement分析
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        highlight_features : list
            需要高亮显示的特征名称列表
        save_selected_data : bool
            是否保存最终选择的特征数据
        selected_data_filename : str, optional
            保存选择特征数据的文件名
        """
        print("🚀 开始完整的特征refinement分析...")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            self.load_data(data_path)
            
            # 2. 应用采样
            self.apply_undersampling()
            
            # 3. 分割数据
            self.split_data()
            
            # 4. RFE排名
            self.perform_rfe_ranking()
            
            # 5. 前向特征选择
            self.perform_forward_selection()
            
            # 6. 创建可视化
            self.create_visualization(highlight_features=highlight_features)
            
            # 7. 保存结果和选择的特征数据
            self.save_results(
                original_data_path=data_path,
                save_selected_data=save_selected_data,
                selected_data_filename=selected_data_filename
            )
            
            print("=" * 60)
            print("✅ 特征refinement分析完成!")
            print(f"最终模型AUC: {self.selection_results['ROC'].iloc[-1]:.4f}")
            print(f"选择的特征数量: {len(self.selection_results)}")
            print(f"结果已保存到: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            raise


class RFESelector:
    """Lightweight wrapper used by ClinMetMLPipeline for RFE selection.

    This class adapts FeatureRefinementSelector to the simpler interface
    expected by ClinMetMLPipeline.run_rfe_selection.
    """

    def __init__(self):
        self.last_results_dir: Optional[str] = None

    def select_features_rfe(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Run RFE-based refinement and return the selected-features DataFrame."""

        # Resolve output directory
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            output_dir = get_rfe_dir()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Map estimator names from short codes to FeatureRefinementSelector algorithms
        estimator = kwargs.get("estimator", "rf")
        estimator_map = {
            "rf": "random_forest",
            "random_forest": "random_forest",
            "logistic": "logistic",
            "svm": "svm",
            "gb": "gradient_boosting",
            "gradient_boosting": "gradient_boosting",
            "et": "extra_trees",
            "extra_trees": "extra_trees",
            "dt": "decision_tree",
            "decision_tree": "decision_tree",
        }
        algorithm = estimator_map.get(estimator, "random_forest")

        n_features_to_select = kwargs.get("n_features_to_select", 20)
        id_column = kwargs.get("id_column")
        test_size = kwargs.get("test_size", 0.3)
        random_state = kwargs.get("random_state", 42)
        sampling_strategy = kwargs.get("sampling_strategy", "auto")

        # Persist current data to CSV so we can reuse the existing file-based API
        input_csv = output_path / "rfe_input_data.csv"
        data.to_csv(input_csv, index=False)

        selector = FeatureRefinementSelector(
            target_column=target_column,
            algorithm=algorithm,
            test_size=test_size,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            estimator_params=None,
            output_dir=str(output_path),
            n_features_to_select=n_features_to_select,
            id_column=id_column,
        )

        selector.run_complete_analysis(
            data_path=str(input_csv),
            highlight_features=None,
            save_selected_data=True,
            selected_data_filename="rfe_selected_data.csv",
        )

        # Load the final selected-features dataset
        final_path = output_path / "rfe_selected_data.csv"
        if not final_path.exists():
            raise FileNotFoundError(
                f"Expected RFE selected dataset at '{final_path}', but it was not found."
            )

        self.last_results_dir = str(output_path)
        selected_data = pd.read_csv(final_path)
        return selected_data


def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description='基于RFE的特征refinement分析')
    
    parser.add_argument('--data_path', type=str, 
                       default=None,
                       help='输入数据文件路径 (默认: auto-detected from multicollinearity reduction output)')
    parser.add_argument('--target_column', type=str, required=True,
                       help='目标变量列名')
    parser.add_argument('--algorithm', type=str, default='random_forest',
                       choices=['logistic', 'random_forest', 'gradient_boosting', 'ada_boost', 
                               'extra_trees', 'ridge', 'sgd', 'svm', 'knn', 'naive_bayes', 'decision_tree'],
                       help='机器学习算法 (默认: logistic)')
    parser.add_argument('--test_size', type=float, default=0.3,
                       help='测试集比例 (默认: 0.3)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--force_strategy', choices=['balanced', 'imbalanced', 'auto'],
                       default='auto', help='采样策略: auto(自动判断), balanced(强制采样), imbalanced(不采样) (默认: auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: auto-managed)')
    parser.add_argument('--highlight_features', type=str, nargs='*',
                       help='需要高亮显示的特征名称')
    parser.add_argument('--n_features_to_select', type=int, default=10,
                       help='按照RFE排名依次选择的特征数量，如果不指定则使用所有特征 (默认: None)')
    parser.add_argument('--selected_data_filename', type=str, default=None,
                       help='保存选择特征数据的文件名，如果不指定则自动生成')
    parser.add_argument('--id_column', type=str, default=None,
                       help='ID列名称，将在特征矩阵中被排除，不参与RFE分析')
    
    args = parser.parse_args()
    
    # 设置默认数据路径
    if args.data_path is None:
        import os
        args.data_path = os.path.join(get_multicollinearity_dir(), "feature_selected_data_no_collinearity.csv")
    
    # 构建估计器参数 (根据算法类型)
    estimator_params = {'random_state': args.random_state}
    
    # 创建特征选择器
    selector = FeatureRefinementSelector(
        target_column=args.target_column,
        algorithm=args.algorithm,
        test_size=args.test_size,
        random_state=args.random_state,
        sampling_strategy=args.force_strategy,
        estimator_params=estimator_params,
        output_dir=args.output_dir,
        n_features_to_select=args.n_features_to_select,
        id_column=args.id_column
    )
    
    # 运行分析
    selector.run_complete_analysis(
        data_path=args.data_path,
        highlight_features=args.highlight_features,
        save_selected_data=True,
        selected_data_filename=args.selected_data_filename
    )


if __name__ == "__main__":
    main()
