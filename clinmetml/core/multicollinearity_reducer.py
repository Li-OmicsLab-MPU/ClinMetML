import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ClinMetML path management
from ..utils.paths import get_multicollinearity_dir, get_feature_selection_dir


def remove_collinearity(data: pd.DataFrame, 
                       target_col: str,
                       correlation_threshold: float = 0.8,
                       distance_cutoff: float = 1.2,
                       method: str = 'spearman',
                       clustering_method: str = 'ward',
                       drop_columns: Optional[List[str]] = None,
                       remove_covariates: bool = False,
                       covariates: Optional[List[str]] = None,
                       visualize: bool = True,
                       save_plots: bool = False,
                       output_dir: str = './') -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove collinearity from features using hierarchical clustering and correlation analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with features
    target_col : str
        Target column name to be removed from analysis
    correlation_threshold : float, default=0.8
        Threshold for high correlation to identify collinear features
    distance_cutoff : float, default=1.2
        Distance cutoff for hierarchical clustering
    method : str, default='spearman'
        Correlation method ('spearman', 'pearson')
    clustering_method : str, default='ward'
        Hierarchical clustering method
    drop_columns : List[str], optional
        Additional columns to drop before analysis (e.g., ['V1'])
    remove_covariates : bool, default=False
        Whether to remove covariate columns from analysis
    covariates : List[str], optional
        List of covariate column names (default: ['BMI', 'age22', 'sex'])
    visualize : bool, default=True
        Whether to create visualizations
    save_plots : bool, default=False
        Whether to save plots to files
    output_dir : str, default='./'
        Directory to save plots
        
    Returns:
    --------
    Tuple containing:
    - refined_data: DataFrame with collinear features removed
    - correlation_matrix: Correlation matrix of original features
    - analysis_results: Dictionary with analysis results and statistics
    """
    
    # Data preprocessing
    processed_data = data.copy()
    
    # Remove target column
    if target_col in processed_data.columns:
        processed_data = processed_data.drop(columns=[target_col])
        print(f"Removed target column: {target_col}")
    else:
        print(f"Warning: Target column '{target_col}' not found in data")
    
    # Remove covariates if specified
    if remove_covariates:

        
        existing_covariates = [col for col in covariates if col in processed_data.columns]
        if existing_covariates:
            processed_data = processed_data.drop(columns=existing_covariates)
            print(f"Removed covariate columns: {existing_covariates}")
        else:
            print(f"Warning: No covariate columns found in data. Looking for: {covariates}")
    
    # Drop additional specified columns
    if drop_columns:
        existing_drop_cols = [col for col in drop_columns if col in processed_data.columns]
        if existing_drop_cols:
            processed_data = processed_data.drop(columns=existing_drop_cols)
            print(f"Removed additional columns: {existing_drop_cols}")
    
    print(f"Original data shape: {processed_data.shape}")
    
    # Calculate correlation matrix
    print(f"Calculating {method} correlation matrix...")
    correlation_matrix = processed_data.corr(method=method)
    
    # Hierarchical clustering
    print("Performing hierarchical clustering...")
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Ensure the distance matrix is symmetric with zero diagonal to satisfy
    # scipy.spatial.distance.squareform requirements.
    # Numerical imprecision or NaNs in the correlation matrix can lead to
    # slight asymmetry, which would otherwise raise
    
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix.values, 0.0)

    gene_linkage = linkage(squareform(distance_matrix), method=clustering_method)
    
    # Create dendrogram and get clustering results
    dendro = dendrogram(gene_linkage, 
                       labels=correlation_matrix.columns.tolist(),
                       no_plot=True)
    
    # Get clusters based on distance cutoff
    clusters = fcluster(gene_linkage, distance_cutoff, criterion='distance')
    
    # Identify features to remove based on clustering and correlation
    features_to_remove = identify_collinear_features(
        correlation_matrix, clusters, correlation_threshold
    )
    
    # Remove collinear features
    refined_data = processed_data.drop(columns=features_to_remove)
    
    print(f"Refined data shape: {refined_data.shape}")
    print(f"Removed {len(features_to_remove)} collinear features: {features_to_remove}")
    
    # Prepare analysis results
    analysis_results = {
        'original_features': list(processed_data.columns),
        'refined_features': list(refined_data.columns),
        'removed_features': features_to_remove,
        'clusters': dict(zip(correlation_matrix.columns, clusters)),
        'n_clusters': len(np.unique(clusters)),
        'linkage_matrix': gene_linkage,
        'dendrogram_data': dendro
    }
    
    # Visualization
    if visualize:
        create_visualizations(correlation_matrix, gene_linkage, dendro, 
                            distance_cutoff, save_plots, output_dir)
    
    return refined_data, correlation_matrix, analysis_results


class MulticollinearityReducer:
    """High-level multicollinearity reduction interface used by ClinMetMLPipeline.

    This class provides a `reduce_multicollinearity` method compatible with the
    pipeline configuration shown in the tests and examples, but internally it
    reuses the `remove_collinearity` function implemented above.
    """

    def __init__(self):
        self.last_results_dir: Optional[str] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[dict] = None

    def reduce_multicollinearity(
        self,
        data: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Reduce multicollinearity and return a DataFrame with refined features.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe including the target column.
        target_column : str
            Name of the target column in `data`.
        **kwargs : dict
            Additional configuration, typically coming from
            `pipeline_config['multicollinearity_reduction']`, e.g.:

            - correlation_threshold
            - distance_cutoff
            - method
            - drop_columns
            - remove_covariates / covariates
            - visualize, save_plots
            - output_dir
            - vif_threshold (accepted but currently unused)
        """

        # Resolve output directory
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            output_dir = get_multicollinearity_dir()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Map common parameters with sensible defaults
        correlation_threshold = kwargs.get("correlation_threshold", 0.8)
        distance_cutoff = kwargs.get("distance_cutoff", 1.2)
        method = kwargs.get("method", "spearman")
        clustering_method = kwargs.get("clustering_method", "ward")
        drop_columns = kwargs.get("drop_columns")
        remove_covariates = kwargs.get("remove_covariates", False)
        covariates = kwargs.get("covariates")
        visualize = kwargs.get("visualize", True)
        save_plots = kwargs.get("save_plots", True)

        # Accept but ignore vif_threshold for now to stay compatible with configs
        _ = kwargs.get("vif_threshold", None)

        refined_data, corr_matrix, analysis_results = remove_collinearity(
            data=data,
            target_col=target_column,
            correlation_threshold=correlation_threshold,
            distance_cutoff=distance_cutoff,
            method=method,
            clustering_method=clustering_method,
            drop_columns=drop_columns,
            remove_covariates=remove_covariates,
            covariates=covariates,
            visualize=visualize,
            save_plots=save_plots,
            output_dir=str(output_path),
        )

        # Re-attach target column for downstream steps (e.g., RFE and model training),
        # mirroring the behavior of save_refined_dataset which keeps target_col.
        if target_column in data.columns and target_column not in refined_data.columns:
            refined_with_target = pd.concat([data[[target_column]].reset_index(drop=True),
                                             refined_data.reset_index(drop=True)], axis=1)
        else:
            refined_with_target = refined_data

        # Store references for potential downstream use
        self.last_results_dir = str(output_path)
        self.correlation_matrix = corr_matrix
        self.analysis_results = analysis_results

        # Save refined data for inspection and potential reuse
        output_file = output_path / "multicollinearity_reduced_data.csv"
        refined_with_target.to_csv(output_file, index=False)

        return refined_with_target


def identify_collinear_features(correlation_matrix: pd.DataFrame, 
                               clusters: np.ndarray,
                               threshold: float = 0.8) -> List[str]:
    """
    Identify collinear features to remove based on correlation and clustering.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix of features
    clusters : np.ndarray
        Cluster assignments for each feature
    threshold : float
        Correlation threshold for identifying collinear features
        
    Returns:
    --------
    List of feature names to remove
    """
    features_to_remove = []
    feature_names = correlation_matrix.columns.tolist()
    
    # Group features by clusters
    cluster_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(feature_names[i])
    
    # Within each cluster, identify highly correlated features
    for cluster_id, features in cluster_groups.items():
        if len(features) > 1:
            cluster_corr = correlation_matrix.loc[features, features]
            
            # Find pairs with high correlation
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    corr_val = abs(cluster_corr.iloc[i, j])
                    if corr_val > threshold:
                        # Remove the feature with higher average correlation with others
                        feat1, feat2 = features[i], features[j]
                        avg_corr1 = abs(correlation_matrix[feat1]).mean()
                        avg_corr2 = abs(correlation_matrix[feat2]).mean()
                        
                        feature_to_remove = feat1 if avg_corr1 > avg_corr2 else feat2
                        if feature_to_remove not in features_to_remove:
                            features_to_remove.append(feature_to_remove)
    
    return features_to_remove


def create_visualizations(correlation_matrix: pd.DataFrame,
                         linkage_matrix: np.ndarray,
                         dendro_data: dict,
                         distance_cutoff: float = 1.2,
                         save_plots: bool = False,
                         output_dir: str = './'):
    """
    Create visualizations for correlation analysis and clustering.
    """
    
    # Create output directory if it doesn't exist
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
        # Create image subdirectory
        image_dir = os.path.join(output_dir, 'image')
        os.makedirs(image_dir, exist_ok=True)
    
    # 1. Dendrogram
    plt.figure(figsize=(15, 8))
    plt.title('Dendrogram of Predictors', fontsize=16)
    plt.xlabel('Predictor', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    
    dendrogram(linkage_matrix,
               labels=correlation_matrix.columns.tolist(),
               leaf_rotation=90,
               leaf_font_size=10)
    
    plt.axhline(y=distance_cutoff, 
                color='r',
                linestyle='--',
                linewidth=1.5,
                label=f'Distance Cutoff = {distance_cutoff}')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{output_dir}/image/dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Clustered correlation heatmap
    reordered_indices = dendro_data['leaves']
    reordered_labels = correlation_matrix.columns[reordered_indices]
    clustered_cor = correlation_matrix.reindex(index=reordered_labels, columns=reordered_labels)
    
    plt.figure(figsize=(12, 12))
    plt.title('Spearman Correlation Heatmap (Clustered)', fontsize=16)
    
    sns.heatmap(clustered_cor,
                cmap='RdYlBu_r',
                annot=False,
                linewidths=.5,
                cbar_kws={
                    "label": "Spearman Correlation",
                    "shrink": 0.7
                })
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{output_dir}/image/correlation_heatmap_clustered.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Clustermap
    plt.figure(figsize=(12, 12))
    g = sns.clustermap(correlation_matrix,
                       method='ward',
                       cmap='RdYlBu_r',
                       annot=False,
                       linewidths=.5,
                       figsize=(12, 12),
                       xticklabels=True,
                       yticklabels=True,
                       cbar_kws={
                           "label": "Spearman Correlation",
                           "shrink": 0.7
                       })
    
    if save_plots:
        g.savefig(f'{output_dir}/image/correlation_clustermap.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_collinearity_statistics(correlation_matrix: pd.DataFrame,
                                  analysis_results: dict) -> dict:
    """
    Generate statistical summary of collinearity analysis.
    """
    stats = {
        'total_features': len(analysis_results['original_features']),
        'retained_features': len(analysis_results['refined_features']),
        'removed_features': len(analysis_results['removed_features']),
        'removal_percentage': len(analysis_results['removed_features']) / len(analysis_results['original_features']) * 100,
        'n_clusters': analysis_results['n_clusters'],
        'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
        'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
        'mean_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
        'std_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].std()
    }
    
    return stats


def save_refined_dataset(original_data_path: str,
                        removed_features: List[str],
                        output_path: str = None,
                        output_dir: str = None,
                        target_col: str = None,
                        remove_covariates: bool = False,
                        covariates: List[str] = None) -> str:
    """
    将原始数据文件去除共线性特征后保存为新文件。
    
    Parameters:
    -----------
    original_data_path : str
        原始数据文件路径
    removed_features : List[str]
        需要去除的共线性特征列表
    output_path : str, optional
        输出文件路径，如果为None则自动生成
    output_dir : str, optional
        输出目录，如果为None则使用路径管理器的默认目录
    target_col : str, optional
        目标列名称，会被保留在最终数据中
    remove_covariates : bool, default=False
        是否移除协变量列
    covariates : List[str], optional
        协变量列名称列表
        
    Returns:
    --------
    str: 保存的文件路径
    """
    
    # 加载原始数据
    print(f"Loading original data from: {original_data_path}")
    original_data = pd.read_csv(original_data_path)
    print(f"Original data shape: {original_data.shape}")
    
    # 创建处理后的数据副本
    refined_data = original_data.copy()
    
    # 去除共线性特征
    if removed_features:
        existing_removed_features = [col for col in removed_features if col in refined_data.columns]
        if existing_removed_features:
            refined_data = refined_data.drop(columns=existing_removed_features)
            print(f"Removed {len(existing_removed_features)} collinear features: {existing_removed_features}")
        else:
            print("No collinear features found in the data to remove")
    else:
        print("No collinear features to remove")
    
    # 移除协变量（如果指定）
    if remove_covariates and covariates:
        existing_covariates = [col for col in covariates if col in refined_data.columns]
        if existing_covariates:
            refined_data = refined_data.drop(columns=existing_covariates)
            print(f"Removed covariate columns: {existing_covariates}")
    
    # 生成输出文件路径
    if output_path is None:
        import os
        base_name = os.path.basename(original_data_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 使用传入的output_dir或默认值
        if output_dir is None:
            output_dir = get_multicollinearity_dir()
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name_without_ext}_no_collinearity.csv")
    
    # 保存处理后的数据
    refined_data.to_csv(output_path, index=False)
    print(f"Refined dataset saved to: {output_path}")
    print(f"Final data shape: {refined_data.shape}")
    
    # 显示处理摘要
    original_features = len(original_data.columns)
    final_features = len(refined_data.columns)
    removed_count = original_features - final_features
    
    print(f"\n=== Dataset Refinement Summary ===")
    print(f"Original features: {original_features}")
    print(f"Final features: {final_features}")
    print(f"Removed features: {removed_count}")
    print(f"Retention rate: {final_features/original_features*100:.1f}%")
    
    return output_path


# Example usage and main function
def main(target_col: str, 
         data_path: str = None,
         covariates: List[str] = None,
         **kwargs):
    """
    Example usage of the collinearity removal function.
    
    Args:
        target_col: Target column name from user input
        data_path: Path to input data file
        covariates: List of covariate columns from user input
        **kwargs: Additional parameters for remove_collinearity function
    """
    # covariates will be None if not provided by user
    
    # Set default data path if not provided
    if data_path is None:
        import os
        data_path = os.path.join(get_feature_selection_dir(), "feature_selected_data.csv")
    
    # Load data (adjust path as needed)
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data with shape: {data.shape}")
        
        # Remove collinearity
        refined_data, correlation_matrix, analysis_results = remove_collinearity(
            data=data,
            target_col=target_col,  # From user input --target-col
            correlation_threshold=kwargs.get('correlation_threshold', 0.8),
            distance_cutoff=kwargs.get('distance_cutoff', 1.2),
            method=kwargs.get('method', 'spearman'),
            drop_columns=None,  # No additional columns to drop, target_col is handled separately
            remove_covariates=kwargs.get('remove_covariates', False),
            covariates=covariates,  # From user input --covariates
            visualize=kwargs.get('visualize', True),
            save_plots=kwargs.get('save_plots', True),
            output_dir=kwargs.get('output_dir', './')
        )
        
        # Generate statistics
        stats = analyze_collinearity_statistics(correlation_matrix, analysis_results)
        
        print("\n=== Collinearity Analysis Results ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save refined data (current method - for compatibility)
        # Generate output filename based on original data file name
        import os
        original_base_name = os.path.basename(data_path)
        original_name_without_ext = os.path.splitext(original_base_name)[0]
        default_output_file = f"{original_name_without_ext}_no_collinearity.csv"
        output_file = kwargs.get('output_file', default_output_file)
        refined_data.to_csv(output_file, index=False)
        print(f"\nRefined data saved to '{output_file}'")
        
        # Save refined dataset from original data (new method)
        if kwargs.get('save_refined_dataset', True):  # Default to True
            refined_dataset_path = save_refined_dataset(
                original_data_path=data_path,
                removed_features=analysis_results['removed_features'],
                output_dir=kwargs.get('output_dir', get_multicollinearity_dir()),
                target_col=target_col,
                remove_covariates=kwargs.get('remove_covariates', False),
                covariates=covariates
            )
        
        return refined_data, correlation_matrix, analysis_results
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please make sure the file path is correct or run feature selection first.")
        return None, None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Refinement - Remove Collinearity")
    parser.add_argument("--data", default=None,
                       help="Input data file path (default: auto-detected from feature selection output)")
    parser.add_argument("--target-col", required=True,
                       help="Target column name to remove from analysis")
    parser.add_argument("--correlation-threshold", type=float, default=0.8,
                       help="Correlation threshold for identifying collinear features (default: 0.8)")
    parser.add_argument("--distance-cutoff", type=float, default=1.2,
                       help="Distance cutoff for hierarchical clustering (default: 1.2)")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"],
                       help="Correlation method (default: spearman)")
    parser.add_argument("--clustering-method", default="ward",
                       help="Hierarchical clustering method (default: ward)")
    parser.add_argument("--drop-columns", nargs='+', 
                       help="Additional columns to drop (space-separated)")
    parser.add_argument("--remove-covariates", action="store_true",
                       help="Remove covariate columns from analysis")
    parser.add_argument("--covariates", nargs='+', default=None,
                       help="Covariate column names (optional)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for plots and analysis results (default: auto-managed)")
    parser.add_argument("--output-file", default=None,
                       help="Output file name for refined data (default: [original_filename]_no_collinearity.csv)")
    
    args = parser.parse_args()
    
    try:
        # Call main function with user parameters
        refined_data, correlation_matrix, analysis_results = main(
            target_col=args.target_col,  # From user input --target-col
            data_path=args.data,
            covariates=args.covariates,  # From user input --covariates
            correlation_threshold=args.correlation_threshold,
            distance_cutoff=args.distance_cutoff,
            method=args.method,
            clustering_method=args.clustering_method,
            drop_columns=args.drop_columns,
            remove_covariates=args.remove_covariates,
            visualize=True,
            save_plots=True,
            output_dir=args.output_dir,
            output_file=args.output_file,
            save_refined_dataset=True
        )
        
        if refined_data is not None:
            # Save analysis results
            import json
            # Create output directory for analysis results
            import os
            results_dir = args.output_dir  # Use output_dir from command line arguments
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate results file path based on original data file name
            original_base_name = os.path.basename(args.data)
            original_name_without_ext = os.path.splitext(original_base_name)[0]
            results_filename = f"{original_name_without_ext}_no_collinearity_analysis.json"
            results_file = os.path.join(results_dir, results_filename)
            with open(results_file, 'w') as f:
                # Convert numpy arrays and numpy scalars to JSON serializable types
                def convert_numpy_types(obj):
                    """Convert numpy types to native Python types for JSON serialization"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                json_results = {}
                for key, value in analysis_results.items():
                    if key == 'dendrogram_data':
                        # Skip dendrogram data as it's complex to serialize
                        continue
                    else:
                        json_results[key] = convert_numpy_types(value)
                    
                json.dump(json_results, f, indent=2)
            print(f"Analysis results saved to '{results_file}'")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {args.data}")
        print("Please make sure the file path is correct or run feature selection first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
