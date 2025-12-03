"""
Data Cleaning Module for Tabular Biomarker Analysis

This module provides comprehensive data preprocessing capabilities including:
1. Data quality analysis
2. Missing value imputation (KNN, BPCA, PPCA, SVD, CARTPMM, Lasso.norm)
3. Column filtering based on missing value percentage
4. Data normalization, log2 transformation, and Autonorm
5. Data saving functionality
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

# Core libraries
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing class for biomarker data.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the DataCleaner.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.data_quality_report = {}
        self.imputation_history = {}
        self.transformation_history = {}
        
    def analyze_data_quality(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze data quality and generate comprehensive report.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save quality report plots
            
        Returns:
            Dictionary containing data quality metrics
        """
        print("🔍 Analyzing data quality...")
        
        report = {
            'basic_info': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
            },
            'missing_values': {},
            'duplicates': {
                'n_duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'numerical_summary': {},
            'outliers': {}
        }
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        report['missing_values'] = {
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'columns_over_20_percent_missing': missing_percentages[missing_percentages > 20].index.tolist()
        }
        
        # Numerical columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not df[col].isnull().all():
                col_data = df[col].dropna()
                report['numerical_summary'][col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data))
                }
                
                # Outlier detection using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                report['outliers'][col] = {
                    'n_outliers': len(outliers),
                    'outlier_percentage': (len(outliers) / len(col_data)) * 100
                }
        
        # Generate visualization if output directory provided
        if output_dir:
            self._generate_quality_plots(df, output_dir)
        
        self.data_quality_report = report
        
        # Print summary
        print(f"📊 Data Quality Summary:")
        print(f"   • Shape: {report['basic_info']['n_rows']} rows × {report['basic_info']['n_columns']} columns")
        print(f"   • Missing values: {len(report['missing_values']['columns_with_missing'])} columns affected")
        print(f"   • Duplicates: {report['duplicates']['n_duplicate_rows']} rows ({report['duplicates']['duplicate_percentage']:.2f}%)")
        print(f"   • Columns with >20% missing: {len(report['missing_values']['columns_over_20_percent_missing'])}")
        
        return report
    
    def _generate_quality_plots(self, df: pd.DataFrame, output_dir: str):
        """Generate data quality visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Missing values heatmap
        plt.figure(figsize=(12, 8))
        missing_data = df.isnull()
        sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'missing_values_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Missing values bar plot
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            plt.figure(figsize=(12, 6))
            missing_counts.plot(kind='bar')
            plt.title('Missing Values Count by Column')
            plt.xlabel('Columns')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'missing_values_barplot.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def impute_missing_values(self, df: pd.DataFrame, method: str = 'knn', **kwargs) -> pd.DataFrame:
        """
        Impute missing values using specified method.
        
        Args:
            df: Input DataFrame
            method: Imputation method (default: 'knn')
                   Available methods: 'knn', 'bpca', 'ppca', 'svd', 'cartpmm', 'lasso_norm'
            **kwargs: Additional parameters for imputation methods:
                     - n_neighbors (int): For KNN method (default: 5)
                     - n_components (int): For PCA-based methods (default: 10)
                     - alpha (float): For Lasso method (default: 1.0)
            
        Returns:
            DataFrame with imputed values
        """
        print(f"🔧 Imputing missing values using {method.upper()} method...")
        
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("⚠️ No numeric columns found for imputation")
            return df_imputed
        
        # Store original missing pattern
        missing_before = df[numeric_cols].isnull().sum()
        
        if method.lower() == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        elif method.lower() == 'bpca':
            df_imputed[numeric_cols] = self._bpca_imputation(df[numeric_cols], **kwargs)
            
        elif method.lower() == 'ppca':
            df_imputed[numeric_cols] = self._ppca_imputation(df[numeric_cols], **kwargs)
            
        elif method.lower() == 'svd':
            df_imputed[numeric_cols] = self._svd_imputation(df[numeric_cols], **kwargs)
            
        elif method.lower() == 'cartpmm':
            df_imputed[numeric_cols] = self._cart_pmm_imputation(df[numeric_cols], **kwargs)
            
        elif method.lower() == 'lasso_norm':
            df_imputed[numeric_cols] = self._lasso_norm_imputation(df[numeric_cols], **kwargs)
            
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # Store imputation history
        missing_after = df_imputed[numeric_cols].isnull().sum()
        self.imputation_history[method] = {
            'missing_before': missing_before.to_dict(),
            'missing_after': missing_after.to_dict(),
            'imputed_values': (missing_before - missing_after).to_dict()
        }
        
        print(f"✅ Imputation completed. Imputed {(missing_before - missing_after).sum()} values")
        return df_imputed
    
    def _bpca_imputation(self, df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
        """Bayesian PCA imputation."""
        from sklearn.decomposition import PCA
        
        # Simple BPCA approximation using iterative PCA
        df_filled = df.fillna(df.mean())
        
        if n_components is None:
            n_components = min(10, df.shape[1] - 1)
        
        # Store original missing pattern
        original_mask = df.isnull()
        
        for _ in range(5):  # 5 iterations
            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = pca.fit_transform(df_filled)
            reconstructed = pca.inverse_transform(transformed)
            
            # Convert reconstructed array back to DataFrame
            reconstructed_df = pd.DataFrame(reconstructed, columns=df.columns, index=df.index)
            
            # Update only originally missing values
            df_filled = df_filled.mask(original_mask, reconstructed_df)
        
        return df_filled
    
    def _ppca_imputation(self, df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
        """Probabilistic PCA imputation."""
        from sklearn.decomposition import PCA
        
        if n_components is None:
            n_components = min(10, df.shape[1] - 1)
        
        # Fill missing values with mean initially
        df_filled = df.fillna(df.mean())
        
        # Store original missing pattern
        original_mask = df.isnull()
        
        # Iterative PPCA
        for _ in range(10):
            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = pca.fit_transform(df_filled)
            reconstructed = pca.inverse_transform(transformed)
            
            # Add noise for probabilistic component
            noise = np.random.normal(0, 0.1, reconstructed.shape)
            reconstructed += noise
            
            # Convert reconstructed array back to DataFrame
            reconstructed_df = pd.DataFrame(reconstructed, columns=df.columns, index=df.index)
            
            # Update only originally missing values
            df_filled = df_filled.mask(original_mask, reconstructed_df)
        
        return df_filled
    
    def _svd_imputation(self, df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
        """SVD-based imputation."""
        from sklearn.decomposition import TruncatedSVD
        
        if n_components is None:
            n_components = min(10, df.shape[1] - 1)
        
        # Fill missing values with column means
        df_filled = df.fillna(df.mean())
        
        # Apply SVD
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        transformed = svd.fit_transform(df_filled)
        reconstructed = svd.inverse_transform(transformed)
        
        # Convert reconstructed array back to DataFrame
        reconstructed_df = pd.DataFrame(reconstructed, columns=df.columns, index=df.index)
        
        # Update missing values
        original_mask = df.isnull()
        df_result = df.copy()
        df_result = df_result.mask(original_mask, reconstructed_df)
        
        return df_result
    
    def _cart_pmm_imputation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """CART with Predictive Mean Matching imputation."""
        from sklearn.ensemble import RandomForestRegressor
        
        df_imputed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().any():
                # Separate complete and incomplete cases
                complete_mask = df[col].notna()
                incomplete_mask = df[col].isna()
                
                if complete_mask.sum() == 0:
                    continue
                
                # Use other columns as predictors
                predictor_cols = [c for c in df.columns if c != col]
                
                # Fill missing values in predictors with mean for training
                X_complete = df.loc[complete_mask, predictor_cols].fillna(df[predictor_cols].mean())
                y_complete = df.loc[complete_mask, col]
                
                X_incomplete = df.loc[incomplete_mask, predictor_cols].fillna(df[predictor_cols].mean())
                
                if len(X_incomplete) == 0:
                    continue
                
                # Train Random Forest
                rf = RandomForestRegressor(n_estimators=10, random_state=self.random_state)
                rf.fit(X_complete, y_complete)
                
                # Predict missing values
                predictions = rf.predict(X_incomplete)
                
                # PMM: Find closest observed values
                for i, pred in enumerate(predictions):
                    distances = np.abs(y_complete - pred)
                    closest_idx = distances.nsmallest(5).index
                    imputed_value = np.random.choice(y_complete[closest_idx])
                    df_imputed.loc[incomplete_mask, col].iloc[i] = imputed_value
        
        return df_imputed
    
    def _lasso_norm_imputation(self, df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
        """Lasso regression with normalization for imputation."""
        from sklearn.preprocessing import StandardScaler
        
        df_imputed = df.copy()
        scaler = StandardScaler()
        
        # Normalize data
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df.fillna(df.mean())),
            columns=df.columns,
            index=df.index
        )
        
        for col in df.columns:
            if df[col].isnull().any():
                # Separate complete and incomplete cases
                complete_mask = df[col].notna()
                incomplete_mask = df[col].isna()
                
                if complete_mask.sum() == 0:
                    continue
                
                # Use other columns as predictors
                predictor_cols = [c for c in df.columns if c != col]
                
                X_complete = df_normalized.loc[complete_mask, predictor_cols]
                y_complete = df_normalized.loc[complete_mask, col]
                
                X_incomplete = df_normalized.loc[incomplete_mask, predictor_cols]
                
                if len(X_incomplete) == 0:
                    continue
                
                # Train Lasso
                lasso = Lasso(alpha=alpha, random_state=self.random_state)
                lasso.fit(X_complete, y_complete)
                
                # Predict and denormalize
                predictions_norm = lasso.predict(X_incomplete)
                
                # Denormalize predictions
                col_mean = df[col].mean()
                col_std = df[col].std()
                predictions = predictions_norm * col_std + col_mean
                
                df_imputed.loc[incomplete_mask, col] = predictions
        
        return df_imputed
    
    def filter_high_missing_columns(self, df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
        """
        Filter out columns with missing values above threshold.
        
        Args:
            df: Input DataFrame
            threshold: Missing value threshold (default: 0.2 = 20%)
                      Range: 0.0 to 1.0 (e.g., 0.1 = 10%, 0.3 = 30%)
                      Columns with missing percentage > threshold will be removed
            
        Returns:
            DataFrame with filtered columns
        """
        print(f"🔍 Filtering columns with >{threshold*100}% missing values...")
        
        missing_percentages = df.isnull().sum() / len(df)
        columns_to_keep = missing_percentages[missing_percentages <= threshold].index
        columns_to_remove = missing_percentages[missing_percentages > threshold].index
        
        print(f"📊 Removing {len(columns_to_remove)} columns: {list(columns_to_remove)}")
        print(f"✅ Keeping {len(columns_to_keep)} columns")
        
        return df[columns_to_keep]
    
    def normalize_and_transform(self, df: pd.DataFrame, 
                              normalization: str = 'standard',
                              log_transform: bool = True,
                              autonorm: bool = True) -> pd.DataFrame:
        """
        Apply normalization, log2 transformation, and autonorm.
        
        Args:
            df: Input DataFrame
            normalization: Type of normalization ('standard', 'minmax', 'robust')
            log_transform: Whether to apply log2 transformation
            autonorm: Whether to apply autonorm
            
        Returns:
            Transformed DataFrame
        """
        print("🔄 Applying normalization and transformations...")
        
        df_transformed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("⚠️ No numeric columns found for transformation")
            return df_transformed
        
        # Log2 transformation
        if log_transform:
            print("   • Applying log2 transformation...")
            for col in numeric_cols:
                # Add small constant to handle zeros and negative values
                min_val = df_transformed[col].min()
                if min_val <= 0:
                    df_transformed[col] = df_transformed[col] - min_val + 1
                df_transformed[col] = np.log2(df_transformed[col])
        
        # Normalization
        print(f"   • Applying {normalization} normalization...")
        if normalization == 'standard':
            scaler = StandardScaler()
        elif normalization == 'minmax':
            scaler = MinMaxScaler()
        elif normalization == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
        
        # Autonorm (additional normalization step)
        if autonorm:
            print("   • Applying autonorm...")
            df_transformed[numeric_cols] = self._autonorm(df_transformed[numeric_cols])
        
        # Store transformation history
        self.transformation_history = {
            'normalization': normalization,
            'log_transform': log_transform,
            'autonorm': autonorm,
            'numeric_columns': list(numeric_cols)
        }
        
        print("✅ Transformations completed")
        return df_transformed
    
    def _autonorm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply autonorm transformation (auto-scaling with additional normalization).
        """
        df_autonorm = df.copy()
        
        for col in df.columns:
            col_data = df[col]
            # Calculate robust statistics
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))  # Median Absolute Deviation
            
            if mad != 0:
                df_autonorm[col] = (col_data - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
            else:
                # Fallback to standard normalization if MAD is 0
                mean = col_data.mean()
                std = col_data.std()
                if std != 0:
                    df_autonorm[col] = (col_data - mean) / std
        
        return df_autonorm
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str, 
                         save_report: bool = True) -> None:
        """
        Save cleaned data and optional quality report.
        
        Args:
            df: Cleaned DataFrame
            output_path: Output file path
            save_report: Whether to save quality report
        """
        print(f"💾 Saving cleaned data to {output_path}...")
        
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(output_path + '.csv', index=False)
        
        # Save quality report
        if save_report and self.data_quality_report:
            report_path = output_file.parent / f"{output_file.stem}_quality_report.json"
            
            # Combine all reports
            full_report = {
                'timestamp': datetime.now().isoformat(),
                'data_quality': self.data_quality_report,
                'imputation_history': self.imputation_history,
                'transformation_history': self.transformation_history
            }
            
            with open(report_path, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            
            print(f"📊 Quality report saved to {report_path}")
        
        print("✅ Data saved successfully")
    
    def clean_pipeline(self, df: pd.DataFrame,
                      imputation_method: str = 'knn',
                      filter_missing_threshold: Optional[float] = None,
                      normalization: str = 'standard',
                      log_transform: bool = True,
                      autonorm: bool = True,
                      output_path: Optional[str] = None,
                      quality_plots_dir: Optional[str] = None,
                      id_column: Optional[str] = None,
                      target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Args:
            df: Input DataFrame
            imputation_method: Method for missing value imputation (default: 'knn')
                              Options: 'knn', 'bpca', 'ppca', 'svd', 'cartpmm', 'lasso_norm'
            filter_missing_threshold: Threshold for filtering high-missing columns (default: None to skip)
                                    Range: 0.0-1.0 (e.g., 0.2 = remove columns with >20% missing)
            normalization: Normalization method (default: 'standard')
                          Options: 'standard', 'minmax', 'robust'
            log_transform: Whether to apply log2 transformation (default: True)
            autonorm: Whether to apply autonorm (default: True)
            output_path: Path to save cleaned data (default: None to skip saving)
            quality_plots_dir: Directory to save quality plots (None to skip)
            id_column: Name of ID column to exclude from cleaning (default: None)
            target_column: Name of target column to exclude from cleaning (default: None)
            
        Returns:
            Cleaned DataFrame
        """
        print("🚀 Starting complete data cleaning pipeline...")
        print("=" * 50)
        
        # Identify columns to exclude from cleaning
        exclude_columns = []
        if id_column and id_column in df.columns:
            exclude_columns.append(id_column)
            print(f"🔒 ID column '{id_column}' will be excluded from cleaning")
        elif id_column and id_column not in df.columns:
            print(f"⚠️ Warning: ID column '{id_column}' not found in data")
            
        if target_column and target_column in df.columns:
            exclude_columns.append(target_column)
            print(f"🔒 Target column '{target_column}' will be excluded from cleaning")
        elif target_column and target_column not in df.columns:
            print(f"⚠️ Warning: Target column '{target_column}' not found in data")
        
        # Separate excluded columns from data to be cleaned
        if exclude_columns:
            excluded_data = df[exclude_columns].copy()
            df_to_clean = df.drop(columns=exclude_columns)
            print(f"📊 Cleaning {df_to_clean.shape[1]} columns (excluding {len(exclude_columns)} protected columns)")
        else:
            excluded_data = None
            df_to_clean = df.copy()
            print(f"📊 Cleaning all {df_to_clean.shape[1]} columns")
        
        # Step 1: Analyze data quality (on all data including excluded columns)
        self.analyze_data_quality(df, quality_plots_dir)
        
        # Step 2: Filter high missing columns (optional) - only on cleanable columns
        if filter_missing_threshold is not None:
            df_to_clean = self.filter_high_missing_columns(df_to_clean, filter_missing_threshold)
        
        # Step 3: Impute missing values - only on cleanable columns
        df_to_clean = self.impute_missing_values(df_to_clean, imputation_method)
        
        # Step 4: Apply transformations - only on cleanable columns
        df_to_clean = self.normalize_and_transform(df_to_clean, normalization, log_transform, autonorm)
        
        # Step 5: Recombine with excluded columns
        if excluded_data is not None:
            # Ensure the indices match
            if len(excluded_data) == len(df_to_clean):
                df_final = pd.concat([excluded_data.reset_index(drop=True), 
                                    df_to_clean.reset_index(drop=True)], axis=1)
                print(f"✅ Recombined cleaned data with {len(exclude_columns)} protected columns")
            else:
                print(f"⚠️ Warning: Row count mismatch after cleaning. Excluded columns may not align properly.")
                df_final = df_to_clean
        else:
            df_final = df_to_clean
        
        # Step 6: Save cleaned data (optional)
        if output_path:
            self.save_cleaned_data(df_final, output_path)
        
        print("=" * 50)
        print("🎉 Data cleaning pipeline completed successfully!")
        
        return df_final


# Example usage and utility functions
def example_usage():
    """Example of how to use the DataCleaner class."""
    
    # Create sample data with missing values
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    data = np.random.randn(n_samples, n_features)
    
    # Introduce missing values
    missing_mask = np.random.random((n_samples, n_features)) < 0.1
    data[missing_mask] = np.nan
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    
    print("📝 Example: Data Cleaning Pipeline")
    print("=" * 40)
    
    # Initialize cleaner
    cleaner = DataCleaner(random_state=42)
    
    # Run complete pipeline
    cleaned_df = cleaner.clean_pipeline(
        df=df,
        imputation_method='knn',
        filter_missing_threshold=0.2,
        normalization='standard',
        log_transform=True,
        autonorm=True,
        output_path='cleaned_data.csv',
        quality_plots_dir='quality_plots'
    )
    
    print(f"\n📊 Original shape: {df.shape}")
    print(f"📊 Cleaned shape: {cleaned_df.shape}")
    print(f"📊 Missing values before: {df.isnull().sum().sum()}")
    print(f"📊 Missing values after: {cleaned_df.isnull().sum().sum()}")


def main():
    """Command line interface for data cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Cleaning Pipeline for Tabular Biomarker Analysis"
    )
    
    # Required arguments
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to input data file (CSV, Excel, or Parquet)'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=False,
        default=None,
        help='Path to save cleaned data (optional, auto-generated if not specified)'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--imputation_method', 
        type=str, 
        default='knn',
        choices=['knn', 'bpca', 'ppca', 'svd', 'cartpmm', 'lasso_norm'],
        help='Missing value imputation method (default: knn)'
    )
    
    parser.add_argument(
        '--filter_threshold', 
        type=float, 
        default=0.2,
        help='Filter columns with missing percentage above this threshold (default: 0.2 = 20%%)'
    )
    
    parser.add_argument(
        '--normalization', 
        type=str, 
        default='standard',
        choices=['standard', 'minmax', 'robust'],
        help='Normalization method (default: standard)'
    )
    
    parser.add_argument(
        '--no_log_transform', 
        action='store_true',
        help='Skip log2 transformation (default: apply log2 transform)'
    )
    
    parser.add_argument(
        '--no_autonorm', 
        action='store_true',
        help='Skip autonorm transformation (default: apply autonorm)'
    )
    
    parser.add_argument(
        '--quality_plots_dir', 
        type=str, 
        default=None,
        help='Directory to save data quality plots'
    )
    
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--id_column', 
        type=str, 
        default=None,
        help='Name of ID column to exclude from cleaning (optional)'
    )
    
    parser.add_argument(
        '--target_column', 
        type=str, 
        default=None,
        help='Name of target column to exclude from cleaning (optional)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from: {args.data_path}")
    if args.data_path.endswith('.csv'):
        df = pd.read_csv(args.data_path)
    elif args.data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(args.data_path)
    elif args.data_path.endswith('.parquet'):
        df = pd.read_parquet(args.data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.data_path}")
    
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Generate output path automatically
    from pathlib import Path
    from ..utils.paths import get_data_cleaning_dir
    
    input_path = Path(args.data_path)
    original_filename = input_path.stem  # 获取不带扩展名的文件名
    
    # 使用统一的路径管理获取输出目录
    output_dir = Path(get_data_cleaning_dir())
    
    # 生成输出文件名: [原始文件名]_[插补方法]_cleaned.csv
    output_filename = f"{original_filename}_{args.imputation_method}_cleaned.csv"
    auto_output_path = output_dir / output_filename
    
    # 如果用户指定了output_path，使用用户指定的；否则使用自动生成的
    final_output_path = args.output_path if args.output_path else str(auto_output_path)
    
    print(f"📁 Output will be saved to: {final_output_path}")
    
    # Initialize cleaner
    cleaner = DataCleaner(random_state=args.random_state)
    
    # Run pipeline
    cleaned_df = cleaner.clean_pipeline(
        df=df,
        imputation_method=args.imputation_method,
        filter_missing_threshold=args.filter_threshold,
        normalization=args.normalization,
        log_transform=not args.no_log_transform,
        autonorm=not args.no_autonorm,
        output_path=final_output_path,
        quality_plots_dir=args.quality_plots_dir,
        id_column=args.id_column,
        target_column=args.target_column
    )
    
    print(f"🎉 Cleaning completed! Output saved to: {final_output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command line mode
        main()
    else:
        # Example mode
        example_usage()
