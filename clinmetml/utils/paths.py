"""
Path configuration and management utilities for ClinMetML.

This module provides centralized path management to ensure consistent
output directory structure across all ClinMetML components.
"""

import os
from pathlib import Path
from typing import Optional, Union


class ClinMetMLPathManager:
    """
    Centralized path management for ClinMetML outputs.
    
    This class manages all output directories and ensures consistent
    path structure across different components.
    """
    
    def __init__(self, base_output_dir: Union[str, Path] = "clinmetml_outputs"):
        """
        Initialize path manager.
        
        Parameters
        ----------
        base_output_dir : str or Path
            Base directory for all ClinMetML outputs
        """
        self.base_dir = Path(base_output_dir).resolve()
        self._ensure_base_dir()
    
    def _ensure_base_dir(self):
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def data_cleaning_dir(self) -> Path:
        """Get data cleaning output directory."""
        return self.base_dir / "data_cleaning"
    
    @property
    def feature_selection_dir(self) -> Path:
        """Get feature selection output directory."""
        return self.base_dir / "feature_selection_outputs"
    
    @property
    def multicollinearity_dir(self) -> Path:
        """Get multicollinearity reduction output directory."""
        return self.base_dir / "reduce_multicollinearity_outputs"
    
    @property
    def rfe_dir(self) -> Path:
        """Get RFE selection output directory."""
        return self.base_dir / "rfe_outputs"
    
    @property
    def model_training_dir(self) -> Path:
        """Get model training output directory."""
        return self.base_dir / "model_outputs"
    
    @property
    def interpretability_dir(self) -> Path:
        """Get interpretability analysis output directory."""
        return self.base_dir / "interpretability_outputs"
    
    @property
    def processed_data_dir(self) -> Path:
        """Get processed data output directory."""
        return self.base_dir / "processed_outputs"
    
    def get_feature_selection_subdir(self, subdir: str) -> Path:
        """
        Get feature selection subdirectory.
        
        Parameters
        ----------
        subdir : str
            Subdirectory name (e.g., 'resampling', 'feature_selection', 'analysis')
        """
        return self.feature_selection_dir / subdir
    
    def ensure_dir(self, directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists and return Path object.
        
        Parameters
        ----------
        directory : str or Path
            Directory path to create
            
        Returns
        -------
        Path
            Path object for the directory
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_relative_path(self, target_dir: Union[str, Path]) -> str:
        """
        Get relative path from current working directory to target.
        
        Parameters
        ----------
        target_dir : str or Path
            Target directory
            
        Returns
        -------
        str
            Relative path string
        """
        target_path = Path(target_dir)
        try:
            return str(target_path.relative_to(Path.cwd()))
        except ValueError:
            # If relative path cannot be computed, return absolute path
            return str(target_path.resolve())
    
    def create_all_dirs(self):
        """Create all standard ClinMetML output directories."""
        dirs_to_create = [
            self.data_cleaning_dir,
            self.feature_selection_dir,
            self.multicollinearity_dir,
            self.rfe_dir,
            self.model_training_dir,
            self.interpretability_dir,
            self.processed_data_dir,
        ]
        
        for directory in dirs_to_create:
            self.ensure_dir(directory)
        
        # Create feature selection subdirectories
        feature_subdirs = ['resampling', 'feature_selection', 'analysis']
        for subdir in feature_subdirs:
            self.ensure_dir(self.get_feature_selection_subdir(subdir))


# Global path manager instance
_default_path_manager = None


def get_default_path_manager() -> ClinMetMLPathManager:
    """Get the default global path manager instance."""
    global _default_path_manager
    if _default_path_manager is None:
        _default_path_manager = ClinMetMLPathManager()
    return _default_path_manager


def set_default_base_dir(base_dir: Union[str, Path]):
    """Set the default base directory for all ClinMetML outputs."""
    global _default_path_manager
    _default_path_manager = ClinMetMLPathManager(base_dir)


def get_output_dir(component: str, base_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Get output directory for a specific component.
    
    Parameters
    ----------
    component : str
        Component name ('data_cleaning', 'feature_selection', 'multicollinearity', 
        'rfe', 'model_training', 'interpretability')
    base_dir : str or Path, optional
        Base directory. If None, uses default path manager.
        
    Returns
    -------
    str
        Output directory path as string
    """
    if base_dir is not None:
        path_manager = ClinMetMLPathManager(base_dir)
    else:
        path_manager = get_default_path_manager()
    
    component_map = {
        'data_cleaning': path_manager.data_cleaning_dir,
        'feature_selection': path_manager.feature_selection_dir,
        'multicollinearity': path_manager.multicollinearity_dir,
        'rfe': path_manager.rfe_dir,
        'model_training': path_manager.model_training_dir,
        'interpretability': path_manager.interpretability_dir,
        'processed_data': path_manager.processed_data_dir,
    }
    
    if component not in component_map:
        raise ValueError(f"Unknown component: {component}. "
                        f"Available components: {list(component_map.keys())}")
    
    output_dir = component_map[component]
    path_manager.ensure_dir(output_dir)
    return str(output_dir)


def get_feature_selection_subdir(subdir: str, base_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Get feature selection subdirectory.
    
    Parameters
    ----------
    subdir : str
        Subdirectory name
    base_dir : str or Path, optional
        Base directory. If None, uses default path manager.
        
    Returns
    -------
    str
        Subdirectory path as string
    """
    if base_dir is not None:
        path_manager = ClinMetMLPathManager(base_dir)
    else:
        path_manager = get_default_path_manager()
    
    output_dir = path_manager.get_feature_selection_subdir(subdir)
    path_manager.ensure_dir(output_dir)
    return str(output_dir)


# Convenience functions for backward compatibility
def get_data_cleaning_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get data cleaning output directory."""
    return get_output_dir('data_cleaning', base_dir)


def get_feature_selection_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get feature selection output directory."""
    return get_output_dir('feature_selection', base_dir)


def get_multicollinearity_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get multicollinearity reduction output directory."""
    return get_output_dir('multicollinearity', base_dir)


def get_rfe_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get RFE selection output directory."""
    return get_output_dir('rfe', base_dir)


def get_model_training_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get model training output directory."""
    return get_output_dir('model_training', base_dir)


def get_interpretability_dir(base_dir: Optional[Union[str, Path]] = None) -> str:
    """Get interpretability analysis output directory."""
    return get_output_dir('interpretability', base_dir)
