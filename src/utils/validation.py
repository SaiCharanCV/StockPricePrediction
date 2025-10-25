"""Data validation utilities."""
import pandas as pd
from typing import List, Dict, Any

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    max_missing_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Validate a DataFrame against specified requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        max_missing_threshold: Maximum allowed proportion of missing values
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        ValueError: If validation fails
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'high_missing_cols': [],
        'error_message': None
    }
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['missing_columns'] = list(missing_cols)
        validation_results['error_message'] = f"Missing required columns: {missing_cols}"
        return validation_results
    
    # Check missing values
    missing_proportions = df[required_columns].isnull().mean()
    high_missing = missing_proportions[missing_proportions > max_missing_threshold]
    
    if not high_missing.empty:
        validation_results['is_valid'] = False
        validation_results['high_missing_cols'] = list(high_missing.index)
        validation_results['error_message'] = (
            f"Columns exceeding missing value threshold: {list(high_missing.index)}"
        )
    
    return validation_results