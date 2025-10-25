"""Unit tests for data preprocessing module."""
import pytest
import pandas as pd
import numpy as np
from src.data.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Stock_Name': ['AAPL', 'GOOGL'] * 10,
        'Close': np.random.randn(20),
        'Date': pd.date_range(start='2023-01-01', periods=20)
    })

@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance."""
    return DataPreprocessor()

def test_feature_engineering(preprocessor, sample_data):
    """Test feature engineering process."""
    result = preprocessor.preprocess_data(sample_data)
    
    # Check created features
    assert 'rolling_mean_1' in result.columns
    assert 'Price_Change' in result.columns
    assert 'Price_Trend' in result.columns

def test_outlier_removal(preprocessor, sample_data):
    """Test outlier removal process."""
    # Add some outliers
    sample_data.loc[0, 'Close'] = 1000000
    
    processed = preprocessor.preprocess_data(sample_data)
    result = preprocessor.remove_outliers(processed)
    
    assert len(result) < len(processed)