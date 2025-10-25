"""Unit tests for data loader module."""
import pytest
import pandas as pd
from src.data.data_loader import DataLoader

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Stock_Name': ['AAPL', 'GOOGL'],
        'Close': [100, 200],
        'Date': ['2023-01-01', '2023-01-01']
    })

def test_data_loader_initialization():
    """Test DataLoader initialization."""
    loader = DataLoader()
    assert loader.df is None
    assert loader.dataset_path.name == "Stock Price.csv"

def test_data_validation_missing_columns(sample_data):
    """Test data validation with missing columns."""
    loader = DataLoader()
    sample_data = sample_data.drop('Close', axis=1)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.df = sample_data
        loader.validate_data()