"""Data loading module for stock price prediction."""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
from utils.validation import validate_dataframe
from utils.monitoring import log_execution_time

logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class for loading and validating stock price data.
    
    Attributes:
        config (dict): Configuration parameters
        dataset_path (Path): Path to the dataset file
        df (Optional[pd.DataFrame]): Loaded DataFrame
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the DataLoader.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = Path(self.config['data']['input_file'])
        self.df: Optional[pd.DataFrame] = None
        
    @log_execution_time
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the dataset.
        
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If data validation fails
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path)
        
        # Validate the data
        validation_results = validate_dataframe(
            self.df,
            self.config['preprocessing']['validation']['required_columns'],
            self.config['preprocessing']['validation']['max_missing_threshold']
        )
        
        if not validation_results['is_valid']:
            raise ValueError(validation_results['error_message'])
            
        logger.info(f"Dataset loaded successfully with shape: {self.df.shape}")
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded DataFrame. Loads it first if not already loaded.
        
        Returns:
            The loaded DataFrame
        """
        if self.df is None:
            self.load_data()
        return self.df