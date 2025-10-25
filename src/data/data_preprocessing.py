"""Data preprocessing module for stock price prediction."""
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml
from utils.monitoring import log_execution_time, timer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class for preprocessing stock price data.
    
    Attributes:
        config (dict): Configuration parameters
        time_span (List[int]): Time spans for feature engineering
        feature_cols (List[str]): Columns to be used as features
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.time_span = self.config['preprocessing']['time_span']
        self.feature_cols = self.config['preprocessing']['feature_cols']
    
    @log_execution_time
    def preprocess_data(self, data: pd.DataFrame, le: Any) -> pd.DataFrame:
        """
        Preprocess the input data with feature engineering.
        
        Args:
            data: Input DataFrame
            le: Label encoder for stock names
            
        Returns:
            Preprocessed DataFrame
        """
        with timer("Data preprocessing"):
            db = data.copy()
            classes = le.classes_
            
            # Feature engineering
            self._create_time_features(db)
            self._create_price_features(db)
            
            # Filter stocks
            db = db[db['Stock_Name'].isin(classes)]
            logger.info("Data preprocessing completed")
            
            return db
    
    def _create_time_features(self, db: pd.DataFrame) -> None:
        """Create time-based features."""
        for i in self.time_span:
            with timer(f"Creating time features for span {i}"):
                db[f"lag_{i}"] = db.groupby('Stock_Name')['Close'].shift(i)
                db[f"rolling_mean_{i}"] = (
                    db.groupby('Stock_Name')['Close']
                    .rolling(window=i)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                if i != 1:
                    db[f"rolling_std_{i}"] = (
                        db.groupby('Stock_Name')['Close']
                        .rolling(window=i)
                        .std()
                        .reset_index(level=0, drop=True)
                    )
    
    def _create_price_features(self, db: pd.DataFrame) -> None:
        """Create price-based features."""
        with timer("Creating price features"):
            db['Price_Change'] = db.groupby('Stock_Name')['Close'].diff()
            db['Price_Change_Percentage'] = db.groupby('Stock_Name')['Close'].pct_change()
            db['Price_Trend'] = db['Price_Change'].apply(lambda x: 1 if x > 0 else 0)
            db['Date'] = db.index
            
            for lag in [1, 3, 7]:
                db[f'price_change_lag{lag}'] = (
                    db.groupby('Stock_Name')['Price_Change'].shift(lag)
                )
    
    @log_execution_time
    def encode_stock_names(self, db: pd.DataFrame, le: Any) -> pd.DataFrame:
        """Encode stock names using the label encoder."""
        db['Stock_encoded'] = le.transform(db['Stock_Name'])
        logger.info("Stock name encoding completed")
        return db
    
    @log_execution_time
    def select_features(self, db: pd.DataFrame) -> pd.DataFrame:
        """Select the specified feature columns."""
        logger.info(f"Selecting features: {self.feature_cols}")
        return db[self.feature_cols]
    
    @log_execution_time
    def remove_outliers(self, db: pd.DataFrame, iso: Any) -> pd.DataFrame:
        """Remove outliers using Isolation Forest."""
        with timer("Outlier removal"):
            mask = iso.predict(db) == 1
            n_outliers = (~mask).sum()
            logger.info(f"Removed {n_outliers} outliers")
            return db[mask]
    
    @log_execution_time
    def scale_features(self, db: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """Scale the features using the provided scaler."""
        with timer("Feature scaling"):
            # Store column names and index for reconstruction
            columns = db.columns
            index = db.index
            
            # Scale the data
            scaled_data = scaler.transform(db)
            
            # Reconstruct DataFrame with original columns and index
            scaled_df = pd.DataFrame(scaled_data, columns=columns, index=index)
            logger.info("Feature scaling completed")
            
            return scaled_df