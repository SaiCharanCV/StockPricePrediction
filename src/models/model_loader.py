"""Model loading and management module."""
import pickle
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import yaml
from utils.monitoring import log_execution_time

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

class ModelLoader:
    """
    A class for loading and managing machine learning models.
    
    Attributes:
        config (dict): Configuration parameters
        model_path (Path): Path to the model file
        model (Any): The main prediction model
        label_encoder (Any): Label encoder for categorical variables
        scaler (Any): Feature scaler
        target_scaler (Any): Target variable scaler
        iso (Any): Isolation Forest model
        load_timestamp (datetime): When the model was loaded
    """
    
    REQUIRED_COMPONENTS = {
        'Model': 'Main prediction model',
        'Label_Encoder': 'Label encoder for categorical variables',
        'Scaler_X': 'Feature scaler',
        'Scaler_Y': 'Target variable scaler',
        'Isolation_Forest': 'Outlier detection model'
    }
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the ModelLoader.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_path = Path(self.config['data']['model_file'])
        self.load_timestamp = None
        self._load_model()
    
    def _validate_model_components(self, model_dict: Dict[str, Any]) -> None:
        """
        Validate that all required model components are present.
        
        Args:
            model_dict: Dictionary containing model components
            
        Raises:
            ModelLoadError: If any required component is missing
        """
        missing_components = []
        for component in self.REQUIRED_COMPONENTS:
            if component not in model_dict:
                missing_components.append(component)
        
        if missing_components:
            raise ModelLoadError(
                f"Missing required model components: {', '.join(missing_components)}"
            )
    
    @log_execution_time
    def _load_model(self) -> None:
        """
        Load the model and its components from the pickle file.
        
        Raises:
            ModelLoadError: If there's any error during model loading
        """
        try:
            if not self.model_path.exists():
                raise ModelLoadError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                model_dict = pickle.load(f)
            
            # Validate model components
            self._validate_model_components(model_dict)
            
            # Load components
            self.model = model_dict['Model']
            self.label_encoder = model_dict['Label_Encoder']
            self.scaler = model_dict['Scaler_X']
            self.target_scaler = model_dict['Scaler_Y']
            self.iso = model_dict['Isolation_Forest']
            
            self.load_timestamp = datetime.now()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': str(self.model_path),
            'load_timestamp': str(self.load_timestamp),
            'components': list(self.REQUIRED_COMPONENTS.keys())
        }