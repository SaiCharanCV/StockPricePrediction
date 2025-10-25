"""Main script for stock price prediction."""
import logging
from utils.logger import setup_logging
from data.data_loader import DataLoader
from data.data_preprocessing import DataPreprocessor
from models.model_loader import ModelLoader

def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info("Loading model...")
        model_loader = ModelLoader()
        
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader()
        raw_data = data_loader.load_data()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        df = preprocessor.preprocess_data(raw_data, model_loader.label_encoder)
        df = preprocessor.encode_stock_names(df, model_loader.label_encoder)
        df = preprocessor.select_features(df)
        df = preprocessor.remove_outliers(df, model_loader.iso)
        df_scaled = preprocessor.scale_features(df, model_loader.scaler)
        
        logger.info(f"Final preprocessed data shape: {df_scaled.shape}")
        logger.info(f"Final preprocessed columns: {df_scaled.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()