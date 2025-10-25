"""Logger configuration module."""
import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(config_path: str = 'config/config.yaml') -> None:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to the configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': config['logging']['format']
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'level': config['logging']['level'],
                'formatter': 'standard',
                'filename': config['logging']['file'],
                'mode': 'a',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'level': config['logging']['level'],
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': config['logging']['level'],
                'propagate': True
            }
        }
    }
    
    # Create logs directory if it doesn't exist
    Path(config['logging']['file']).parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.config.dictConfig(logging_config)