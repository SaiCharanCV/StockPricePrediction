# Stock Price Predictor Backend

A machine learning backend for stock price prediction.

## Project Structure

```
Stock Price Backend/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model management
│   └── utils/          # Utilities
├── tests/              # Unit tests
└── logs/               # Log files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py
```

## Testing

Run tests:
```bash
pytest tests/
```

## Configuration

Edit `config/config.yaml` to modify:
- Data paths
- Preprocessing parameters
- Logging settings

## Project Components

1. **Data Loading** (`src/data/data_loader.py`):
   - CSV file loading
   - Data validation
   - Error handling

2. **Data Preprocessing** (`src/data/data_preprocessing.py`):
   - Feature engineering
   - Outlier removal
   - Data scaling

3. **Model Management** (`src/models/model_loader.py`):
   - Model loading
   - Component validation
   - Error handling

4. **Utilities** (`src/utils/`):
   - Logging configuration
   - Performance monitoring
   - Data validation

## Contributing

1. Create a feature branch
2. Make changes
3. Run tests
4. Submit pull request