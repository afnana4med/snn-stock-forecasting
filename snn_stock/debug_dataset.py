import logging
import sys
from pathlib import Path
from snn_stock.data.dataset_loader import PriceDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def debug_dataset():
    """Debug the dataset loading and processing."""
    try:
        # First, check if the test file exists
        test_file = str(Path(__file__).resolve().parent.parent /
                        "data/processed/test_AAPL_1min.csv")
        if not Path(test_file).exists():
            logging.error(f"Test file not found: {test_file}")
            logging.info("Please run create_test_dataset.py first:")
            logging.info("python scripts/create_test_dataset.py")
            return

        logging.info("Loading test dataset...")
        
        # Configuration for minimal test dataset
        dataset_config = {
            "file_paths": [test_file],
            "sequence_length": 10,
            "prediction_horizon": 1,
            "task": "regression",
            "features": ["Open", "High", "Low", "Close", "Volume"],
            "target_column": "Close",
            "normalize": True
        }

        # Try to load the dataset
        dataset = PriceDataset(**dataset_config)
        logging.info(f"✅ Dataset loaded successfully with {len(dataset)} samples")

        # Print dataset statistics
        logging.info("Dataset Statistics:")
        logging.info(f"Total samples: {len(dataset)}")
        logging.info(f"Features: {dataset.features}")
        logging.info(f"Sequence length: {dataset.sequence_length}")
        
        # Check first few samples
        for idx in range(min(3, len(dataset))):
            X, y = dataset[idx]
            logging.info(f"\nSample {idx + 1}:")
            logging.info(f"Input shape: {X.shape}")
            logging.info(f"Target value: {y}")
            
    except Exception as e:
        logging.error(f"Error during dataset debugging: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    debug_dataset()