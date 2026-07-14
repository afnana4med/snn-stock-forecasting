# snn_stock/main.py

import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
from snn_stock.training.trainer import run_training
import sys
import os

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config):
    """Setup logging configuration"""
    exp_name = config["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["logging"]["save_dir"]) / f"{exp_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    return log_dir

def check_cuda_availability():
    """Check whether the GeNN CUDA backend is available"""
    try:
        from pygenn import cuda_backend  # noqa: F401
        logging.info("GeNN CUDA backend is available. Using GPU.")
        return True
    except ImportError:
        logging.info("GeNN CUDA backend not available. Using CPU backend.")
        return False
    except Exception as e:
        logging.warning(f"Error checking CUDA: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rate_eprop.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    log_dir = setup_logging(config)

    # Set device
    config["device"] = "cuda" if check_cuda_availability() else "cpu"

    logging.info(f"Starting experiment: {config['experiment_name']} on {config['device']}")

    try:
        results = run_training(config)
        logging.info(f"Training completed. Results: {results}")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
