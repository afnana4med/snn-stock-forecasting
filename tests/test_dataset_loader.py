import unittest
import numpy as np
from snn_stock.data.dataset_loader import PriceDataset

class TestPriceDataset(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        data_file = Path(__file__).resolve().parent.parent / \
            "data/processed/test_AAPL_1min.csv"
        self.dataset = PriceDataset(
            file_paths=[str(data_file)],
            sequence_length=60,
            prediction_horizon=1
        )

    def test_dataset_loading(self):
        self.assertGreater(len(self.dataset), 0)
        
    def test_sample_shape(self):
        X, y = self.dataset[0]
        self.assertEqual(X.shape, (60, len(self.dataset.features)))
        self.assertTrue(isinstance(y, np.ndarray))