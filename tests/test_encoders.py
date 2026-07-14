import unittest
import numpy as np
from snn_stock.encoders.rate_coding import RateEncoder
from snn_stock.encoders.temporal_coding import TemporalEncoder

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.rate_encoder = RateEncoder(n_steps=100)
        self.temporal_encoder = TemporalEncoder(n_steps=100)
        self.test_input = np.random.rand(10, 5)  # 10 timesteps, 5 features
        
    def test_rate_encoder(self):
        spikes = self.rate_encoder.encode(self.test_input)
        self.assertEqual(spikes.shape, (100, 50))  # n_steps × (T×F)
        self.assertTrue(np.all((spikes == 0) | (spikes == 1)))
        
    def test_temporal_encoder(self):
        spikes = self.temporal_encoder.encode(self.test_input)
        self.assertEqual(spikes.shape, (100, 50))
        self.assertEqual(np.sum(spikes, axis=0).max(), 1)  # One spike per input

    # Add to both encoder classes
    def visualize_spikes(self, spikes, title=None):
        """Visualize spike train"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.imshow(spikes.T, aspect='auto', cmap='binary')
        plt.xlabel('Time step')
        plt.ylabel('Neuron index')
        if title:
            plt.title(title)
        plt.colorbar(label='Spike')
        plt.show()