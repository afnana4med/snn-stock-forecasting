# snn_stock/encoders/temporal_coding.py

import numpy as np


class TemporalEncoder:
    """Time-to-first-spike coding: each input value is assigned one input
    neuron which emits a single spike; larger values spike earlier.

    Input:  a (sequence_length, n_features) window of values in [0, 1]
    Output: a (n_steps, sequence_length * n_features) binary spike array
    """

    def __init__(self, n_steps=100):
        if n_steps <= 1:
            raise ValueError("n_steps must be greater than 1")
        self.n_steps = n_steps

    def encode(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Values are expected to be normalized already (dataset loader).
        # If they fall outside [0, 1], rescale the window as a fallback.
        if data.min() < 0.0 or data.max() > 1.0:
            d_min, d_max = data.min(), data.max()
            data = (data - d_min) / (d_max - d_min + 1e-8)
        values = np.clip(data.flatten(), 0.0, 1.0)

        num_neurons = values.size
        spikes = np.zeros((self.n_steps, num_neurons), dtype=np.float32)

        # Larger value -> earlier spike
        spike_times = np.round((1.0 - values) * (self.n_steps - 1)).astype(int)
        spikes[spike_times, np.arange(num_neurons)] = 1.0

        return spikes
