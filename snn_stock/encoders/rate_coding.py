# snn_stock/encoders/rate_coding.py

import numpy as np


class RateEncoder:
    """Rate coding: each input value is assigned one input neuron whose
    number of spikes within the encoding window is proportional to the value.

    Input:  a (sequence_length, n_features) window of values in [0, 1]
    Output: a (n_steps, sequence_length * n_features) binary spike array
    """

    def __init__(self, n_steps=100, max_rate=100.0, dt=1.0):
        """
        :param n_steps: Number of timesteps in spike train
        :param max_rate: Maximum spike rate (Hz) - caps spikes per window
        :param dt: Time step size (ms)
        """
        if n_steps <= 0 or max_rate <= 0 or dt <= 0:
            raise ValueError("n_steps, max_rate, and dt must be positive")

        self.n_steps = n_steps
        self.max_rate = max_rate
        self.dt = dt

    def encode(self, data):
        # Ensure data is 2D: (sequence_length, n_features)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Values are expected to be normalized already (dataset loader).
        # If they fall outside [0, 1], rescale the window as a fallback.
        if data.min() < 0.0 or data.max() > 1.0:
            d_min, d_max = data.min(), data.max()
            data = (data - d_min) / (d_max - d_min + 1e-8)
        values = np.clip(data.flatten(), 0.0, 1.0)  # one neuron per value

        num_neurons = values.size
        spikes = np.zeros((self.n_steps, num_neurons), dtype=np.float32)

        # Spikes per window: proportional to value, capped by max_rate
        max_spikes = min(self.n_steps,
                         int(round(self.max_rate * self.dt * self.n_steps / 1000.0)))
        if max_spikes < 1:
            max_spikes = self.n_steps  # max_rate too low to matter at this dt

        for n, v in enumerate(values):
            n_spikes = int(round(v * max_spikes))
            if n_spikes > 0:
                # Evenly space the spikes across the window (deterministic)
                positions = np.linspace(0, self.n_steps - 1, n_spikes).astype(int)
                spikes[positions, n] = 1.0

        return spikes
