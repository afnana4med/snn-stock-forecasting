# snn_stock/encoders/temporal_contrast.py

import numpy as np


class TemporalContrastEncoder:
    """Temporal contrast (threshold) coding: each feature gets an ON and an
    OFF neuron. A spike is emitted when the step-to-step change of the
    feature exceeds an adaptive threshold - ON for upward moves, OFF for
    downward moves. Steps with small changes produce no spikes, so the code
    is sparse and highlights fluctuations rather than levels.

    Input:  a (sequence_length, n_features) window
    Output: a (n_steps, 2 * n_features) binary spike array
    """

    def __init__(self, n_steps=100, threshold_scale=0.5):
        if n_steps <= 1:
            raise ValueError("n_steps must be greater than 1")
        self.n_steps = n_steps
        # Threshold as a fraction of the per-window std of the changes
        self.threshold_scale = threshold_scale

    def encode(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        seq_len, n_features = data.shape
        diffs = np.diff(data, axis=0)  # (seq_len - 1, n_features)

        spikes = np.zeros((self.n_steps, 2 * n_features), dtype=np.float32)
        if seq_len < 2:
            return spikes

        # Map each of the seq_len-1 change steps onto a spike-train time bin
        bins = np.round(np.linspace(0, self.n_steps - 1,
                                    diffs.shape[0])).astype(int)

        for j in range(n_features):
            col = diffs[:, j]
            thresh = self.threshold_scale * (np.std(col) + 1e-12)
            on_steps = np.nonzero(col > thresh)[0]
            off_steps = np.nonzero(col < -thresh)[0]
            spikes[bins[on_steps], 2 * j] = 1.0        # ON neuron
            spikes[bins[off_steps], 2 * j + 1] = 1.0   # OFF neuron

        return spikes
