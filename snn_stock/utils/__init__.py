from typing import List

import numpy as np

from ml_genn.utils.data import PreprocessedSpikes, preprocess_spikes


def convert_rate_to_spike_times(encoded_spikes, dt=1.0) -> List[PreprocessedSpikes]:
    """
    Converts binary spike arrays to ml_genn's PreprocessedSpikes format.

    Args:
        encoded_spikes: Binary array of shape (batch, time_steps, num_neurons)
                        (a single (time_steps, num_neurons) sample also works)
        dt: Simulation timestep in ms; spike at index t occurs at time t * dt

    Returns:
        List of PreprocessedSpikes objects, one per sample in the batch
    """
    encoded_spikes = np.asarray(encoded_spikes)
    if encoded_spikes.ndim == 2:
        encoded_spikes = encoded_spikes[np.newaxis]
    if encoded_spikes.ndim != 3:
        raise ValueError(f"Expected (batch, time_steps, num_neurons) spike "
                         f"array, got shape {encoded_spikes.shape}")

    num_neurons = encoded_spikes.shape[2]
    processed_spikes = []
    for sample in encoded_spikes:
        time_idx, neuron_ids = np.nonzero(sample > 0)
        processed_spikes.append(
            preprocess_spikes(time_idx.astype(np.float64) * dt,
                              neuron_ids, num_neurons))

    return processed_spikes
