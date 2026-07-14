import os

import matplotlib
matplotlib.use("Agg")  # Render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true, y_pred, epoch, save_dir,
                     ylabel="Normalized price", suffix=""):
    """
    Plot true vs predicted values for stock price prediction

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epoch: Current epoch number
        save_dir: Directory to save the plot
        ylabel: Y axis label
        suffix: Optional suffix appended to the file names
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True', marker='o', markersize=3)
    plt.plot(y_pred, label='Predicted', marker='x', markersize=3)
    plt.title(f'Stock Price Prediction - Epoch {epoch + 1} (validation set)')
    plt.xlabel('Validation sample')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/predictions_epoch_{epoch + 1}{suffix}.png', dpi=150)
    plt.close()

    # Also plot the prediction error
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(y_true - y_pred))
    plt.title(f'Prediction Error - Epoch {epoch + 1}')
    plt.xlabel('Validation sample')
    plt.ylabel('Absolute Error')
    plt.grid(alpha=0.3)
    plt.savefig(f'{save_dir}/error_epoch_{epoch + 1}{suffix}.png', dpi=150)
    plt.close()

    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"mse": mse}


def plot_loss_curves(train_history, val_history, save_dir, metric_name="loss"):
    """Plot training and validation metric curves across epochs"""
    epochs = np.arange(1, len(train_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history, label=f'Train {metric_name}', marker='o')
    plt.plot(epochs, val_history, label=f'Validation {metric_name}', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
    plt.title('Training Progress')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/loss_curves.png', dpi=150)
    plt.close()


def plot_spike_raster(spike_array, save_dir, title="Input spike raster",
                      filename="input_spike_raster.png"):
    """Plot a raster of one encoded sample.

    Args:
        spike_array: Binary array of shape (time_steps, num_neurons)
        save_dir: Directory to save the plot
    """
    spike_array = np.asarray(spike_array)
    times, neurons = np.nonzero(spike_array > 0)
    plt.figure(figsize=(10, 6))
    plt.scatter(times, neurons, s=4, marker='|')
    plt.xlabel('Timestep')
    plt.ylabel('Input neuron')
    plt.title(title)
    plt.grid(alpha=0.2)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{filename}', dpi=150)
    plt.close()


def plot_weight_changes(weights_history, save_dir, epoch):
    """
    Plot weight changes over training

    Args:
        weights_history: Dictionary mapping layer names to lists of weight changes
        save_dir: Directory to save the plot
        epoch: Current epoch number
    """
    plt.figure(figsize=(10, 6))
    for layer_name, changes in weights_history.items():
        plt.plot(changes, label=layer_name)
    plt.title(f'Weight Changes During Training - Up to Epoch {epoch + 1}')
    plt.xlabel('Batch')
    plt.ylabel('Mean Weight Change')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/weight_changes_epoch_{epoch + 1}.png', dpi=150)
    plt.close()


def plot_weight_distributions(weights_data, save_dir, epoch):
    """Plot histograms of weight distributions to diagnose learning issues"""
    plt.figure(figsize=(12, 4 * len(weights_data)))

    for i, (name, weights) in enumerate(weights_data.items()):
        plt.subplot(len(weights_data), 1, i + 1)
        flat_weights = np.asarray(weights).flatten()

        plt.hist(flat_weights, bins=50, alpha=0.7)
        plt.title(f'Weight Distribution - {name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')

        stats_text = (f"Mean: {np.mean(flat_weights):.4f}\n"
                      f"Std: {np.std(flat_weights):.4f}\n"
                      f"Min: {np.min(flat_weights):.4f}\n"
                      f"Max: {np.max(flat_weights):.4f}")
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/weight_distributions_epoch_{epoch + 1}.png', dpi=150)
    plt.close()
