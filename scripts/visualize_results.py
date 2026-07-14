import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def plot_training_curves(results, save_path=None):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(results['train_losses'], label='Training Loss')
    plt.plot(results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_predictions(results, save_path=None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(results['y_true'], label='Actual', alpha=0.7)
    plt.plot(results['y_pred'], label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Prediction Results')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()