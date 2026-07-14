import json
import logging
from pathlib import Path

import numpy as np

from ml_genn.callbacks import Checkpoint, OptimiserParamSchedule
from ml_genn.compilers import EPropCompiler, EventPropCompiler, InferenceCompiler
from ml_genn.metrics import Metric
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy

from snn_stock.data.dataset_loader import PriceDataset
from snn_stock.encoders.rate_coding import RateEncoder
from snn_stock.encoders.temporal_coding import TemporalEncoder
from snn_stock.models.snn_model import build_snn
from snn_stock.utils import convert_rate_to_spike_times
from snn_stock.utils.visualization import (plot_loss_curves, plot_predictions,
                                           plot_spike_raster,
                                           plot_weight_distributions)


class LastTimestepMSE(Metric):
    """MSE between the readout at the end of each example and the target.

    The MeanSquareError *loss* trains against a per-timestep target of shape
    (batch, timesteps, outputs), but the network readout returns one value per
    example, so the built-in MSE metric cannot be used directly.
    """

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred, communicator=None):
        y_true = np.asarray(y_true)
        if y_true.ndim == 3:
            y_true = y_true[:, -1, :]
        y_pred = np.asarray(y_pred)[:len(y_true)]
        self.sum_se += float(np.sum(np.square(y_true - y_pred)))
        self.total += y_true.size

    def reset(self):
        self.sum_se = 0.0
        self.total = 0

    @property
    def result(self):
        return None if self.total == 0 else self.sum_se / self.total


def get_encoder(config):
    """Returns the spike encoder based on configuration."""
    encoding_type = config["data"]["encoding"]
    n_steps = config.get("n_steps", 100)

    if encoding_type == "rate":
        # Default max_rate of 1000 Hz means one spike per 1 ms timestep at
        # most, i.e. the full [0, 1] value range maps onto 0..n_steps spikes
        max_rate = config["data"].get("max_rate", 1000.0)
        return RateEncoder(n_steps=n_steps, max_rate=max_rate,
                           dt=config.get("dt", 1.0))
    elif encoding_type == "temporal":
        return TemporalEncoder(n_steps=n_steps)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


def encode_dataset(dataset, encoder, max_samples=None):
    """Encodes input windows into spike trains of shape (N, n_steps, n_neurons)."""
    spike_sequences = []
    labels = []

    for i, (x, y) in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        spikes = encoder.encode(np.asarray(x))
        if i == 0:
            logging.info(f"Encoded spike shape per sample: {spikes.shape} "
                         f"({spikes.sum():.0f} spikes in first sample)")
        spike_sequences.append(spikes)
        labels.append(y)

    spike_sequences = np.stack(spike_sequences)  # (N, T, I)
    labels = np.asarray(labels)
    return spike_sequences, labels


def run_training(config):
    # Validate config
    required_keys = ["data", "model", "training", "algorithm", "task", "dt"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    task = config["task"]
    algorithm = config["algorithm"].lower()
    n_steps = config.get("n_steps", 100)
    dt = float(config["dt"])
    batch_size = int(config["training"]["batch_size"])
    backend = "cuda" if config.get("device", "cpu") == "cuda" else "single_threaded_cpu"
    if backend == "single_threaded_cpu" and batch_size != 1:
        logging.info("ℹ️ GeNN's single-threaded CPU backend only supports "
                     "batch size 1 - overriding configured batch size "
                     f"{batch_size}")
        batch_size = 1
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    seed = int(config["training"].get("random_seed", 42))
    np.random.seed(seed)

    save_dir = Path(config["logging"]["save_dir"]) / config["experiment_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("🚀 Starting training with configuration:")
    logging.info(config)

    # ------------------------------------------------------------------
    # Data loading and spike encoding
    # ------------------------------------------------------------------
    target_mode = config["data"].get("target_mode", "level")
    dataset = PriceDataset(
        file_paths=config["data"]["files"],
        sequence_length=config["data"]["sequence_length"],
        prediction_horizon=config["data"]["prediction_horizon"],
        task=task,
        features=config["data"]["features"],
        target_column=config["data"].get("target_column", "Close"),
        normalize=config["data"]["normalize"],
        target_mode=target_mode)
    logging.info(f"✅ Dataset loaded with {len(dataset)} samples.")

    encoder = get_encoder(config)
    max_samples = config.get("max_samples", None)
    logging.info(f"🔄 Encoding up to {max_samples or len(dataset)} samples "
                 f"with {encoder.__class__.__name__}...")
    spike_sequences, labels = encode_dataset(dataset, encoder, max_samples)
    logging.info(f"✅ Encoding complete. Spike array shape: {spike_sequences.shape}")

    # ------------------------------------------------------------------
    # Chronological train/validation split (trimmed to full batches,
    # because ml_genn's MSE loss requires complete batches)
    # ------------------------------------------------------------------
    val_split = float(config["training"].get("validation_split", 0.2))
    n_total = len(spike_sequences)
    train_size = (int(n_total * (1.0 - val_split)) // batch_size) * batch_size
    val_size = ((n_total - train_size) // batch_size) * batch_size
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Not enough samples ({n_total}) for batch size "
                         f"{batch_size} with validation split {val_split}")

    train_spikes = spike_sequences[:train_size]
    val_spikes = spike_sequences[train_size:train_size + val_size]

    if task == "classification":
        train_y = labels[:train_size].astype(np.int64).ravel()
        val_y = labels[train_size:train_size + val_size].astype(np.int64).ravel()
    else:
        y = labels.reshape(n_total, -1).astype(np.float32)
        # MeanSquareError loss trains the readout towards the target at
        # every timestep, so repeat the target across the example
        train_y = np.repeat(y[:train_size, np.newaxis, :], n_steps, axis=1)
        val_y = np.repeat(y[train_size:train_size + val_size, np.newaxis, :],
                          n_steps, axis=1)

    logging.info(f"📊 Training samples: {train_size}, Validation samples: {val_size}")

    # Convert binary spike arrays to ml_genn's PreprocessedSpikes format
    train_x = convert_rate_to_spike_times(train_spikes, dt=dt)
    val_x = convert_rate_to_spike_times(val_spikes, dt=dt)

    # ------------------------------------------------------------------
    # Build and compile the network
    # ------------------------------------------------------------------
    input_size = spike_sequences.shape[-1]
    output_size = int(config["model"]["output_neurons"])

    # EventProp classification requires an integrating readout such as
    # average membrane voltage; regression uses the plain voltage readout
    output_readout = config["model"].get("output_readout", "var")
    if algorithm == "eventprop" and task == "classification":
        output_readout = "avg_var"

    max_input_spikes = batch_size * input_size * n_steps
    net, input_pop, output_pop = build_snn(
        input_size=input_size,
        hidden_layers=config["model"]["hidden_layers"],
        output_size=output_size,
        neuron_params=config["model"]["neuron_params"],
        output_readout=output_readout,
        algorithm=algorithm,
        max_input_spikes=max_input_spikes)

    if task == "classification":
        loss = "sparse_categorical_crossentropy"
        metrics = "sparse_categorical_accuracy"
        metric_name = "accuracy"
    else:
        loss = "mean_square_error"
        metrics = LastTimestepMSE()
        metric_name = "mse"

    compiler_args = {
        "example_timesteps": n_steps,
        "losses": loss,
        "optimiser": Adam(learning_rate),
        "batch_size": batch_size,
        "dt": dt,
        "rng_seed": seed,
        "backend": backend,
    }

    logging.info(f"🧠 Compiling SNN using {algorithm.upper()} ({backend})...")
    if algorithm == "eprop":
        compiler = EPropCompiler(**compiler_args)
    elif algorithm == "eventprop":
        # Spike-count regularization keeps hidden neurons firing - EventProp
        # gradients can only flow through spikes
        compiler = EventPropCompiler(max_spikes=max(500, n_steps + 1),
                                     reg_lambda_upper=1e-8,
                                     reg_lambda_lower=1e-8,
                                     reg_nu_upper=10,
                                     **compiler_args)
    else:
        raise ValueError(f"Unsupported learning rule: {algorithm}")

    compiled_net = compiler.compile(net)

    # ------------------------------------------------------------------
    # Training loop (one ml_genn train call per epoch to record curves)
    # ------------------------------------------------------------------
    serialiser = Numpy(str(save_dir / "checkpoints"))
    train_history = []
    val_history = []

    logging.info(f"📈 Training for {epochs} epochs...")
    lr_decay = config["training"].get("lr_decay")
    lr_decay_every = int(config["training"].get("lr_decay_every", 10))

    def alpha_schedule(epoch, alpha):
        if lr_decay and epoch > 0 and epoch % lr_decay_every == 0:
            logging.info(f"⤵️ Decaying learning rate to {alpha * lr_decay:.6f}")
            return alpha * lr_decay
        return alpha

    with compiled_net:
        callbacks = [Checkpoint(serialiser)]
        if lr_decay:
            callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule))
        for epoch in range(epochs):
            train_metrics, val_metrics, _, _ = compiled_net.train(
                {input_pop: train_x}, {output_pop: train_y},
                num_epochs=1, start_epoch=epoch, shuffle=True,
                metrics=metrics,
                callbacks=callbacks, validation_callbacks=[],
                validation_x={input_pop: val_x},
                validation_y={output_pop: val_y})

            train_result = train_metrics[output_pop].result
            val_result = val_metrics[output_pop].result
            train_history.append(float(train_result))
            val_history.append(float(val_result))
            logging.info(f"✅ Epoch {epoch + 1}/{epochs} | "
                         f"Train {metric_name}: {train_result:.6f} | "
                         f"Val {metric_name}: {val_result:.6f}")

    # ------------------------------------------------------------------
    # Evaluation: reload the final checkpoint into an inference network
    # and generate real predictions on the validation set
    # ------------------------------------------------------------------
    logging.info("🔍 Compiling inference network from final checkpoint...")
    net.load((epochs - 1,), serialiser)
    inference_compiler = InferenceCompiler(evaluate_timesteps=n_steps,
                                           dt=dt, batch_size=batch_size,
                                           rng_seed=seed, backend=backend)
    inference_net = inference_compiler.compile(net)

    with inference_net:
        y_pred_dict, _ = inference_net.predict({input_pop: val_x}, output_pop,
                                               callbacks=[])
    y_pred = np.asarray(y_pred_dict[output_pop])

    results = {"train_history": train_history, "val_history": val_history,
               "metric": metric_name}

    if task == "classification":
        pred_class = np.argmax(y_pred, axis=1)
        accuracy = float(np.mean(pred_class == val_y))
        results["val_accuracy"] = accuracy
        # Compare against always predicting the majority class
        majority = np.bincount(val_y).argmax()
        results["majority_baseline_accuracy"] = float(np.mean(val_y == majority))
        logging.info(f"🎯 Validation accuracy: {accuracy:.4f} "
                     f"(majority-class baseline: "
                     f"{results['majority_baseline_accuracy']:.4f})")
        plot_predictions(val_y, pred_class, epochs - 1, save_dir,
                         ylabel="Price direction (0=down, 1=up)")
    else:
        y_true = val_y[:, -1, :].ravel()
        y_hat = y_pred.ravel()
        mse = float(np.mean(np.square(y_true - y_hat)))
        mae = float(np.mean(np.abs(y_true - y_hat)))
        results["val_mse"] = mse
        results["val_rmse"] = float(np.sqrt(mse))
        results["val_mae"] = mae
        logging.info(f"🎯 Validation MSE: {mse:.6f}, "
                     f"RMSE: {results['val_rmse']:.6f}, MAE: {mae:.6f} "
                     f"(normalized units)")
        plot_predictions(y_true, y_hat, epochs - 1, save_dir,
                         ylabel="Normalized close price")

        # Map predictions back to price units for reporting
        true_price = pred_price = naive_price = last_close = None
        if dataset.window_meta:
            meta = np.asarray(dataset.window_meta[train_size:
                                                  train_size + val_size])
            t_min, t_span, last_close = meta[:, 0], meta[:, 1], meta[:, 2]
            if target_mode == "delta":
                true_price = last_close + y_true * t_span
                pred_price = last_close + y_hat * t_span
            else:
                true_price = t_min + y_true * t_span
                pred_price = t_min + y_hat * t_span
            # Persistence baseline: tomorrow's price = today's price
            naive_price = last_close
        elif dataset.scalers.get("target") is not None:
            target_scaler = dataset.scalers["target"]
            true_price = target_scaler.inverse_transform(
                y_true.reshape(-1, 1)).ravel()
            pred_price = target_scaler.inverse_transform(
                y_hat.reshape(-1, 1)).ravel()
            naive_price = np.concatenate(([true_price[0]], true_price[:-1]))

        if true_price is not None:
            plot_predictions(true_price, pred_price, epochs - 1, save_dir,
                             ylabel="Close price ($)", suffix="_dollars")
            rmse_dollars = float(np.sqrt(np.mean(
                np.square(true_price - pred_price))))
            naive_rmse_dollars = float(np.sqrt(np.mean(
                np.square(true_price - naive_price))))
            results["val_rmse_dollars"] = rmse_dollars
            results["persistence_baseline_rmse_dollars"] = naive_rmse_dollars
            logging.info(f"💵 Validation RMSE: ${rmse_dollars:.4f} "
                         f"(persistence baseline: ${naive_rmse_dollars:.4f})")

            if last_close is not None:
                # Directional accuracy: did the model call tomorrow's move
                # (up/down vs today's close) correctly?
                true_up = true_price >= last_close
                pred_up = pred_price >= last_close
                dir_acc = float(np.mean(true_up == pred_up))
                results["directional_accuracy"] = dir_acc
                results["majority_direction_baseline"] = float(
                    max(np.mean(true_up), 1.0 - np.mean(true_up)))
                logging.info(f"🧭 Directional accuracy: {dir_acc:.4f} "
                             f"(majority baseline: "
                             f"{results['majority_direction_baseline']:.4f})")

    # ------------------------------------------------------------------
    # Diagnostic figures: loss curves, weight distributions, spike raster
    # ------------------------------------------------------------------
    plot_loss_curves(train_history, val_history, save_dir, metric_name)
    plot_spike_raster(spike_sequences[0], save_dir,
                      title=f"{encoder.__class__.__name__} spike raster (sample 0)")

    weights_data = {}
    for f in sorted((save_dir / "checkpoints").glob(f"{epochs - 1}-Conn*-g.npy")):
        name = f.stem[len(f"{epochs - 1}-"):-len("-g")]
        weights_data[name] = np.load(f)
    if weights_data:
        plot_weight_distributions(weights_data, save_dir, epochs - 1)

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"📁 Results and figures written to {save_dir}")

    return results
