"""Rigorous evaluation protocol for the SNN forecasting models.

Implements:
- Chronological expanding-window (walk-forward) cross-validation
- Multi-seed training with seed-ensembled predictions
- Diebold-Mariano forecast-comparison significance test
- Classical baselines on identical inputs: persistence, ridge regression, LSTM
- Computational-cost accounting: synaptic events (SNN) vs MACs (LSTM/ridge)
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

from ml_genn.callbacks import Checkpoint, SpikeRecorder
from ml_genn.compilers import EPropCompiler, EventPropCompiler, InferenceCompiler
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy

from snn_stock.data.dataset_loader import PriceDataset
from snn_stock.models.snn_model import build_snn
from snn_stock.training.trainer import (LastTimestepMSE, encode_dataset,
                                        get_encoder)
from snn_stock.utils import convert_rate_to_spike_times


# ---------------------------------------------------------------------------
# Walk-forward folds and statistics
# ---------------------------------------------------------------------------

def walk_forward_folds(n_samples, n_folds=4, initial_train_frac=0.4):
    """Expanding-window folds: train on everything before each test block."""
    first_test_start = int(n_samples * initial_train_frac)
    block = (n_samples - first_test_start) // n_folds
    folds = []
    for k in range(n_folds):
        test_start = first_test_start + k * block
        test_end = test_start + block if k < n_folds - 1 else n_samples
        folds.append((np.arange(0, test_start),
                      np.arange(test_start, test_end)))
    return folds


def diebold_mariano(errors_a, errors_b, h=1):
    """DM test on squared-error loss differentials (Harvey-adjusted).

    Returns (dm_statistic, p_value). Negative statistic means model A has
    lower loss than model B.
    """
    d = np.asarray(errors_a) ** 2 - np.asarray(errors_b) ** 2
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance with h-1 lags (h=1 -> plain variance)
    gamma = [np.mean((d[:n - lag] - dbar) * (d[lag:] - dbar))
             for lag in range(h)]
    lrv = gamma[0] + 2.0 * sum(gamma[1:])
    if lrv <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(lrv / n)
    # Harvey, Leybourne & Newbold small-sample adjustment
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


# ---------------------------------------------------------------------------
# Classical baselines (identical inputs and targets as the SNN)
# ---------------------------------------------------------------------------

def fit_predict_ridge(X_train, y_train, X_test):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model.predict(X_test.reshape(len(X_test), -1))


def fit_predict_lstm(X_train, y_train, X_test, seed=42, hidden=32,
                     epochs=150, patience=15):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class LSTMBaseline(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    # Hold out the chronologically-last 15% of the training data for
    # early stopping
    n_val = max(int(0.15 * len(X_train)), 1)
    Xt = torch.tensor(X_train[:-n_val], dtype=torch.float32)
    yt = torch.tensor(y_train[:-n_val], dtype=torch.float32).reshape(-1, 1)
    Xv = torch.tensor(X_train[-n_val:], dtype=torch.float32)
    yv = torch.tensor(y_train[-n_val:], dtype=torch.float32).reshape(-1, 1)

    model = LSTMBaseline(X_train.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
    return pred.numpy().ravel()


def lstm_macs_per_prediction(seq_len, n_features, hidden=32):
    """Multiply-accumulates for one LSTM forward pass (4 gates + readout)."""
    per_step = 4 * hidden * (n_features + hidden)
    return seq_len * per_step + hidden


# ---------------------------------------------------------------------------
# SNN fold runner
# ---------------------------------------------------------------------------

def train_snn_fold(config, spike_arrays, y_targets, train_idx, test_idx,
                   seed, measure_spikes=False):
    """Train the SNN on one fold and return (test_predictions, spike_stats)."""
    n_steps = config["n_steps"]
    dt = float(config["dt"])
    algorithm = config["algorithm"].lower()
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    lr_decay = config["training"].get("lr_decay")
    lr_decay_every = int(config["training"].get("lr_decay_every", 10))
    np.random.seed(seed)

    input_size = spike_arrays.shape[-1]
    net, input_pop, output_pop = build_snn(
        input_size=input_size,
        hidden_layers=config["model"]["hidden_layers"],
        output_size=1,
        neuron_params=config["model"]["neuron_params"],
        output_readout="var",
        algorithm=algorithm,
        max_input_spikes=input_size * n_steps,
        recurrent=config["model"].get("recurrent", False))

    compiler_args = dict(example_timesteps=n_steps, losses="mean_square_error",
                         optimiser=Adam(lr), batch_size=1, dt=dt,
                         rng_seed=seed, backend="single_threaded_cpu")
    if algorithm == "eprop":
        compiler = EPropCompiler(**compiler_args)
    else:
        compiler = EventPropCompiler(max_spikes=max(500, n_steps + 1),
                                     **compiler_args)
    compiled_net = compiler.compile(net)

    train_x = convert_rate_to_spike_times(spike_arrays[train_idx], dt=dt)
    test_x = convert_rate_to_spike_times(spike_arrays[test_idx], dt=dt)
    train_y = np.repeat(y_targets[train_idx, np.newaxis, :], n_steps, axis=1)

    def alpha_schedule(epoch, alpha):
        if lr_decay and epoch > 0 and epoch % lr_decay_every == 0:
            return alpha * lr_decay
        return alpha

    from ml_genn.callbacks import OptimiserParamSchedule
    with tempfile.TemporaryDirectory() as ckpt_dir:
        serialiser = Numpy(ckpt_dir)
        callbacks = [Checkpoint(serialiser)]
        if lr_decay:
            callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule))
        with compiled_net:
            compiled_net.train({input_pop: train_x}, {output_pop: train_y},
                               num_epochs=epochs, shuffle=True,
                               metrics=LastTimestepMSE(),
                               callbacks=callbacks)

        net.load((epochs - 1,), serialiser)
        inference = InferenceCompiler(
            evaluate_timesteps=n_steps, dt=dt, batch_size=1, rng_seed=seed,
            backend="single_threaded_cpu").compile(net)

        spike_stats = None
        with inference:
            callbacks = []
            hidden_pops = [p for p in net.populations
                           if p not in (input_pop, output_pop)]
            if measure_spikes:
                callbacks = [SpikeRecorder(hidden_pops[0], key="hidden")]
            y_pred_dict, cb_data = inference.predict(
                {input_pop: test_x}, output_pop, callbacks=callbacks)
            y_pred = np.asarray(y_pred_dict[output_pop]).ravel()

            if measure_spikes:
                hidden_units = config["model"]["hidden_layers"][0]["units"]
                in_spikes = float(np.mean(
                    spike_arrays[test_idx].sum(axis=(1, 2))))
                hid_spikes = float(np.mean(
                    [len(t) for t in cb_data["hidden"][0]]))
                recurrent = config["model"].get("recurrent", False)
                syn_events = (in_spikes * hidden_units +
                              hid_spikes * (1 + (hidden_units if recurrent
                                                 else 0)))
                spike_stats = {
                    "input_spikes_per_example": in_spikes,
                    "hidden_spikes_per_example": hid_spikes,
                    "synaptic_events_per_prediction": syn_events,
                    "neuron_updates_per_prediction":
                        float(n_steps * (hidden_units + 1)),
                }

    return y_pred, spike_stats


# ---------------------------------------------------------------------------
# Full protocol
# ---------------------------------------------------------------------------

def run_final_evaluation(config, seeds=(42, 123, 777), n_folds=4,
                         max_samples=None):
    """Walk-forward, multi-seed evaluation of one config vs baselines.

    Returns a dict with per-model pooled RMSE ($), DM tests vs the SNN, and
    computational-cost statistics.
    """
    dataset = PriceDataset(
        file_paths=config["data"]["files"],
        sequence_length=config["data"]["sequence_length"],
        prediction_horizon=config["data"]["prediction_horizon"],
        task="regression",
        features=config["data"]["features"],
        target_column=config["data"].get("target_column", "Close"),
        normalize="window",
        target_mode="delta",
        engineered_features=config["data"].get("engineered_features", False),
        context_files=config["data"].get("context_files"))

    encoder = get_encoder(config)
    spike_arrays, labels = encode_dataset(dataset, encoder, max_samples)
    n = len(spike_arrays)
    y = labels.reshape(n, -1).astype(np.float32)

    # Raw feature windows for the classical baselines: identical
    # window-normalized inputs, identical delta targets
    X_windows = np.stack([np.asarray(dataset[i][0]) for i in range(n)])

    meta = np.asarray(dataset.window_meta[:n])
    t_span, last_close = meta[:, 1], meta[:, 2]

    folds = walk_forward_folds(n, n_folds=n_folds)
    logging.info(f"🏁 Final evaluation: {n} windows, {n_folds} folds, "
                 f"{len(seeds)} seeds, encoder {encoder.__class__.__name__}, "
                 f"{spike_arrays.shape[-1]} input neurons")

    all_test_idx = np.concatenate([te for _, te in folds])
    preds = {"snn": [], "ridge": [], "lstm": []}
    spike_stats = None

    for k, (tr, te) in enumerate(folds):
        # SNN: average predictions over seeds (seed ensemble)
        seed_preds = []
        for s_i, seed in enumerate(seeds):
            measure = (k == 0 and s_i == 0)
            p, st = train_snn_fold(config, spike_arrays, y, tr, te, seed,
                                   measure_spikes=measure)
            seed_preds.append(p)
            if st:
                spike_stats = st
            logging.info(f"  fold {k + 1}/{n_folds} seed {seed}: "
                         f"SNN fold RMSE(delta) = "
                         f"{np.sqrt(np.mean((p - y[te].ravel())**2)):.4f}")
        preds["snn"].append(np.mean(seed_preds, axis=0))

        preds["ridge"].append(fit_predict_ridge(X_windows[tr], y[tr].ravel(),
                                                X_windows[te]))
        preds["lstm"].append(fit_predict_lstm(X_windows[tr], y[tr].ravel(),
                                              X_windows[te]))
        logging.info(f"  fold {k + 1}/{n_folds}: baselines done")

    # Pool folds and convert everything to dollars
    span_te = t_span[all_test_idx]
    last_te = last_close[all_test_idx]
    true_price = last_te + y[all_test_idx].ravel() * span_te

    dollar_errors = {}
    for name in preds:
        pred_price = last_te + np.concatenate(preds[name]) * span_te
        dollar_errors[name] = pred_price - true_price
    dollar_errors["persistence"] = last_te - true_price

    results = {"n_test_predictions": int(len(all_test_idx)),
               "n_folds": n_folds, "seeds": list(seeds),
               "models": {}, "dm_tests": {}}
    for name, err in dollar_errors.items():
        results["models"][name] = {
            "rmse_dollars": float(np.sqrt(np.mean(err ** 2))),
            "mae_dollars": float(np.mean(np.abs(err))),
        }
    for name in ("persistence", "ridge", "lstm"):
        dm, p = diebold_mariano(dollar_errors["snn"], dollar_errors[name])
        results["dm_tests"][f"snn_vs_{name}"] = {
            "dm_statistic": dm, "p_value": p,
            "interpretation": ("snn better" if dm < 0 else "snn worse") +
                              (" (significant)" if p < 0.05 else
                               " (not significant)")}

    # Directional accuracy (sign of the predicted move)
    for name in preds:
        pred_up = np.concatenate(preds[name]) >= 0
        true_up = y[all_test_idx].ravel() >= 0
        results["models"][name]["directional_accuracy"] = float(
            np.mean(pred_up == true_up))
    results["models"]["persistence"]["directional_accuracy"] = float(
        max(np.mean(y[all_test_idx].ravel() >= 0),
            1 - np.mean(y[all_test_idx].ravel() >= 0)))

    # Computational cost
    seq_len = config["data"]["sequence_length"]
    n_features = X_windows.shape[-1]
    hidden = config["model"]["hidden_layers"][0]["units"]
    if spike_stats:
        results["efficiency"] = dict(spike_stats)
        results["efficiency"]["lstm_macs_per_prediction"] = float(
            lstm_macs_per_prediction(seq_len, n_features, hidden=32))
        results["efficiency"]["ridge_macs_per_prediction"] = float(
            seq_len * n_features)
        results["efficiency"]["snn_total_ops_per_prediction"] = float(
            spike_stats["synaptic_events_per_prediction"] +
            spike_stats["neuron_updates_per_prediction"])

    # Keep raw pooled series for plotting
    results["_series"] = {
        "true_price": true_price.tolist(),
        "pred_price_snn": (last_te + np.concatenate(preds["snn"]) *
                           span_te).tolist(),
    }
    return results
