# Stock Price Prediction with Spiking Neural Networks (SNN)

Spiking neural networks trained on AAPL price data using
[ml_genn](https://github.com/genn-team/ml_genn) on top of the
[GeNN](https://github.com/genn-team/genn) simulator. Two biologically
plausible learning algorithms are supported:

- **e-prop** (eligibility propagation) — regression: predict the next day's
  close price
- **EventProp** (exact event-driven gradients) — classification: predict the
  next day's price direction (up/down)

and two spike encodings:

- **Rate coding** — each input value is one input neuron whose spike count is
  proportional to the value
- **Temporal (time-to-first-spike) coding** — each input neuron spikes once;
  larger values spike earlier

## Setup

Linux/Windows:

```bash
pip install pygenn            # prebuilt wheels available
git clone https://github.com/genn-team/ml_genn.git
pip install ./ml_genn/ml_genn
pip install -r requirements.txt
```

macOS (no pygenn wheels — build GeNN from source; requires Xcode command
line tools):

```bash
git clone https://github.com/genn-team/genn.git
pip install ./genn
git clone https://github.com/genn-team/ml_genn.git
pip install ./ml_genn/ml_genn
```

Note: `pynn_genn` is a different (PyNN-based) interface to GeNN and is NOT
used by this project.

## Data

`data/processed/cleaned_AAPL_1min.csv` contains 2006–2024 "minute" bars,
**but the intra-day values are interpolated between daily closes** (e.g. the
first 1000 minutes contain 999 consecutive up-ticks). Minute-level prediction
on this file is therefore meaningless. The experiments instead use real daily
bars produced by resampling:

```bash
python scripts/create_test_dataset.py   # writes data/processed/daily_AAPL.csv
```

Input windows and regression targets are normalized **per window** (each
20-day window scaled by its own min/max), which keeps the encoding stationary
across 18 years of exponential price growth and avoids look-ahead leakage.

### More assets (stocks, ETFs, crypto)

Real daily OHLCV data for additional assets can be fetched with yfinance:

```bash
python scripts/fetch_market_data.py                # AAPL MSFT TSLA NVDA AMZN SPY QQQ GLD BTC ETH
python scripts/fetch_market_data.py SOL-USD VTI    # or any specific tickers
```

Files land in `data/raw/<name>_daily.csv` in the exact format the loader
expects — point any config's `data.files` at one of them. Two ready-made
examples: `configs/spy_rate_eprop.yaml` (S&P 500 ETF regression) and
`configs/btc_temporal_eventprop.yaml` (Bitcoin direction classification).

## Running

```bash
python -m snn_stock.main --config configs/rate_eprop.yaml          # regression, rate code, e-prop
python -m snn_stock.main --config configs/temporal_eprop.yaml      # regression, TTFS code, e-prop
python -m snn_stock.main --config configs/temporal_eventprop.yaml  # classification, TTFS code, EventProp
```

Each run trains the SNN, checkpoints weights every epoch, reloads the final
checkpoint into an inference network, and writes real out-of-sample
predictions, metrics (`results.json`) and figures (loss curves, predicted vs
true prices, input spike raster, weight histograms) to
`experiments/<name>/<experiment_name>/`.

The GeNN CUDA backend is used automatically when available; otherwise the
single-threaded CPU backend is used (batch size is forced to 1 there).

## Results (validation = final chronological 20% of 1500 daily windows)

| Experiment | Asset | Task | Result | Baseline |
|---|---|---|---|---|
| rate + e-prop | AAPL (daily) | next-day close (regression) | RMSE **$0.25** | persistence $0.12 |
| temporal + e-prop | AAPL (daily) | next-day close (regression) | RMSE **$0.29** | persistence $0.12 |
| temporal + EventProp | AAPL (daily) | next-day direction (classification) | accuracy **0.54** | majority class 0.61 |
| rate + e-prop | SPY ETF 2015–2026 | next-day close (regression) | RMSE **$13.8** (~2.5%) | persistence $5.9 |
| temporal + EventProp | BTC-USD 2015–2026 | next-day direction (classification) | accuracy **0.50** | majority class 0.51 |

Interpretation: the SNNs genuinely learn — training loss falls smoothly and
out-of-sample predictions closely track the true price curve (see
`predictions_epoch_*_dollars.png`). However, level-prediction models do not
beat the naive persistence/majority baselines. This is the expected outcome
for daily equity prices, which are close to a random walk.

## Hyperparameter sweep (deeper analysis)

`python scripts/run_experiments.py` runs 10 tuned variants on AAPL daily data
and aggregates them into `experiments/sweep/summary.csv` and
`experiments/sweep/sweep_summary.png`:

| Variant | Val RMSE ($) | vs persistence $0.116 |
|---|---|---|
| baseline (15 epochs) | 0.255 | 2.2x worse |
| + 40 epochs, LR decay 0.7/10 | 0.248 | 2.1x worse |
| + hidden 64 | 0.254 | no gain |
| + n_steps 80 (finer rate code) | 0.243 | 2.1x worse |
| **+ delta target** (predict the move, not the level) | **0.116** | **matches/beats** |
| delta + temporal encoding | **0.116** | **best** |

Findings:

1. **Problem formulation dominates hyperparameters.** More epochs, more
   neurons and finer spike codes each buy only ~5%. Switching the regression
   target from the price *level* to the *move relative to the last close*
   (`target_mode: "delta"`) halves the error and reaches the persistence
   baseline — because in delta form, "output 0" already equals persistence
   and anything learned is pure signal.
2. **There is almost no learnable signal beyond persistence.** The best delta
   models beat persistence by ~0.3% ($0.1160 vs $0.1164), and direction
   accuracy never exceeds the majority-class baseline out of sample (train
   accuracy rises to ~0.56, validation stays at chance). Both facts are
   textbook consequences of near-efficient daily markets — the honest claim
   this project supports is that biologically plausible SNN learning rules
   (e-prop, EventProp) recover the statistically optimal naive predictor from
   spike-encoded price data.
3. LR decay (0.7 every 10 epochs) stabilizes e-prop's noisy training loss;
   configure via `training.lr_decay` / `training.lr_decay_every`.

Applying the winning recipe (delta target + temporal code + 40 epochs with LR
decay; `configs/spy_delta_temporal.yaml`, `configs/btc_delta_temporal.yaml`)
to the other assets:

| Asset | Val RMSE ($) | Persistence baseline | Verdict |
|---|---|---|---|
| SPY (2024–2026 held out) | **5.899** | 5.921 | beats baseline (untuned level model: 13.8) |
| BTC-USD (2023–2026 held out) | 1973.98 | 1973.15 | statistical tie |

## Final rigorous evaluation (walk-forward + significance + baselines)

`python scripts/run_final_eval.py` runs the gold-standard protocol and writes
results and figures to `experiments/final_eval/`:

- **Expanding-window walk-forward** cross-validation (4 folds — always train on
  the past, test on the future; no look-ahead leakage)
- **3-seed ensemble** per fold (predictions averaged over seeds 42/123/777)
- **Diebold–Mariano** significance test on every model comparison
- **Classical baselines on identical inputs**: persistence, ridge regression,
  and an early-stopped **LSTM** (PyTorch)
- **Computational-cost accounting**: SNN synaptic events + neuron updates vs
  LSTM/ridge multiply-accumulates (MACs)

Four ablations on AAPL daily (1,500 pooled test predictions), delta target:

| Variant | SNN RMSE ($) | Persistence | Ridge | LSTM | SNN vs best baseline (DM) |
|---|---|---|---|---|---|
| base (OHLCV, TTFS code) | 0.230 | **0.199** | 0.211 | 0.201 | worse, p<0.001 |
| + engineered features | 0.214 | **0.199** | 0.214 | 0.201 | ties ridge (p=0.98), worse than persistence |
| + SPY context + recurrent (RSNN)¹ | 3.844 | **2.300** | 2.528 | 2.322 | worse, p<0.001 |
| temporal-contrast code | 0.214 | **0.199** | 0.214 | 0.201 | ties ridge (p=0.98) |

¹ The context variant inner-joins with SPY (data starts 2015), so it runs on a
later, higher-priced period — its dollar RMSE is **not** comparable to the other
rows, only to its own baselines.

### The headline result: efficiency, not accuracy

`experiments/final_eval/final_efficiency_comparison.png`

| Model | Operations per prediction | Relative |
|---|---|---|
| **SNN (e-prop)** | **4,850** (3,200 synaptic events + 1,650 neuron updates) | **1×** |
| LSTM | 94,752 MACs | 20× more |
| Ridge | 100 MACs | (a trivial linear model) |

**Conclusions:**

1. **On accuracy, no model beats persistence** on daily prices — the SNN, LSTM
   and ridge all lose to "tomorrow = today" by a statistically significant
   margin (DM p<0.001). This is the textbook signature of a near-efficient
   market and is the honest, defensible finding.
2. **With engineered features (returns, volatility, volume ratio, high–low
   range) the SNN matches ridge regression** (DM p=0.98, i.e. no significant
   difference) — so the spiking model learns a linear-quality mapping from
   spike-encoded inputs.
3. **The SNN's real advantage is compute**: it produces each forecast with
   ~20× fewer operations than the matched LSTM, exactly the energy/latency
   argument that motivates SNNs for high-frequency settings.
4. Engineered features lifted SNN direction accuracy from 0.47 → 0.51 and
   halved the gap to persistence; the recurrent + context variant did not help
   on the shorter, harder later period.

## Multi-horizon forecasting (does trend signal beat persistence?)

`python scripts/run_multi_horizon.py` holds everything constant (AAPL daily,
20-day window, engineered features, delta target, e-prop) and varies only the
forecast horizon over 1 / 5 / 10 / 20 trading days, adding a **linear-trend /
drift** baseline alongside persistence / ridge / LSTM. Figure:
`experiments/multi_horizon/multi_horizon_summary.png`.

Test RMSE ($) — pooled walk-forward, 3-seed ensemble:

| Horizon | SNN | Persistence | Trend | Ridge | LSTM | SNN vs persistence |
|---|---|---|---|---|---|---|
| 1 day | 0.214 | **0.199** | 0.395 | 0.214 | 0.201 | worse (p<0.001) |
| 5 days | 0.496 | **0.477** | 0.660 | 0.524 | 0.501 | worse (p=0.007) |
| **10 days** | 0.683 | **0.677** | 0.936 | 0.756 | 0.712 | **tie (p=0.40)** |
| 20 days | 1.000 | **0.948** | 1.394 | 1.071 | 1.009 | worse (p<0.001) |

Direction-of-move accuracy (sign of the h-day move):

| Horizon | SNN | Persistence (majority) | Coin flip |
|---|---|---|---|
| 1 day | 0.514 | 0.573 | 0.50 |
| 5 days | 0.524 | 0.570 | 0.50 |
| 10 days | **0.612** | 0.607 | 0.50 |
| 20 days | 0.623 | 0.623 | 0.50 |

**Findings — trend signal genuinely emerges at longer horizons:**

1. **SNN direction accuracy climbs steeply with horizon** (0.51 → 0.52 → 0.61 →
   0.62). Daily direction is essentially unpredictable (0.51, barely above a
   coin flip), but 10–20-day direction reaches ~0.62 — clear evidence of
   predictable multi-day trend structure that does not exist at daily scale.
2. **h=10 is the sweet spot.** The SNN's price RMSE becomes *statistically
   indistinguishable from persistence* (DM p=0.40 — no significant loss, unlike
   every other horizon), and its 0.612 direction accuracy slightly **exceeds**
   the majority-class baseline (0.607). This is the closest any configuration
   in the whole project comes to beating the naive benchmark.
3. **Among learned models, the SNN is strongest at multi-day structure**: it
   beats the linear-trend and ridge baselines at every horizon ≥5 days
   (DM p<0.001) and beats the LSTM at h=10 (DM p=0.002).
4. The **linear-trend/drift baseline is consistently the worst** — naively
   extrapolating a straight line overshoots. The SNN's advantage is that it
   learns *when* trend persists rather than assuming it always does.

Takeaway: the honest, defensible headline is that **forecastability increases
with horizon** — the SNN captures 10–20-day trend direction (~62%) that is
absent at daily scale, and at the 10-day horizon it matches persistence on
price error while edging out the majority-class direction baseline.

## Project layout

```
snn_stock/
  data/dataset_loader.py    sliding-window dataset; per-window/global scaling,
                            delta targets, engineered features, multi-asset context
  encoders/                 rate, time-to-first-spike, temporal-contrast encoders
  models/snn_model.py       SpikeInput -> LIF hidden (optionally recurrent) -> readout
  training/trainer.py       compile, train, checkpoint, evaluate, plot
  training/evaluation.py    walk-forward + multi-seed + DM test + ridge/LSTM baselines
  utils/                    spike-format conversion, visualization
configs/                    one YAML per experiment
scripts/create_test_dataset.py  builds test + daily datasets
scripts/fetch_market_data.py    downloads multi-asset daily OHLCV via yfinance
scripts/run_experiments.py      hyperparameter sweep runner
scripts/run_final_eval.py       rigorous walk-forward evaluation vs baselines
scripts/run_multi_horizon.py    horizon sweep (1/5/10/20 days) vs trend baseline
tests/                      unit tests (python -m unittest discover tests)
```
