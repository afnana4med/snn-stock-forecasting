import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class PriceDataset(Dataset):
    def __init__(self,
                 file_paths,
                 sequence_length=60,
                 prediction_horizon=1,
                 task='regression',  # 'regression' or 'classification'
                 features=['Close'],
                 target_column='Close',
                 normalize=True,  # False, True/'global', or 'window'
                 target_mode='level',  # 'level' or 'delta' (window mode only)
                 scaler_fit_fraction=0.8,
                 use_cache=False,
                 cache_dir='./cache'):

        # Validate inputs
        if not isinstance(file_paths, (list, str)):
            raise TypeError("file_paths must be a string or list of strings")
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        if not all(os.path.exists(f) for f in file_paths):
            raise FileNotFoundError("One or more input files not found")

        if task not in ['regression', 'classification']:
            raise ValueError("task must be either 'regression' or 'classification'")

        if target_mode not in ['level', 'delta']:
            raise ValueError("target_mode must be 'level' or 'delta'")
        if target_mode == 'delta' and normalize != 'window':
            raise ValueError("target_mode 'delta' requires normalize='window'")
        self.target_mode = target_mode

        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.task = task
        self.features = list(features)
        self.target_column = target_column
        self.normalize = normalize
        # Scalers are fitted on the leading fraction of each file only, so
        # that statistics from the validation portion never leak into training
        self.scaler_fit_fraction = scaler_fit_fraction
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.scalers = {}
        # For 'window' normalization: per-sample (target_min, target_span,
        # last_close) so predictions can be mapped back to price units
        self.window_meta = []
        self.data = self._load_and_process_data()

    def _load_and_process_data(self):
        all_sequences = []
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)

            # Check for missing columns
            required_columns = set(self.features + [self.target_column])
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            df = df.dropna(subset=list(required_columns)).reset_index(drop=True)

            # Keep the raw target so the target scaler is not applied on top
            # of already-scaled feature values (the target column usually
            # also appears in the feature list)
            raw_target = df[[self.target_column]].to_numpy(dtype=np.float64)

            window_normalize = (self.normalize == 'window')
            if self.normalize and not window_normalize:
                fit_rows = max(int(len(df) * self.scaler_fit_fraction), 2)
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df[self.features].iloc[:fit_rows])
                feature_values = feature_scaler.transform(df[self.features])

                target_scaler = MinMaxScaler()
                target_scaler.fit(raw_target[:fit_rows])
                target_values = target_scaler.transform(raw_target).ravel()

                self.scalers['features'] = feature_scaler
                self.scalers['target'] = target_scaler
            else:
                feature_values = df[self.features].to_numpy(dtype=np.float64)
                target_values = raw_target.ravel()

            seq, hor = self.sequence_length, self.prediction_horizon
            for i in range(len(df) - seq - hor + 1):
                X = feature_values[i:i + seq]

                if window_normalize:
                    # Normalize each window by its own range. This keeps
                    # inputs stationary over long, trending price histories
                    # (a global scaler fitted on the training years cannot
                    # cover later prices)
                    f_min = X.min(axis=0)
                    f_span = np.maximum(X.max(axis=0) - f_min, 1e-8)
                    X = (X - f_min) / f_span

                    t_window = raw_target[i:i + seq, 0]
                    t_min = t_window.min()
                    # Floor the span at 0.1% of the price to avoid exploding
                    # targets in flat windows
                    t_span = max(t_window.max() - t_min,
                                 1e-8 + 1e-3 * abs(t_min))

                X = X.astype(np.float32)
                last_close = raw_target[i + seq - 1, 0]

                if self.task == 'classification':
                    # Direction of the price move between the end of the
                    # window and the prediction horizon (0=down, 1=up)
                    future_close = raw_target[i + seq + hor - 1, 0]
                    y = int(future_close >= last_close)
                elif window_normalize:
                    future_close = raw_target[i + seq + hor - 1, 0]
                    if self.target_mode == 'delta':
                        # Predict the move relative to the last observed
                        # close: an output of 0 equals the persistence
                        # baseline, so anything learned is pure signal
                        y = np.array([(future_close - last_close) / t_span],
                                     dtype=np.float32)
                    else:
                        y = np.array([(future_close - t_min) / t_span],
                                     dtype=np.float32)
                    self.window_meta.append((t_min, t_span, last_close))
                else:
                    y = np.array([target_values[i + seq + hor - 1]],
                                 dtype=np.float32)

                all_sequences.append((X, y))
        return all_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def print_dataset_stats(self):
        """Print basic statistics about the dataset"""
        n_samples = len(self)
        X, y = self[0]

        print("Dataset Statistics:")
        print(f"Total samples: {n_samples}")
        print(f"Features: {self.features}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Prediction horizon: {self.prediction_horizon}")
        print(f"Input shape: {X.shape}")

        if self.task == 'classification':
            class_counts = {}
            for _, label in self.data:
                class_counts[label] = class_counts.get(label, 0) + 1
            print("Task: Classification")
            print(f"Number of classes: {len(class_counts)}")
            print(f"Class distribution: {class_counts}")
        else:
            print("Task: Regression")
            print(f"Target shape: {y.shape}")

    @property
    def input_shape(self):
        """Return the shape of input sequences"""
        return (self.sequence_length, len(self.features))

    @property
    def output_shape(self):
        """Return the shape of output targets"""
        if self.task == 'regression':
            return (1,)
        else:
            # For classification, we return the number of classes (2 for binary)
            return (2,)  # Binary classification: 0=down, 1=up
