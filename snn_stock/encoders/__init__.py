# snn_stock/encoders/__init__.py

from .rate_coding import RateEncoder
from .temporal_coding import TemporalEncoder

def get_encoder(encoding_type, n_steps=100):
    if encoding_type == "rate":
        return RateEncoder(n_steps=n_steps)
    elif encoding_type == "temporal":
        return TemporalEncoder(n_steps=n_steps)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")