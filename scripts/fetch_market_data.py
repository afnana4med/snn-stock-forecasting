"""Download daily OHLCV data for stocks, ETFs and crypto with yfinance.

Each ticker is saved to data/raw/<name>_daily.csv with the columns the
PriceDataset loader expects (Date, Open, High, Low, Close, Volume), so any
of them can be dropped straight into a config's data.files entry.

Usage:
    python scripts/fetch_market_data.py            # fetch the default set
    python scripts/fetch_market_data.py TSLA BTC-USD   # fetch specific tickers
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_TICKERS = {
    # Stocks
    "AAPL": "apple", "MSFT": "microsoft", "TSLA": "tesla",
    "NVDA": "nvidia", "AMZN": "amazon",
    # ETFs
    "SPY": "sp500_etf", "QQQ": "nasdaq100_etf", "GLD": "gold_etf",
    # Crypto (trades 7 days/week)
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum",
}


def fetch_ticker(ticker, name=None, start="2015-01-01"):
    name = name or ticker.replace("-", "_").lower()
    logging.info(f"Fetching {ticker} daily data since {start}...")
    df = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        logging.warning(f"No data returned for {ticker} - skipped")
        return None

    # yfinance returns a MultiIndex column frame; flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{name}_daily.csv"
    df.to_csv(out_file, index=False)
    logging.info(f"Saved {len(df)} rows -> {out_file}")
    return out_file


if __name__ == "__main__":
    requested = sys.argv[1:]
    if requested:
        tickers = {t: None for t in requested}
    else:
        tickers = DEFAULT_TICKERS

    fetched = 0
    for ticker, name in tickers.items():
        try:
            if fetch_ticker(ticker, name) is not None:
                fetched += 1
        except Exception as e:
            logging.error(f"Failed to fetch {ticker}: {e}")
    logging.info(f"Done: {fetched}/{len(tickers)} datasets fetched")
