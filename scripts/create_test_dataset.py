import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data/processed/cleaned_AAPL_1min.csv"


def create_test_dataset(n_rows=1000):
    """Create a small minute-level file for quick pipeline tests.

    NOTE: the minute-level values in this dataset are interpolated between
    daily closes, so they contain no real intraday movement - use the daily
    dataset created by create_daily_dataset() for meaningful experiments.
    """
    output_file = PROJECT_ROOT / "data/processed/test_AAPL_1min.csv"
    logging.info(f"Reading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, nrows=n_rows)
    df.to_csv(output_file, index=False)
    logging.info(f"Created test dataset with {len(df)} rows: {output_file}")


def create_daily_dataset():
    """Resample the minute-level file to real daily OHLCV bars."""
    output_file = PROJECT_ROOT / "data/processed/daily_AAPL.csv"
    logging.info(f"Reading input file: {INPUT_FILE} (this can take a minute)")
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    df["day"] = df["Date"].dt.date

    daily = df.groupby("day").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum")).reset_index()
    daily = daily.rename(columns={"day": "Date"})

    daily.to_csv(output_file, index=False)
    logging.info(f"Created daily dataset with {len(daily)} rows: {output_file}")


if __name__ == "__main__":
    create_test_dataset()
    create_daily_dataset()
