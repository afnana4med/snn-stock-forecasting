import pandas as pd
import os

# Input and output paths
input_path = "data/raw/AAPL_1min/aapl_1min_data From 2006 -2024.csv"
output_path = "data/processed/cleaned_AAPL_1min.csv"

# Load, clean and save
df = pd.read_csv(input_path)
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path)

print(f"✅ Cleaned data saved to: {output_path}")
