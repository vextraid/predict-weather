"""Daily append script — adds today's observations to the history CSV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import date
import pandas as pd
from predict_weather.pipeline import collect_live
from predict_weather.synthetic import build_synthetic_dataset
from predict_weather.data_model import to_dataframe

OUT = Path("output/history.csv")
OUT.parent.mkdir(exist_ok=True)

try:
    obs = collect_live()
    if not obs:
        raise ValueError("no observations")
    source = "live"
except Exception as e:
    print(f"live failed ({e}), using synthetic fallback")
    obs = build_synthetic_dataset(n=10)
    source = "synthetic-fallback"

df = to_dataframe(obs)
df["collected_at"] = date.today().isoformat()
df["source"] = source

if OUT.exists():
    existing = pd.read_csv(OUT)
    df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=["market_id", "collected_at"])

df.to_csv(OUT, index=False)
print(f"Saved {len(df)} total rows to {OUT}")
print(df[["collected_at","city","title","poly_prob","noaa_prob"]].tail(10).to_string())
