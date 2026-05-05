# predict-weather

Calibration analysis of Polymarket weather markets vs NOAA forecasts.

The pipeline:

1. **Collects** resolved Polymarket weather markets via the public
   `gamma-api` and the J-1 mid-price via the CLOB `prices-history` endpoint.
2. **Aligns** each market with a NOAA probability derived from the NWS
   gridpoint forecast (`probabilityOfPrecipitation` for rain markets,
   normal-tail approximation around the day-1 forecast spread for
   temperature-threshold markets).
3. **Computes** Brier scores and a per-bucket calibration table; flags
   buckets where the |Polymarket − NOAA| gap exceeds 15 %.
4. **Simulates** a "buy when NOAA disagrees by ≥ 15 %" strategy with a 2 %
   fee and a quarter-Kelly stake; reports ROI, a one-sided t-test against
   zero, and a percentile bootstrap CI on the mean trade ROI.
5. **Plots** a reliability diagram (predicted probability vs realised
   frequency) for both sources.

## Layout

```
src/predict_weather/
    polymarket.py     gamma-api + CLOB client
    noaa.py           api.weather.gov client (replace `source` for archive)
    calibration.py    Brier + bucket table + reliability plot
    simulation.py     edge strategy + Kelly + significance tests
    synthetic.py      reproducible offline dataset
    pipeline.py       orchestrator + CLI
```

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# offline demo (synthetic data, fully deterministic)
python -m predict_weather.pipeline --source synthetic

# live run (needs internet access to gamma-api.polymarket.com and api.weather.gov)
python -m predict_weather.pipeline --source live
```

Outputs land in `output/`:

| file | content |
| --- | --- |
| `observations.csv` | one row per resolved market |
| `calibration_table.csv` | per-bucket gap, flagged at >15 % |
| `trades.csv` | simulated trades |
| `reliability.png` | reliability diagram |
| `summary.json` | Brier scores, ROI, p-value, bootstrap CI |

## Tests

```bash
pytest
```

## Caveats for a real backtest

* `api.weather.gov` only serves *current* forecasts. For an honest
  retrospective, swap the default `NOAAClient.source = None` with a
  `HistoricalForecastSource` that pulls the **NDFD archive** from NCEI —
  that archive contains the gridded forecast as it stood at any past time.
* Polymarket weather markets are sporadic; expect O(10²) observations
  over six months, not thousands. Significance is therefore power-limited.
* The synthetic dataset deliberately injects a tail-shrinkage bias so the
  analysis layer has a signal to detect; do not interpret the offline
  results as evidence of a real edge on Polymarket.
