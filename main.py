from datetime import datetime, timedelta
from time import time_ns

from alpaca.data.timeframe import TimeFrame

from ds.bars import Conversion
from ds.config import Config
from ds.retrieval import Security
from ds.thresholds import Threshold
from ds.train import train


def main() -> None:
  # Configure a 7-day, 1-minute retrieval window with all parameters
  config = Config(
    start=datetime.now() - timedelta(days=2),
    step=TimeFrame.Minute,
    plot_enabled=True,
    plot_save=True,
    plot_show=False,
    conversion=Conversion.DOLLAR,
    threshold_strategy=Threshold.EMA,
    sg_window_length=50,
    sg_polyorder=2,
  )

  # Instantiate and run the offline pipeline
  symbols: list[str] = [
    # "META",
    "AAPL",
    # "GOOG",
    # "GOOGL",
    "NVDA",
    # "TSM",
    # "ORCL",
    "PLTR",
    # "XOM",
    # "RY",
    # "DB",
    # "JPM",
  ]

  start = time_ns()
  results = train(symbols, Security.EQUITIES, config)
  end = time_ns()

  for symbol, data in results.items():
    print(f"\n=== {symbol} ===")
    for seg in data["labels"]:
      start, end = seg["start"], seg["end"]
      label, params, error = seg["label"], seg["params"], seg["error"]
      print(f"{start:>4}:{end:<4}  {label:<12}  {params=!r}  {error=:.4f}")

  print(f"Training completed in {end - start} nanoseconds")

if __name__ == "__main__":
  main()
