from datetime import datetime, timedelta

import seaborn as sns
from alpaca.data.timeframe import TimeFrame

from ds.bars import Conversion
from ds.config import Config, PlotConfig
from ds.train import Train
from ds.retrieval import Security

sns.set_theme(style="darkgrid")


def main() -> None:
    # Configure a 7-day, 1-minute retrieval window with all parameters
    config = Config(
        start=datetime.now() - timedelta(days=7),
        step=TimeFrame.Minute,
        plot=PlotConfig(
            enabled=True,
            save=True,
            show=False,
        ),
        conversion=Conversion.DOLLAR,
        volume_threshold=5_000_000,
        dollar_threshold=500_000_000.0,
        sg_window_length=60,
        sg_polyorder=2,
        segmentation_model="l2",
        segmentation_penalty=10.0,
        segmentation_min_size=1,
        segmentation_jump=1,
    )

    # Instantiate and run the offline pipeline
    pipeline = Train(config)
    symbols: list[str] = [
        # "META",
        # "AAPL",
        # "GOOG",
        # "GOOGL",
        "NVDA",
        # "TSM",
        # "ORCL",
        # "PLTR",
        # "XOM",
        # "RY",
        # "DB",
        # "JPM",
    ]

    results = pipeline.run(symbols, Security.EQUITIES)

    # Print segment info
    for symbol, data in results.items():
        print(f"\n=== {symbol} ===")
        for seg in data["labels"]:
            start, end = seg["start"], seg["end"]
            label, params, error = seg["label"], seg["params"], seg["error"]
            print(f"{start:>4}:{end:<4}  {label:<12}  {params=!r}  {error=:.4f}")

    # plotting is now handled by Train pipeline via config.plot


if __name__ == "__main__":
    main()
