"""
Offline training pipeline tying retrieval, segmentation, and labeling.

This module provides an OfflinePipeline class that:
  1. Fetches time-series bars from Alpaca.
  2. Segments the closing-price series using a change-point detector.
  3. Labels each segment by fitting a small library of analytic curves.
"""

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from alpaca.data.models.bars import Bar as TimeBar
from scipy.signal import savgol_filter


from .bars import Conversion, DollarBar, VolumeBar, Bar
from .config import Config
from .labeler import Labeler
from .retrieval import Alpaca, Security
from .segmenters import PELT, Segmenter
from .thresholds import Threshold


class Train:
    """
    Offline training pipeline that orchestrates retrieval of raw time bars,
    segmentation of the price series, and labeling of each segment with the
    best-fitting analytic curve.
    """

    def __init__(
        self,
        config: Config,
        segmenter: Segmenter | None = None,
        labeler: Labeler | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        Args:
            config: Immutable configuration containing all pipeline parameters.
            segmenter: An instance implementing Segmenter interface.
                      Defaults to RupturesPELT with config parameters.
            labeler: An instance of CurveLabeler.
                     Defaults to the builtin CurveLabeler.
            threshold_calculator: An instance implementing ThresholdCalculator interface.
                                 Defaults to calculator based on config.threshold_method.
        """
        self.config = config
        # Use provided or fallback to default implementations with config
        self.segmenter: Segmenter = segmenter or PELT(
            model=config.segmentation.model,
            pen=config.segmentation.penalty,
            min_size=config.segmentation.min_size,
            jump=config.segmentation.jump,
        )
        self.encoder: Labeler = labeler or Labeler()
        self.thresholder = Threshold.from_config(config)

    def run(
        self,
        symbols: list[str],
        security: Security,
    ) -> dict[str, dict[str, Any]]:
        """
        Execute the full offline pipeline.

        1. Retrieve time bars from Alpaca.
        2. Extract closing prices.
        3. Compute change-point indices.
        4. Label each segment with the best-fitting analytic curve.

        Args:
            symbols: List of ticker symbols to retrieve.
            security: The security type (CRYPTO, STOCKS, or OPTIONS).

        Returns:
            A dict mapping each symbol to a dict with:
              - "breakpoints": List[int] of segment start indices.
              - "labels": List[Dict] of segment metadata, each containing:
                   * start (inclusive), end (exclusive)
                   * label (curve name), params, and mse error.
        """
        # 1. Fetch raw time-series bars
        time_index: dict[str, list[TimeBar]] = Alpaca.historical(
            symbols=symbols,
            start=self.config.start,
            end=self.config.end,
            step=self.config.step,
            security=security,
        )
        results: dict[str, dict[str, Any]] = {}

        for symbol, time_bars in time_index.items():
            conversion = self.config.conversion
            threshold = self.thresholder.threshold(
                symbol,
                time_bars,
                self.config.threshold_target_bars_per_day,
            )

            conv_bars: Sequence[Bar]
            match conversion:
                case Conversion.VOLUME:
                    conv_bars = VolumeBar.from_alpaca(
                        time_bars, volume_threshold=int(threshold)
                    )
                case Conversion.DOLLAR:
                    conv_bars = DollarBar.from_alpaca(
                        time_bars, dollar_threshold=threshold
                    )
                case Conversion.TIME:
                    conv_bars = time_bars

            # Extract closing prices as a simple float series
            time_closes = [time_bar.close for time_bar in time_bars]
            # Compute change-points on the price series
            # (segmentation is applied before any bar-type conversion)
            breakpoints = self.segmenter.predict(time_closes)

            # Adjust breakpoints to match dollar bar indices
            if conversion is not Conversion.TIME:
                breakpoints = [
                    int(breakpoint * len(conv_bars) / len(time_bars))
                    for breakpoint in breakpoints
                ]

            conv_closes = [conv_bar.close for conv_bar in conv_bars]
            smoothed_prices = savgol_filter(
                conv_closes,
                window_length=self.config.sg_window_length,
                polyorder=self.config.sg_polyorder,
            )

            # Label each segment with the best-fitting analytic curve
            labels = self.encoder.label(smoothed_prices, breakpoints)

            if self.config.plot.enabled:
                sns.lineplot(conv_closes, alpha=0.5)
                sns.lineplot(smoothed_prices, alpha=0.75, label=symbol)

                if self.config.plot.save:
                    path = f"{self.config.plot.directory}/{conversion}"
                    os.makedirs(path, exist_ok=True)
                    plt.savefig(
                        f"{path}/{symbol}.png",
                        dpi=self.config.plot.dpi,
                    )

            results[symbol] = {
                "breakpoints": breakpoints,
                "labels": labels,
            }

        if self.config.plot.show:
            plt.show()

        return results
