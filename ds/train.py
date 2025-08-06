"""
Offline training pipeline tying retrieval, segmentation, and labeling,
exposed as a single `train(config)` function.

This module provides:
  - train(config): runs the full offline pipeline given an immutable `Config`.
"""

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

from .bars import Bar, Conversion, DollarBar, VolumeBar
from .config import Config
from .labeler import Labeler
from .retrieval import Alpaca, Security
from .segmenters import Ruptures
from .thresholds import Threshold


def train(
  symbols: list[str],
  security: Security,
  config: Config,
) -> dict[str, dict[str, Any]]:
  """
  Execute the full offline pipeline.

  1. Retrieve time bars from Alpaca.
  2. Segment the closing-price series.
  3. Label each segment with the best-fitting analytic curve.
  4. Optionally plot/save/show the results.

  Args:
      config: Immutable configuration containing all pipeline parameters,
              including .symbols (List[str]) and .security (Security).

  Returns:
      A dict mapping symbol -> {
          "breakpoints": List[int],
          "labels": List[dict],
      }
  """
  # Build default segmenter & labeler from config
  labeler = Labeler()

  # 1. Fetch raw time-series bars
  time_index = Alpaca.historical(
    symbols=symbols,
    start=config.start,
    end=config.end,
    step=config.step,
    security=security,
  )

  results: dict[str, dict[str, Any]] = {}

  for symbol, time_bars in time_index.items():
    threshold: float
    match config.threshold_strategy:
      case Threshold.ADV:
        threshold = Threshold.adv(
          time_bars=time_bars, conversion=config.conversion
        )
      case Threshold.EMA:
        threshold = Threshold.ema(
          time_bars=time_bars, conversion=config.conversion
        )

    conv_bars: Sequence[Bar]
    match config.conversion:
      case Conversion.DOLLAR:
        conv_bars = DollarBar.from_alpaca(
          time_bars=time_bars, dollar_threshold=threshold
        )
      case Conversion.VOLUME:
        conv_bars = VolumeBar.from_alpaca(
          time_bars=time_bars, volume_threshold=threshold
        )

    raw_prices = [tb.close for tb in conv_bars]
    breakpoints = Ruptures.window(
      raw_prices,
      width=config.segmentation_jump,
      model=config.segmentation_model,
      pen=config.segmentation_penalty,
      min_size=config.segmentation_min_size,
      jump=config.segmentation_jump,
    )
    # rescale breakpoints into converted-bar indices
    # breakpoints = [
    #   int(bp * len(conv_bars) / len(time_bars)) for bp in breakpoints
    # ]
    conv_prices = [b.close for b in conv_bars]
    smoothed = savgol_filter(
      conv_prices,
      window_length=config.sg_window_length,
      polyorder=config.sg_polyorder,
    )
    labels = labeler.label(smoothed, breakpoints)

    if config.plot_enabled:
      print(f"{breakpoints = }")
      sns.set_theme(style="darkgrid")
      sns.lineplot(conv_prices, alpha=0.5)
      sns.lineplot(smoothed, alpha=0.75, label=symbol)

      # Add transparent vertical lines at breakpoints
      for bp in breakpoints:
        plt.axvline(x=bp, alpha=0.3, linestyle='--', linewidth=1)

      if config.plot_save:
        outdir = os.path.join(config.plot_directory, config.conversion.value)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{symbol}.png"), dpi=config.plot_dpi)

      plt.clf()

    results[symbol] = {"breakpoints": breakpoints, "labels": labels}

  if config.plot_enabled and config.plot_show:
    plt.show()

  return results
