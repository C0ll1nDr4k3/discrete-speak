"""
Offline training pipeline tying retrieval, segmentation, and labeling,
exposed as a single `train(config)` function.

This module provides:
  - train(config): runs the full offline pipeline given an immutable `Config`.
"""

import os
from itertools import product
from typing import Any, Collection, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter

from .bars import Bar, Conversion, DollarBar, VolumeBar
from .config import Config
from .labeler import Labeler
from .retrieval import Alpaca, Security
from .segmenters import Ruptures
from .thresholds import Threshold


def fit(
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

    curves: dict[str, dict[str, Any]] = {}

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
                plt.axvline(x=bp, alpha=0.3, linestyle="--", linewidth=1)

            # Plot the fitted curves for each segment
            for label in labels:
                start, end, label, params = (
                    label["start"],
                    label["end"],
                    label["label"],
                    label["params"],
                )

                if label != "constant":
                    segment_x = np.arange(1, end - start + 1)
                    fitted_y = labeler.functions[label](segment_x, *params)
                    plt.plot(np.arange(start, end), fitted_y, linestyle="--")

            if config.plot_save:
                outdir = os.path.join(
                    config.plot_directory, config.conversion.value
                )
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(
                    os.path.join(outdir, f"{symbol}.png"), dpi=config.plot_dpi
                )

            plt.clf()

        curves[symbol] = {"breakpoints": breakpoints, "labels": labels}

    if config.plot_enabled and config.plot_show:
        plt.show()

    return curves


def discretize(
    curves: Collection[dict[str, Any]], radius: float, step: float
) -> Dict[str, List[List[float]]]:
    """
    Generates a discrete set of parameters for curve fitting.

    The set of parameters is generated for each curve type present in the
    input `curves`.

    Args:
        curves: A collection of dictionaries, where each dictionary represents
                a curve and contains a "label" key with the curve's name.
        radius: The radius for the discrete parameter range (-radius to +radius).
        step: The step size for the discrete parameter range.

    Returns:
        A dictionary mapping curve labels (e.g., "linear") to a list of
        discrete parameter combinations.
    """
    discrete_range = np.arange(-radius, radius + step, step)

    # Get unique curve labels from the input
    labels = set(
        curve["label"]
        for curve in curves
        if "label" in curve and curve["label"] != "constant"
    )

    labeler = Labeler()
    num_params_map = {name: len(p0) for name, p0 in labeler.p0.items()}

    discrete_params: Dict[str, List[List[float]]] = {}

    for label in labels:
        if label in num_params_map:
            num_params = num_params_map[label]
            # Generate all combinations
            param_combinations = product(discrete_range, repeat=num_params)
            discrete_params[label] = [list(p) for p in param_combinations]

    return discrete_params
