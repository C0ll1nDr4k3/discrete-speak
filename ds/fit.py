"""
Offline training pipeline tying retrieval, segmentation, and labeling,
exposed as a single `train(config)` function.

This module provides:
  - train(config): runs the full offline pipeline given an immutable `Config`.
"""

import copy
import os
from itertools import product
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter

from .bars import Bar, Conversion, DollarBar, VolumeBar
from .config import Config
from .discretization import Discretizer
from .labeler import Labeler
from .retrieval import Alpaca, Security
from .segmenters import Ruptures
from .thresholds import Threshold


def discretize(
    labels: List[Dict[str, Any]],
    max_power: int,
    min_power: int,
    points_per_magnitude: int,
) -> List[Dict[str, Any]]:
    """
    Map curve parameters to a discrete, logarithmically spaced space.
    
    Delegates to the Discretizer class.
    """
    discretizer = Discretizer(min_power, max_power, points_per_magnitude)
    print(f"Number of parameters generated: {discretizer.vocab_size} for {discretizer.vocab_size * (2 + 3 + 4 + 2)} total curves.")
    return discretizer.discretize(labels)


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
            "discretized_labels": Optional[List[dict]],
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

        conv_prices_full = [b.close for b in conv_bars]
        
        # Chunking logic
        max_len = config.max_time_series_len
        if max_len is None:
            chunks = [(0, conv_prices_full)]
        else:
            chunks = []
            for i in range(0, len(conv_prices_full), max_len):
                chunks.append((i, conv_prices_full[i : i + max_len]))

        all_breakpoints = []
        all_labels = []

        for offset, chunk_prices in chunks:
            if len(chunk_prices) < config.segmentation_min_size:
                continue

            breakpoints = Ruptures.window(
                chunk_prices,
                width=config.segmentation_jump,
                model=config.segmentation_model,
                pen=config.segmentation_penalty,
                min_size=config.segmentation_min_size,
                jump=config.segmentation_jump,
            )
            
            # Adjust breakpoints to be relative to the full series
            global_breakpoints = [bp + offset for bp in breakpoints]
            all_breakpoints.extend(global_breakpoints)

            smoothed = savgol_filter(
                chunk_prices,
                window_length=min(config.sg_window_length, len(chunk_prices)), # Ensure window length is valid
                polyorder=config.sg_polyorder,
            )
            
            chunk_labels = labeler.label(smoothed, breakpoints)
            
            # Adjust label start/end to be relative to the full series
            for label in chunk_labels:
                label["start"] += offset
                label["end"] += offset
                
            all_labels.extend(chunk_labels)

        # Discretize all labels together
        all_labels = discretize(
            all_labels,
            config.discretization_log_max_power,
            config.discretization_log_min_power,
            config.discretization_points_per_magnitude,
        )

        result_for_symbol: Dict[str, Any] = {
            "breakpoints": all_breakpoints,
            "labels": all_labels,
        }

        if config.plot_enabled:
            print(f"{all_breakpoints = }")
            sns.set_theme(style="darkgrid")
            sns.lineplot(conv_prices_full, alpha=0.5)
            # Plotting smoothed is tricky with chunks, let's skip or plot segments
            # For simplicity, we won't plot the smoothed line for the full series if chunked, 
            # or we could reconstruct it.
            # sns.lineplot(smoothed, alpha=0.75, label=symbol) 

            # Add transparent vertical lines at breakpoints
            for bp in all_breakpoints:
                plt.axvline(x=bp, alpha=0.3, linestyle="--", linewidth=1)

            # Plot the fitted curves for each segment
            for label in all_labels:
                start, end, label_name, params = (
                    label["start"],
                    label["end"],
                    label["label"],
                    label["params"],
                )

                if label_name != "constant":
                    segment_x = np.arange(1, end - start + 1)
                    fitted_y = labeler.functions[label_name](segment_x, *params)
                    # We need to shift fitted_y to the correct vertical position?
                    # No, fit is on the values. But if we chunked, are values continuous?
                    # Yes, we chunked the prices directly.
                    # Wait, `labeler.label` fits curves to the data.
                    # If we plot `fitted_y` at `np.arange(start, end)`, it should overlay the prices.
                    plt.plot(np.arange(start, end), fitted_y, linestyle="--")

            if config.plot_save:
                outdir = os.path.join(
                    config.plot_directory, config.conversion.value
                )
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(
                    os.path.join(outdir, f"{symbol}.png"),
                    dpi=config.plot_dpi,
                )

            plt.clf()

        curves[symbol] = result_for_symbol

    if config.plot_enabled and config.plot_show:
        plt.show()

    return curves


def print_labels(symbols: dict[str, dict[str, Any]]):
    for symbol, data in symbols.items():
        print(f"\n=== {symbol} ===")

        labels = data["labels"]

        for i, seg in enumerate(labels):
            start, end = seg["start"], seg["end"]
            label, params, error = seg["label"], seg["params"], seg["error"]

            print_str = f"{start:>4}:{end:<4}  {label:<12}  {params=!r}  "

            print_str += f"{error=:.4f}"
            print(print_str)
