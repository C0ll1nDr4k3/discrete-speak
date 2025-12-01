from dataclasses import dataclass
from datetime import datetime
from typing import Any, final

from alpaca.data.timeframe import TimeFrame

from .bars import Conversion
from .thresholds import Threshold


@final
@dataclass(frozen=True, kw_only=True)
class Config:
    """
    This is ***stateless***.

    As such, it should hold only immutable configuration, not hyperparameters.
    """

    # Retrieval
    start: datetime
    end: datetime = datetime.now()
    step: Any = TimeFrame.Day

    # Smoothing
    sg_window_length: int = 60
    sg_polyorder: int = 2

    # Conversion
    conversion: Conversion = Conversion.DOLLAR

    # Plot
    plot_enabled: bool = False
    plot_show: bool = True
    plot_save: bool = True
    plot_directory: str = "plots"
    plot_dpi: int = 300

    plot_conv_sg: bool = False

    # Discretization
    discretization_log_min_power: int = -5
    discretization_log_max_power: int = 5
    discretization_points_per_magnitude: int = 20

    # Segmentation
    segmentation_model: str = "rbf"  # "l1", "l2", "rbf", or "linear"
    segmentation_penalty: float = 3.0
    segmentation_min_size: int = 4
    segmentation_jump: int = 5

    # Threshold
    threshold_strategy: Threshold = Threshold.ADV
    threshold_target_bars_per_day: int = 1_000
    threshold_adv_lookback_days: int = 20
    threshold_des_alpha: float = 0.1
    threshold_des_scaling_factor: float = 0.1

    threshold_static_dollar_threshold: float = 500_000_000.0
    threshold_staic_volume_threshold: int = 5_000_000
