from dataclasses import dataclass
from datetime import datetime
from typing import Any, final

from alpaca.data.timeframe import TimeFrame

from .bars import Conversion
from .thresholds import ADV, DES, Threshold, ThresholdStrategy


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

  # Segmentation
  segmentation_model: str = "l2"
  segmentation_penalty: float = 10.0
  segmentation_min_size: int = 5
  segmentation_jump: int = 1

  # Threshold
  threshold_strategy: ThresholdStrategy = ThresholdStrategy.ADV
  threshold_target_bars_per_day: int = 1_000
  threshold_adv_lookback_days: int = 20
  threshold_des_alpha: float = 0.1
  threshold_des_scaling_factor: float = 0.1

  threshold_static_dollar_threshold: float = 100_000.0
  threshold_staic_volume_threshold: int = 10_000
  threshold_static_time_threshold: int = 0

  def threshold_from(self) -> Threshold:
    """
    Create threshold calculators based on method enum.

    Args:
        method: ThresholdMethod enum value
        **kwargs: Additional parameters for the chosen method

    Returns:
        ThresholdCalculator instance
    """
    match self.threshold_strategy:
      case ThresholdStrategy.ADV:
        return ADV(
          conversion=self.conversion,
          lookback_days=self.threshold_adv_lookback_days,
        )
      case ThresholdStrategy.DES:
        return DES(
          conversion=self.conversion,
          alpha=self.threshold_des_alpha,
          scaling_factor=self.threshold_des_scaling_factor,
        )
