"""
Threshold calculation strategies for volume and dollar bars.

This module provides two approaches for dynamically calculating thresholds:
1. ADV-based thresholds that target a specific number of bars per trading day
2. Dynamic smoothing using exponential moving averages to reduce day-to-day jumps
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from alpaca.data.models.bars import Bar as TimeBar

from .bars import Conversion


class ThresholdStrategy(Enum):
  """Enumeration of available threshold calculation methods."""

  ADV = "adv"
  DES = "dynamic"


@dataclass
class Stats:
  """Container for daily trading statistics."""

  date: datetime
  total_volume: float
  avg_price: float


class Threshold(ABC):
  """Abstract base class for threshold calculation strategies."""

  @abstractmethod
  def threshold(
    self,
    symbol: str,
    time_bars: list[TimeBar],
    target_bars_per_day: int,
  ) -> float:
    """
    Calculate threshold for a given symbol.

    Args:
        symbol: The ticker symbol
        historical_bars: Historical time bars for threshold calculation
        target_bars_per_day: Target number of bars to generate per trading day

    Returns:
        Threshold value as int (volume) or float (dollar)
    """

  @classmethod
  def daily_stats(
    cls,
    conversion: Conversion,
    time_bars: list[TimeBar],
  ) -> list[Stats]:
    """
    Calculate daily statistics from time bars.

    Args:
        bars: List of time bars sorted by timestamp

    Returns:
        List of daily statistics
    """
    stat_index: dict[str, Stats] = {}

    for time_bar in time_bars:
      date_key = time_bar.timestamp.strftime("%Y-%m-%d")
      date = stat_index[date_key] = Stats(
        date=time_bar.timestamp,
        total_volume=0.0,
        avg_price=0.0,
      )
      midpoint_price = (time_bar.open + time_bar.close) / 2.0
      dollar_volume = time_bar.volume * midpoint_price

      match conversion:
        case Conversion.VOLUME:
          date.total_volume += time_bar.volume
        case Conversion.DOLLAR:
          date.total_volume += dollar_volume

      date.avg_price = dollar_volume / date.total_volume

    daily_stats = list(stat_index.values())

    return daily_stats


class ADV(Threshold):
  """
  Target "Bars per Day" via Average Daily Volume (ADV).

  Decides on a target number of bars, N, per "trading day."
  For each ticker, computes its ADV over some look-back window (say 20 days).
  Sets:
    - volume_bar_threshold = ADV / N
    - dollar_bar_threshold = (ADV * avg_price) / N

  This way, highly liquid names (high ADV) get larger thresholds and less
  liquid names get smaller thresholds, but you'll still, on average,
  generate about N bars each day.
  """

  def __init__(
    self,
    conversion: Conversion,
    lookback_days: int = 20,
  ):
    """
    Initialize the ADV threshold calculator.

    Args:
        lookback_days: Number of days to look back for ADV calculation
    """
    self.conversion = conversion
    self.lookback_days = lookback_days

  def threshold(
    self,
    symbol: str,
    time_bars: list[TimeBar],
    target_bars_per_day: int,
  ) -> float:
    """
    Calculate threshold based on ADV.

    Formula:
    - volume_threshold = ADV / target_bars_per_day
    - dollar_threshold = (ADV * avg_price) / target_bars_per_day
    """
    if target_bars_per_day <= 0:
      raise ValueError(
        f"`target_bars_per_day` must be positive, got {target_bars_per_day}"
      )

    daily_stats = Threshold.daily_stats(self.conversion, time_bars)
    sorted_stats = sorted(daily_stats, key=lambda x: x.date, reverse=True)
    recent_stats = sorted_stats[: self.lookback_days]
    total_volume = sum(stat.total_volume for stat in recent_stats)
    average_volume = total_volume / len(recent_stats)
    volume_threshold = int(average_volume / target_bars_per_day)
    return volume_threshold


class DES(Threshold):
  """
  Dynamic Exponential Smoothing of the Threshold.

  rollVol = α * todayVolume + (1-α) * rollVol_prev
  rollDollars = α * (todayVolume * todayPrice) + (1-α) * rollD_prev

  volumeThreshold = k * rollVol
  dollarThreshold = k * rollDollars

  Where k is a scaling factor to target the desired number of bars per day.
  """

  def __init__(
    self,
    conversion: Conversion,
    alpha: float = 0.1,
    scaling_factor: float = 0.1,
  ):
    """
    Initialize the dynamic smoothing threshold calculator.

    Args:
        alpha: Smoothing parameter (0 < alpha <= 1). Higher values give more
              weight to recent observations.
        scaling_factor: Factor to convert smoothed daily volume/dollar volume
                      to threshold values.
    """
    if not (0 < alpha <= 1):
      raise ValueError("alpha must be between 0 and 1")

    self.conversion = conversion
    self.alpha = alpha
    self.scaling_factor = scaling_factor

    # Storage for smoothed values per symbol
    self._smoothed_volume: dict[str, float] = {}
    self._smoothed_dollar_volume: dict[str, float] = {}

  def _smoothed_values(
    self,
    symbol: str,
    time_bars: list[TimeBar],
  ) -> float:
    """
    Calculate exponentially smoothed volume and dollar volume.

    Args:
        symbol: The ticker symbol
        historical_bars: Historical time bars sorted by timestamp

    Returns:
        Tuple of (smoothed_volume, smoothed_dollar_volume)
    """
    daily_stats = Threshold.daily_stats(
      conversion=self.conversion,
      time_bars=time_bars,
    )
    sorted_stats = sorted(daily_stats, key=lambda x: x.date)
    smoothed_vol = sorted_stats[0].total_volume

    for stat in sorted_stats[1:]:
      smoothed_vol = (
        self.alpha * stat.total_volume + (1 - self.alpha) * smoothed_vol
      )

    self._smoothed_volume[symbol] = smoothed_vol
    return smoothed_vol

  def threshold(
    self,
    symbol: str,
    time_bars: list[TimeBar],
    target_bars_per_day: int,
  ) -> float:
    """
    Calculate threshold using exponential smoothing.

    Formula:
    - volume_threshold = scaling_factor * smoothed_volume
      / target_bars_per_day
    - dollar_threshold = scaling_factor * smoothed_dollar_volume
      / target_bars_per_day
    """
    if target_bars_per_day <= 0:
      raise ValueError("target_bars_per_day must be positive")

    smoothed_vol = self._smoothed_values(symbol, time_bars)
    threshold = self.scaling_factor * smoothed_vol / target_bars_per_day
    return threshold

  def update_smoothed_value(
    self,
    symbol: str,
    new_volume: float,
  ) -> None:
    if symbol not in self._smoothed_volume:
      self._smoothed_volume[symbol] = new_volume
      return

    self._smoothed_volume[symbol] = (
      self.alpha * new_volume + (1 - self.alpha) * self._smoothed_volume[symbol]
    )

  def get_smoothed_value(self, symbol: str) -> float:
    return self._smoothed_volume.get(symbol, 0.0)
