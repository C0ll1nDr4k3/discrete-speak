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
from typing import Any

from alpaca.data.models.bars import Bar as TimeBar

from ..config import Config
from ..bars import Conversion


class ThresholdStrategy(Enum):
    """Enumeration of available threshold calculation methods."""

    ADV = "adv"
    DYNAMIC = "dynamic"


@dataclass
class Stats:
    """Container for daily trading statistics."""

    date: datetime
    total_volume: float
    avg_price: float


class Threshold(ABC):
    """
    Abstract base class for threshold calculation strategies.
    """

    @abstractmethod
    def threshold(
        self,
        symbol: str,
        historical_bars: list[TimeBar],
        target_bars_per_day: int,
    ) -> int | float:
        """
        Calculate threshold for a given symbol.

        Args:
            symbol: The ticker symbol
            historical_bars: Historical time bars for threshold calculation
            target_bars_per_day: Target number of bars to generate per trading day
            for_volume: True for volume threshold, False for dollar threshold

        Returns:
            Threshold value as int (volume) or float (dollar)
        """

    @classmethod
    def from_config(cls, config: Config) -> "Threshold":
        """
        Create threshold calculators based on method enum.

        Args:
            method: ThresholdMethod enum value
            **kwargs: Additional parameters for the chosen method

        Returns:
            ThresholdCalculator instance
        """
        match config.threshold_strategy:
            case ThresholdStrategy.ADV:
                return ADV(
                    conversion=config.conversion,
                    lookback_days=config.threshold_adv_lookback_days,
                )
            case ThresholdStrategy.DYNAMIC:
                return DynamicSmoothing(
                    conversion=conversion,
                    alpha=config.threshold_des_alpha,
                    scaling_factor=config.threshold_des_scaling_factor,
                )


def daily_stats(conversion: Conversion, time_bars: list[TimeBar]) -> list[Stats]:
    """
    Calculate daily statistics from time bars.

    Args:
        bars: List of time bars sorted by timestamp

    Returns:
        List of daily statistics
    """
    if not time_bars:
        return []

    stats: dict[str, Stats] = {}

    for time_bar in time_bars:
        date_key = time_bar.timestamp.strftime("%Y-%m-%d")

        date = stats.get(
            date_key,
            Stats(
                date=time_bar.timestamp,
                total_volume=0.0,
                avg_price=0.0,
            ),
        )

        midpoint_price = (time_bar.open + time_bar.close) / 2.0
        dollar_volume = time_bar.volume * midpoint_price

        match conversion:
            case Conversion.VOLUME:
                date.total_volume += time_bar.volume
            case Conversion.DOLLAR:
                date.total_volume += dollar_volume

        date.avg_price = dollar_volume / date.total_volume

    return list(stats.values())
