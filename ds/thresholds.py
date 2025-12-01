"""
Threshold calculation strategies for volume and dollar bars.

This module provides two approaches for dynamically calculating thresholds:
1. ADV-based thresholds that target a specific number of bars per trading day
2. Dynamic smoothing using exponential moving averages to reduce day-to-day jumps
"""

from enum import Enum

from alpaca.data.models.bars import Bar as TimeBar

from .bars import Conversion


class Threshold(Enum):
    ADV = "adv"
    EMA = "ema"

    @classmethod
    def adv(
        cls,
        time_bars: list[TimeBar],
        conversion: Conversion,
        target_bars_per_day: int = 50,
    ) -> float:
        """
        Calculate threshold based on Average Daily Volume (ADV) to target a specific number of bars per trading day.

        Args:
            time_bars: Historical time bars to calculate ADV from
            conversion: Type of conversion (VOLUME or DOLLAR)
            target_bars_per_day: Desired number of bars per trading day
            trading_hours_per_day: Number of trading hours per day (default 6.5 for US markets)

        Returns:
            Calculated threshold for the specified conversion type
        """
        if not time_bars:
            raise ValueError("`time_bars` cannot be empty")

        if target_bars_per_day <= 0:
            raise ValueError("`target_bars_per_day` must be positive")

        # Group bars by trading date
        daily_values = {}

        for time_bar in time_bars:
            date = time_bar.timestamp.date()

            if date not in daily_values:
                daily_values[date] = 0.0

            match conversion:
                case Conversion.DOLLAR:
                    midpoint_price = (time_bar.open + time_bar.close) / 2.0
                    daily_values[date] += time_bar.volume * midpoint_price
                case Conversion.VOLUME:
                    daily_values[date] += time_bar.volume

        if not daily_values:
            raise ValueError("No data available")

        adv = sum(daily_values.values()) / len(daily_values)
        return adv / target_bars_per_day

    @classmethod
    def ema(
        cls,
        time_bars: list[TimeBar],
        conversion: Conversion,
        span: int = 20,
        target_bars_per_day: int = 50,
    ) -> float:
        """
        Calculate threshold based on Exponential Moving Average (EMA) of daily volumes/dollar volumes.

        Args:
            time_bars: Historical time bars to calculate EMA from
            conversion: Type of conversion (VOLUME or DOLLAR)
            span: Number of periods for EMA calculation (default 20)
            target_bars_per_day: Desired number of bars per trading day

        Returns:
            Calculated threshold based on EMA for the specified conversion type
        """
        if not time_bars:
            raise ValueError("`time_bars` cannot be empty")

        if target_bars_per_day <= 0:
            raise ValueError("`target_bars_per_day` must be positive")

        if span <= 0:
            raise ValueError("`span` must be positive")

        # Group bars by trading date
        daily_values = {}

        for time_bar in time_bars:
            date = time_bar.timestamp.date()

            if date not in daily_values:
                daily_values[date] = 0.0

            match conversion:
                case Conversion.VOLUME:
                    daily_values[date] += time_bar.volume
                case Conversion.DOLLAR:
                    midpoint_price = (time_bar.open + time_bar.close) / 2.0
                    daily_values[date] += time_bar.volume * midpoint_price

        if not daily_values:
            raise ValueError("No data available")

        # Sort daily values by date to ensure chronological order
        sorted_dates = sorted(daily_values.keys())
        sorted_values = [daily_values[date] for date in sorted_dates]

        # if len(sorted_values) < span:
        #   avg = sum(sorted_values) / len(sorted_values)
        #   return avg

        alpha = 2.0 / (span + 1)
        ema = sorted_values[0]  # Initialize with first value

        for value in sorted_values[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema / target_bars_per_day
