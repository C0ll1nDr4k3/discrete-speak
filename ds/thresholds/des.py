from ..config import Config, Conversion
from ..bars import TimeBar
from threshold import Stats, Threshold, daily_stats


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
        config: Config,
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
        self.config = config
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
        # Group bars by day and calculate daily totals
        daily_stats = daily_stats(
            conversion=self.conversion,
            bars=time_bars,
        )

        # Sort by date to process chronologically
        sorted_stats: list[Stats] = sorted(daily_stats, key=lambda x: x.date)

        # Initialize with first day's values
        smoothed_vol = sorted_stats[0].total_volume

        # Apply exponential smoothing
        for stat in sorted_stats[1:]:
            smoothed_vol = (
                self.alpha * stat.total_volume + (1 - self.alpha) * smoothed_vol
            )

        # Cache the smoothed values
        self._smoothed_volume[symbol] = smoothed_vol

        return smoothed_vol

    def threshold(
        self,
        symbol: str,
        historical_bars: list[TimeBar],
        target_bars_per_day: int,
    ) -> int | float:
        """
        Calculate threshold using exponential smoothing.

        Formula:
        - volume_threshold = scaling_factor * smoothed_volume / target_bars_per_day
        - dollar_threshold = scaling_factor * smoothed_dollar_volume / target_bars_per_day
        """
        if target_bars_per_day <= 0:
            raise ValueError("target_bars_per_day must be positive")

        smoothed_vol, smoothed_dollar_vol = self._smoothed_values(
            symbol, historical_bars
        )

        match self.conversion:
            case Conversion.VOLUME:
                volume_threshold = int(
                    self.scaling_factor * smoothed_vol / target_bars_per_day
                )
                # Ensure minimum threshold
                return max(volume_threshold, 1000)
            case Conversion.DOLLAR:
                dollar_threshold = (
                    self.scaling_factor * smoothed_dollar_vol / target_bars_per_day
                )
                # Ensure minimum threshold
                return max(dollar_threshold, 10_000.0)
            case Conversion.TIME:
                # For time-based "conversion", we don't apply a threshold
                return 0.0

    def update_smoothed_values(
        self,
        symbol: str,
        new_volume: float,
        new_dollar_volume: float,
    ) -> None:
        """
        Update the smoothed values with new daily data.

        This method allows for real-time updates without recalculating from scratch.

        Args:
            symbol: The ticker symbol
            new_volume: New daily volume
            new_dollar_volume: New daily dollar volume
        """
        if symbol in self._smoothed_volume:
            self._smoothed_volume[symbol] = (
                self.alpha * new_volume
                + (1 - self.alpha) * self._smoothed_volume[symbol]
            )
        else:
            self._smoothed_volume[symbol] = new_volume

        if symbol in self._smoothed_dollar_volume:
            self._smoothed_dollar_volume[symbol] = (
                self.alpha * new_dollar_volume
                + (1 - self.alpha) * self._smoothed_dollar_volume[symbol]
            )
        else:
            self._smoothed_dollar_volume[symbol] = new_dollar_volume

    def get_smoothed_values(self, symbol: str) -> tuple[float, float]:
        """
        Get the current smoothed values for a symbol.

        Args:
            symbol: The ticker symbol

        Returns:
            Tuple of (smoothed_volume, smoothed_dollar_volume)
        """
        return (
            self._smoothed_volume.get(symbol, 0.0),
            self._smoothed_dollar_volume.get(symbol, 0.0),
        )
