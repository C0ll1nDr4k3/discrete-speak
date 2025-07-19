from ..config import Conversion
from ..bars import TimeBar
from threshold import Threshold


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
        historical_bars: list[TimeBar],
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

        daily_stats = self._daily_stats(historical_bars)

        # Sort by date and take the most recent lookback_days
        sorted_stats = sorted(daily_stats, key=lambda x: x.date, reverse=True)
        recent_stats = sorted_stats[: self.lookback_days]

        match self.conversion:
            case Conversion.VOLUME:
                total_volume = sum(stat.total_volume for stat in recent_stats)
                average_volume = total_volume / len(recent_stats)
                volume_threshold = int(average_volume / target_bars_per_day)
                return max(volume_threshold, 1000)
            case Conversion.DOLLAR:
                total_volume = sum(stat.total_dollar_volume for stat in recent_stats)
                average_volume = total_volume / len(recent_stats)
                dollar_threshold = average_volume / target_bars_per_day
                return max(dollar_threshold, 10_000.0)
            case Conversion.TIME:
                return 0.0
