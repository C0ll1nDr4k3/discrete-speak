import math
from dataclasses import dataclass
from enum import Enum

from alpaca.data.models import Bar as TimeBar


@dataclass(frozen=True, kw_only=True)
class VolumeBar:
    """
    Represents a bar aggregated based on a volume threshold.

    OHLC determined by:
    - Open: Open of the first constituent time bar.
    - High: Highest high among constituent time bars.
    - Low: Lowest low among constituent time bars.
    - Close: Close of the last constituent time bar.
    """

    open: float
    close: float
    high: float
    low: float

    @classmethod
    def from_alpaca(
        cls,
        time_bars: list[TimeBar],
        volume_threshold: float,
    ) -> list["VolumeBar"]:
        """
        Volume bars are created by aggregating values until `volume_threshold`
        has been reached.

        # Usage
        ---
        - Time series processed with `.from_alpaca` should NOT be strung together, as bars below `volume_threshold`
        will be omitted where they could have otherwise been included were the two sequences unified.
            - Wrong: `.from_alpaca([...]) + .from_alpaca([...])`
            - Right: `.from_alpaca([...] + [...])`
        """
        if volume_threshold <= 0:
            raise ValueError("`volume_threshold` must be positive.")

        volume_bars: list["VolumeBar"] = []
        accumulated_volume = 0.0  # Alpaca volume is float

        # subinterval = "constituent bars for current volume bar"
        subinterval = []

        for time_bar in time_bars:
            bar_volume: float = time_bar.volume
            subinterval.append(time_bar)

            if accumulated_volume + bar_volume < volume_threshold:
                accumulated_volume += bar_volume
                continue

            # If the it's over the threshold, form a new bar
            vb_open = subinterval[0].open
            vb_high = max(tb.high for tb in subinterval)
            vb_low = min(tb.low for tb in subinterval)
            vb_close = subinterval[-1].close  # which is time_bar.close

            # Assuming time_bar.timestamp is start of its interval
            # vb_timestamp = constituent_bars_for_current_volume_bar[
            #     -1
            # ].timestamp + timedelta(minutes=1)

            volume_bars.append(
                cls(
                    open=vb_open,
                    high=vb_high,
                    low=vb_low,
                    close=vb_close,
                )
            )

            # Reset for the next VolumeBar
            accumulated_volume = 0.0
            subinterval = []

        return volume_bars


@dataclass(frozen=True, kw_only=True)
class DollarBar:
    """
    Represents a bar aggregated based on a dollar volume threshold.

    The OHLC logic is as follows.
    - Open: Open price of the time bar that triggers/completes the dollar bar.
    - High: Maximum high price encountered from the start of accumulation up to and including the triggering time_bar.
    - Low: Lowest low price encountered similarly.
    - Close: Close price of the time bar that triggers/completes the dollar bar.
    """

    open: float
    close: float
    high: float
    low: float

    @classmethod
    def from_alpaca(
        cls,
        time_bars: list[TimeBar],
        dollar_threshold: float,
    ) -> list["DollarBar"]:
        """
        Dollar bars are created by aggregating values until `dollar_threshold`
        has been reached.

        Once it has, the cycle repeats until the threshold can no longer be met.

        # Usage
        ---
        - Time series processed with `.from_alpaca` should NOT be strung together, as bars below `dollar_threshold`
        will be omitted where they could have otherwise been included were the two sequences unified.
            - Wrong: `.from_alpaca([...]) + .from_alpaca([...])`
            - Right: `.from_alpaca([...] + [...])`
        """
        if dollar_threshold <= 0:
            raise ValueError("`dollar_threshold` must be positive")

        dollar_bars: list["DollarBar"] = []
        running_dollar_volume: float = 0.0

        # Tracks High/Low for the dollar bar currently being accumulated
        current_bar_agg_high: float = 0.0
        current_bar_agg_low: float = math.inf

        for time_bar in time_bars:
            # tb_timestamp: datetime = time_bar.timestamp
            midpoint_price = (time_bar.open + time_bar.close) / 2.0
            dollar_volume_this_bar = time_bar.volume * midpoint_price

            # Update running high/low for the potential dollar bar
            if (
                running_dollar_volume == 0.0
            ):  # First tick for this potential dollar bar
                current_bar_agg_high = time_bar.high
                current_bar_agg_low = time_bar.low
            else:
                current_bar_agg_high = max(current_bar_agg_high, time_bar.high)
                current_bar_agg_low = min(current_bar_agg_low, time_bar.low)

            if (
                running_dollar_volume + dollar_volume_this_bar
                >= dollar_threshold
            ):
                # Dollar bar is completed by this time_bar
                # dollar_bar_close_timestamp = tb_timestamp + timedelta(
                #     minutes=1
                # )  # Assuming 1-min input bars

                dollar_bars.append(
                    cls(
                        open=time_bar.open,  # Open of the triggering bar
                        high=current_bar_agg_high,  # Accumulated High
                        low=current_bar_agg_low,  # Accumulated Low
                        close=time_bar.close,  # Close of the triggering bar
                    )
                )

                # Reset for the next dollar bar
                running_dollar_volume = 0.0
                current_bar_agg_high = 0.0
                current_bar_agg_low = math.inf
            else:
                running_dollar_volume += dollar_volume_this_bar
                # current_bar_agg_high and _low carry over their updated values

        return dollar_bars


Bar = TimeBar | VolumeBar | DollarBar


class Conversion(Enum):
    VOLUME = "volume"
    DOLLAR = "dollar"
