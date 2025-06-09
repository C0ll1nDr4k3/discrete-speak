from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time, timedelta
from colorama import Fore
import os
import warnings
from zoneinfo import ZoneInfo
from alpaca.data import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
    OptionHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest, OptionBarsRequest
from alpaca.data.models.bars import Bar as TimeBar, BarSet
from dotenv import load_dotenv
from typing import Any, TypeAlias


AlpacaResponse: TypeAlias = BarSet | dict[str, list[TimeBar]]


class Security(Enum):
    CRYPTO = "CRYPTO"
    STOCKS = "STOCKS"
    OPTIONS = "OPTIONS"


@dataclass(frozen=True, kw_only=True)
class Alpaca:
    ...

    @classmethod
    def real_time(cls):
        # alpaca_websocket_key = environ.get("ALPACA_WEBSOCKET_KEY")
        # alpaca_websocket_secret = environ.get("ALPACA_WEBSOCKET_SECRET")
        ...

    @classmethod
    def historical(
        cls,
        *,
        symbols: list[str],
        start: datetime,
        end: datetime,
        step: Any,
        security: Security,
    ) -> dict[str, list[TimeBar]]:
        """
        Retrieves from [Alpaca's Historical Data API](https://docs.alpaca.markets/docs/historical-api).

        ## Args
        ---
        - `step`: `step` is untyped on account of Alpaca's `TimeFrame` defining its intervals
        (`.Minute`, `.Day`, etc.) as `classproperty` methods instead of proper class properties. :)
        """
        assert load_dotenv(), "Environment couldn't be loaded."
        alpaca_api_key = os.getenv("ALPACA_API_KEY")
        alpaca_api_secret = os.getenv("ALPACA_API_SECRET")

        _within_hours(start=start, end=end)

        response: AlpacaResponse
        match security:
            case Security.CRYPTO:
                client = CryptoHistoricalDataClient(
                    api_key=alpaca_api_key, secret_key=alpaca_api_secret
                )
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbols,
                    start=start,
                    end=end,
                    timeframe=step,
                )
                response = client.get_crypto_bars(request)
            case Security.STOCKS:
                client = StockHistoricalDataClient(
                    api_key=alpaca_api_key, secret_key=alpaca_api_secret
                )
                request = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    start=start,
                    end=end,
                    timeframe=step,
                )
                response = client.get_stock_bars(request)
            case Security.OPTIONS:
                client = OptionHistoricalDataClient(
                    api_key=alpaca_api_key, secret_key=alpaca_api_secret
                )
                request = OptionBarsRequest(
                    symbol_or_symbols=symbols,
                    start=start,
                    end=end,
                    timeframe=step,
                )
                response = client.get_option_bars(request)

        assert isinstance(response, BarSet), (
            f"Expected a BarSet instance from Alpaca, got {type(response)}."
        )
        return response.data


def _within_hours(*, start: datetime, end: datetime) -> bool:
    """
    Alpaca returns NEITHER after-hours data NOR weekend data.
    Normal market hours comprise 9:30 AM to 4:00 PM EST, Monday through Friday.
    If the provided date range does not contain any valid market hours,
    the user will be warned.
    """
    # Define eastern timezone
    est = ZoneInfo("US/Eastern")

    # Market hours: 9:30 AM to 4:00 PM EST
    market_start_time = time(9, 30)
    market_end_time = time(16, 0)

    # Convert to eastern if not timezone-aware, or if in different timezone
    start = start.astimezone(est) if start.tzinfo else start.replace(tzinfo=est)
    end = end.astimezone(est) if end.tzinfo else end.replace(tzinfo=est)

    # Check if any part of the date range falls within market hours
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    overlaps = False

    while current.date() <= end.date():
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current.weekday() < 5:  # Monday through Friday
            # Check if the current day overlaps with market hours
            day_start = current.replace(
                hour=market_start_time.hour,
                minute=market_start_time.minute,
                second=0,
                microsecond=0,
            )
            day_end = current.replace(
                hour=market_end_time.hour,
                minute=market_end_time.minute,
                second=0,
                microsecond=0,
            )

            # Check if there's any overlap between the requested range and market hours for this day
            if not (end < day_start or start > day_end):
                overlaps = True
                break

        # Move to next day
        current += timedelta(days=1)

    if not overlaps:
        message = (
            f"{Fore.RED}The provided date range ({start.strftime('%a %I:%M %p %Z')} to "
            f"{end.strftime('%a %I:%M %p %Z')}) does not contain any valid "
            f"market hours. Market hours are 9:30 AM to 4:00 PM {start.tzname()}, Monday through Friday."
        )
        warnings.warn(message)

    return overlaps
