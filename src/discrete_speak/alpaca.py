from enum import Enum
from typing import List, Dict, Any
from os import environ
from datetime import datetime
from alpaca.data.timeframe import TimeFrame
from alpaca.data import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
    OptionHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest, OptionBarsRequest
from alpaca.data.models.bars import Bar, BarSet


class Security(Enum):
    Crypto = "crypto"
    Stocks = "stocks"
    Options = "options"


def stream():
    # alpaca_websocket_key = environ.get("ALPACA_WEBSOCKET_KEY")
    # alpaca_websocket_secret = environ.get("ALPACA_WEBSOCKET_SECRET")
    ...


def api(
    symbols: List[str],
    start: datetime,
    end: datetime,
    step: TimeFrame,
    security: Security
) -> Dict[str, List[Bar]]:
    assert isinstance(security, Security), f"Invalid security type: {security}"

    alpaca_api_key = environ.get("ALPACA_API_KEY")
    alpaca_api_secret = environ.get("ALPACA_API_SECRET")

    match security:
        case Security.Crypto:
            client = CryptoHistoricalDataClient(
                api_key=alpaca_api_key, secret_key=alpaca_api_secret
            )
            request = CryptoBarsRequest(
                symbol_or_symbols=symbols, start=start, end=end, timeframe=step
            )
            response: BarSet | Dict[str, List[Any]] = client.get_crypto_bars(request)
        case Security.Stocks:
            client = StockHistoricalDataClient(
                api_key=alpaca_api_key, secret_key=alpaca_api_secret
            )
            request = StockBarsRequest(
                symbol_or_symbols=symbols, start=start, end=end, timeframe=step
            )
            response: BarSet | Dict[str, List[Any]] = client.get_stock_bars(request)
        case Security.Options:
            client = OptionHistoricalDataClient(
                api_key=alpaca_api_key, secret_key=alpaca_api_secret
            )
            request = OptionBarsRequest(
                symbol_or_symbols=symbols, start=start, end=end, timeframe=step
            )
            response: BarSet | Dict[str, List[Any]] = client.get_option_bars(request)

    assert isinstance(response, BarSet), (
        f"Expected a BarSet instance from Alpaca, got {type(response)}."
    )
    return response.data
