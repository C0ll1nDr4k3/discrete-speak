from datetime import datetime, timedelta

from alpaca.data.timeframe import TimeFrame
from ds.retrieval import Alpaca, Security


def test_api():
    """
    A basic test to minimally ensure that the API is working correctly.
    """
    symbols = ["AAPL"]
    start = datetime.now() - timedelta(days=1)

    _time_series = Alpaca.historical(
        symbols=symbols,
        start=start,
        end=datetime.now(),
        step=TimeFrame.Day,
        security=Security.STOCKS,
    )
