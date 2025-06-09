from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from scipy.signal import savgol_filter
from discrete_speak.bars import DollarBar, VolumeBar
from discrete_speak.retrieval import Alpaca, Security
from discrete_speak.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

config = Config(start=datetime.now() - timedelta(days=1), step=TimeFrame.Minute)

# {symbol: timeseries}
index = Alpaca.historical(
    symbols=[
        "META",
        # "AAPL",
        # "GOOG",
        # "GOOGL",
        # "NVDA",
        # "TSM",
        # "ORCL",
        # "PLTR",
        # "XOM",
        # "RY",
        # "DB",
        # "JPM",
    ],
    # symbols=["AAPL"],
    start=config.start,
    end=config.end,
    step=config.step,
    security=Security.STOCKS,
)

volume_index: dict[str, list[DollarBar]] = {
    # security: VolumeBar.from_alpaca(time_series, volume_threshold=250_000)
    security: DollarBar.from_alpaca(time_series, dollar_threshold=1_000_000)
    for security, time_series in index.items()
}

for security, volume_bars in volume_index.items():
    closing_prices = [volume_bar.close for volume_bar in volume_bars]
    sns.lineplot(closing_prices, alpha=0.5)
    # Apply the Savitzky-Golay filter to the _closing_ prices
    smoothed_prices = savgol_filter(closing_prices, window_length=25, polyorder=2)
    sns.lineplot(smoothed_prices, alpha=0.75, label=security)

plt.show()
