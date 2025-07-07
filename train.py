from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame
from scipy.signal import savgol_filter
from discrete_speak.bars import DollarBar, VolumeBar
from discrete_speak.retrieval import Alpaca, Security
from discrete_speak.config import Config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ruptures as rpt  # Import the ruptures library


sns.set_theme(style="darkgrid")

config = Config(start=datetime.now() - timedelta(days=5), step=TimeFrame.Minute)

# {symbol: timeseries}
index = Alpaca.historical(
    symbols=[
        # "META",
        # "AAPL",
        # "GOOG",
        # "GOOGL",
        # "NVDA",
        # "TSM",
        # "ORCL",
        # "PLTR",
        # "XOM",
        # "RY",
        "DB",
        # "JPM",
    ],
    start=config.start,
    end=config.end,
    step=config.step,
    security=Security.STOCKS,
)

dollar_index: dict[str, list[DollarBar]] = {}
universe = np.array([])

# We're building two indexes with a single loop.
for security, time_bars in index.items():
    dollar_index[security] = DollarBar.from_alpaca(time_bars, dollar_threshold=1_000_000)
    closing_prices = np.array(time_bar.close for time_bar in time_bars)
    np.append(universe, closing_prices)

# algo = rpt.Binseg(model="l2").fit(universe)
# breakpoints = algo.predict(n_bkps=8, pen=10)
# rpt.display(universe, breakpoints)
# plt.show()

# dollar_index: dict[str, list[DollarBar]] = {
#     # security: VolumeBar.from_alpaca(time_series, volume_threshold=250_000)
#     security: DollarBar.from_alpaca(time_series, dollar_threshold=100_000_000)
#     for security, time_series in index.items()
# }


for security, volume_bars in dollar_index.items():
    closing_prices = [volume_bar.close for volume_bar in volume_bars]
    sns.lineplot(closing_prices, alpha=0.5)
    # Apply the Savitzky-Golay filter to the _closing_ prices
    smoothed_prices = savgol_filter(closing_prices, window_length=100, polyorder=2)
    sns.lineplot(smoothed_prices, alpha=0.75, label=security)

plt.show()
