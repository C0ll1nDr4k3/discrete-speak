import statistics as stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

from alpaca.data import TimeFrame
from ds.retrieval import Alpaca, Security

# Configuration
CUTOFF_RATIO = 0.6  # Use 60% of data for initial distribution
# NUM_GENERATED will be calculated dynamically based on remaining data
WEIGHT_DECAY = 0.97  # Exponential decay factor for weighting recent samples

securities = Alpaca.historical(
    ["NVDA"],
    start=datetime.now() - timedelta(days=2),
    end=datetime.now(),
    step=TimeFrame.Minute,
    security=Security.EQUITIES,
)

for symbol, time_bars in securities.items():
    closing_prices = [time_bar.close for time_bar in time_bars]
    timestamps = [time_bar.timestamp for time_bar in time_bars]

    # Calculate cutoff point
    cutoff_n = int(len(closing_prices) * CUTOFF_RATIO)

    # Calculate number of samples to generate (same as remaining actual data)
    num_generated = len(closing_prices) - cutoff_n

    # Split data at cutoff
    initial_prices = closing_prices[:cutoff_n]
    actual_remaining = closing_prices[cutoff_n:]

    def create_weighted_distribution(prices):
        """Create a weighted normal distribution from price samples"""
        n = len(prices)
        weights = np.array([WEIGHT_DECAY ** (n - 1 - i) for i in range(n)])
        weighted_mean = np.average(prices, weights=weights)
        weighted_variance = np.average(
            (np.array(prices) - weighted_mean) ** 2, weights=weights
        )
        weighted_stdev = np.sqrt(weighted_variance)
        return (
            stats.NormalDist(weighted_mean, weighted_stdev),
            weighted_mean,
            weighted_stdev,
        )

    # Generate initial distribution
    current_dist, mean, stdev = create_weighted_distribution(initial_prices)

    print(f"Initial weighted distribution stats for {symbol}:")
    print(f"  Weighted Mean: {mean:.2f}")
    print(f"  Weighted Standard Deviation: {stdev:.2f}")
    print(f"  Cutoff at index: {cutoff_n}")
    print(f"  Samples to generate: {num_generated}")
    print(f"  Weight decay factor: {WEIGHT_DECAY}")

    # Iteratively generate values and update distribution
    np.random.seed(42)  # For reproducibility
    generated_values = []
    current_prices = initial_prices.copy()

    for i in range(num_generated):
        # Generate next value from current distribution
        next_generated = np.random.normal(current_dist.mean, current_dist.stdev)
        generated_values.append(next_generated)

        # If we have actual data, add it to update the distribution
        if i < len(actual_remaining):
            actual_next = actual_remaining[i]
            current_prices.append(actual_next)
            # Update distribution with new actual data point
            current_dist, _, _ = create_weighted_distribution(current_prices)

            print(
                f"  Step {i + 1}: Generated={next_generated:.2f}, Actual={actual_next:.2f}, "
                f"New Mean={current_dist.mean:.2f}"
            )

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot original time series up to cutoff
    plt.plot(
        range(cutoff_n),
        initial_prices,
        "b-",
        label=f"Original {symbol} prices (training)",
        linewidth=2,
    )

    # Plot actual remaining values
    if actual_remaining:
        plt.plot(
            range(cutoff_n, len(closing_prices)),
            actual_remaining,
            "g-",
            label=f"Actual {symbol} prices (test)",
            # linewidth=2,
            alpha=0.5,
        )

    # Plot generated values
    generated_indices = range(cutoff_n, cutoff_n + num_generated)
    plt.plot(
        generated_indices,
        generated_values,
        "r--",
        label="Generated from normal dist",
        # linewidth=2,
        alpha=0.5,
    )

    # Plot cutoff line
    plt.axvline(
        x=cutoff_n,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Cutoff (n={cutoff_n})",
    )

    # Customize plot
    plt.title(
        f"{symbol} Price Analysis: Original vs Generated from Normal Distribution"
    )
    plt.xlabel("Time Index")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show statistics on plot
    textstr = f"Updating Weighted Dist:\nInitial Mean: ${mean:.2f}\nInitial Std Dev: ${stdev:.2f}\nDecay: {WEIGHT_DECAY}\nSamples: {num_generated}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.show()

    # Print some analysis
    print(f"\nGenerated {num_generated} new values")
    print(
        f"Generated values range: ${min(generated_values):.2f} - ${max(generated_values):.2f}"
    )

    if actual_remaining:
        actual_mean = stats.mean(actual_remaining)
        actual_stdev = (
            stats.stdev(actual_remaining) if len(actual_remaining) > 1 else 0
        )
        print(f"Actual remaining values mean: ${actual_mean:.2f}")
        print(f"Actual remaining values stdev: ${actual_stdev:.2f}")
        print(f"Difference in means: ${abs(actual_mean - mean):.2f}")
