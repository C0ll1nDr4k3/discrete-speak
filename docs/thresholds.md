# Threshold Calculators

This module provides dynamic threshold calculation strategies for volume and dollar bars in the discrete-speak library. Instead of using fixed thresholds, these calculators adapt based on historical trading data to target a specific number of bars per trading day.

## Overview

The threshold calculators solve a common problem in financial time series analysis: how to set appropriate volume and dollar thresholds that adapt to different securities with varying liquidity profiles. A highly liquid stock like AAPL might need much higher thresholds than a less liquid stock to generate the same number of bars per day.

## Available Calculators

### 1. ADV-Based Threshold Calculator

The `ADVThresholdCalculator` targets a specific number of bars per trading day using Average Daily Volume (ADV).

**Algorithm:**

- Computes ADV over a configurable lookback window (default: 20 days)
- Sets thresholds as:
  - `volume_threshold = ADV ÷ target_bars_per_day`
  - `dollar_threshold = (ADV × avg_price) ÷ target_bars_per_day`

**Benefits:**

- Highly liquid securities get larger thresholds
- Less liquid securities get smaller thresholds
- Consistent bar generation across different securities
- Simple and interpretable

**Example:**

```python
from ds.thresholds import ADVThresholdCalculator

calculator = ADVThresholdCalculator(lookback_days=20)
volume_threshold = calculator.calculate_threshold(
    symbol="AAPL",
    historical_bars=bars,
    target_bars_per_day=50,
    for_volume=True
)
dollar_threshold = calculator.calculate_threshold(
    symbol="AAPL",
    historical_bars=bars,
    target_bars_per_day=50,
    for_volume=False
)
```

### 2. Dynamic Smoothing Threshold Calculator

The `DynamicSmoothingThresholdCalculator` uses exponential smoothing to reduce day-to-day threshold jumps.

**Algorithm:**

```
smoothed_volume = α × today_volume + (1-α) × smoothed_volume_prev
smoothed_dollar_volume = α × today_dollar_volume + (1-α) × smoothed_dollar_volume_prev

volume_threshold = scaling_factor × smoothed_volume ÷ target_bars_per_day
dollar_threshold = scaling_factor × smoothed_dollar_volume ÷ target_bars_per_day
```

**Parameters:**

- `α` (alpha): Smoothing parameter (0 < α ≤ 1). Higher values give more weight to recent observations
- `scaling_factor`: Factor to convert smoothed daily volume/dollar volume to threshold values

**Benefits:**

- Reduces threshold volatility
- Adapts gradually to changing market conditions
- Supports real-time updates
- Configurable smoothing parameters

**Example:**

```python
from ds.thresholds import DynamicSmoothingThresholdCalculator

calculator = DynamicSmoothingThresholdCalculator(
    alpha=0.1,  # 10% weight to new data
    scaling_factor=0.1  # Scale down from daily to threshold
)

dollar_threshold = calculator.calculate_threshold(
    symbol="TSLA",
    historical_bars=bars,
    target_bars_per_day=50,
    for_volume=False
)

# Real-time updates
calculator.update_smoothed_values("TSLA", new_volume, new_dollar_volume)
```

## Configuration

The threshold calculators can be configured through the `Config` class:

```python
from ds.config import Config
from ds.thresholds import create_threshold_calculator, ThresholdMethod

config = Config(
    threshold_method=ThresholdMethod.ADV,  # or ThresholdMethod.DYNAMIC or ThresholdMethod.STATIC
    target_bars_per_day=50,
    adv_lookback_days=20,
    dynamic_alpha=0.1,
    dynamic_scaling_factor=0.1,
    # ... other config options
)

# Create calculator based on config
if config.threshold_method == ThresholdMethod.ADV:
    calculator = create_threshold_calculator(
        config.threshold_method,
        lookback_days=config.adv_lookback_days
    )
elif config.threshold_method == ThresholdMethod.DYNAMIC:
    calculator = create_threshold_calculator(
        config.threshold_method,
        alpha=config.dynamic_alpha,
        scaling_factor=config.dynamic_scaling_factor
    )
```

## Integration with Training Pipeline

The `Train` class automatically uses threshold calculators when configured:

```python
from ds.train import Train
from ds.config import Config
from ds.thresholds import ThresholdMethod

config = Config(
    threshold_method=ThresholdMethod.ADV,
    target_bars_per_day=50,
    # ... other config
)

trainer = Train(config)
results = trainer.run(symbols=["AAPL", "TSLA"], security=Security.STOCKS)
```

## Calculator Creation

Use the enum-based creation function:

```python
from ds.thresholds import create_threshold_calculator, ThresholdMethod

# ADV calculator
adv_calc = create_threshold_calculator(ThresholdMethod.ADV, lookback_days=15)

# Dynamic calculator
dynamic_calc = create_threshold_calculator(
    ThresholdMethod.DYNAMIC,
    alpha=0.2,
    scaling_factor=0.05
)
```

## Best Practices

### Choosing Parameters

**ADV Calculator:**

- `lookback_days`: 20 days is a good default. Increase for more stability, decrease for faster adaptation
- `target_bars_per_day`: 50-100 bars per day is typical for intraday analysis

**Dynamic Calculator:**

- `alpha`: 0.1 (10% weight to new data) provides good balance between stability and adaptation
- `scaling_factor`: Start with 0.1 and adjust based on resulting bar counts

### When to Use Each Calculator

**Use ADV Calculator when:**

- You want simple, interpretable thresholds
- Historical data is relatively stable
- You don't need real-time updates

**Use Dynamic Calculator when:**

- Market conditions change frequently
- You need real-time threshold updates
- You want to reduce threshold volatility

### Testing and Validation

Always validate your threshold settings:

```python
# Test with historical data
calculator = ADVThresholdCalculator()
volume_threshold = calculator.calculate_threshold(
    symbol="AAPL",
    historical_bars=historical_bars,
    target_bars_per_day=50,
    for_volume=True
)

# Generate bars and check count
from ds.bars import VolumeBar
volume_bars = VolumeBar.from_alpaca(historical_bars, volume_threshold)
actual_bars_per_day = len(volume_bars) / num_trading_days

print(f"Target: 50 bars/day, Actual: {actual_bars_per_day:.1f} bars/day")
```

## Error Handling

The calculators include robust error handling:

- Validates input parameters
- Provides fallback values when no historical data is available
- Enforces minimum threshold values to prevent excessive bar generation
- Handles edge cases like empty data or single-day histories

## Performance Considerations

- ADV Calculator: O(n) time complexity where n is the number of historical bars
- Dynamic Calculator: O(n) for initial calculation, O(1) for real-time updates
- Both calculators cache results to avoid recalculation

## Example Usage

See `examples/threshold_example.py` for a complete working example that demonstrates:

- Basic usage of both calculators
- Configuration through the enum-based creation function
- Real-time updates with the dynamic calculator
- Integration with the Alpaca data retrieval system

## Testing

Run the test suite to ensure everything works correctly:

```bash
python -m pytest tests/test_thresholds.py -v
```

The test suite covers:

- Basic functionality of both calculators
- Edge cases and error conditions
- Parameter validation
- Factory function behavior
- Real-time updates for dynamic calculator
