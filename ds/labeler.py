"""
CurveLabeler for assigning analytic function labels to segmented time-series
chunks.

Each segment is fitted against a fixed vocabulary of candidate functions
(e.g. linear, quadratic, cubic, exponential, logarithmic). The function
with the lowest mean squared error is chosen as the label for that segment.
"""

import warnings
from typing import Any

import numpy as np
from colorama import Fore
from numpy.typing import NDArray
from scipy.optimize import OptimizeWarning, curve_fit


class Labeler:
  """
  Assigns analytic-curve labels to segments of a univariate series.

  Attributes:
      functions: Mapping of label names to their corresponding function signatures.
      p0: Initial parameter guesses for each function in `functions`.
  """

  def __init__(self) -> None:
    # Define the candidate functions
    self.functions: dict[str, Any] = {
      "linear": lambda x, a, b: a * x + b,
      "quadratic": lambda x, a, b, c: a * x**2 + b * x + c,
      "cubic": lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
      "exponential": lambda x, a, b: a * np.exp(b * x),
      # "logarithmic": lambda x, a, b: a * np.log(b * x),
    }

    # Initial guesses for curve_fit
    self.p0: dict[str, list[float]] = {
      "linear": [1.0, 0.0],
      "quadratic": [1.0, 0.0, 0.0],
      "cubic": [1.0, 0.0, 0.0, 0.0],
      "exponential": [1.0, 0.0],
      # "logarithmic": [1.0, 1.0],
    }

  def _label_segment(self, y: NDArray) -> tuple[str, Any, float]:
    """
    Fit each candidate curve to the data in `y` and select the best.

    Args:
        y: The data values for one segment.

    Returns:
        Tuple of (best_label, best_params, best_error) where:
        - best_label: The label of the best-fitting curve
        - best_params: The optimized parameters for the chosen curve
        - best_error: The mean squared error of the chosen fit
    """
    x = np.arange(1, len(y) + 1, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    best_label = None
    best_params = None
    best_error = float("inf")

    for name, func in self.functions.items():
      try:
        popt, _ = curve_fit(
          func,
          x,
          y_arr,
          p0=self.p0[name],
          # maxfev=10_000,
        )
        residuals = y_arr - func(x, *popt)
        mse = np.mean(residuals**2)

        if mse < best_error:
          best_error = mse
          best_label = name
          best_params = popt

      except OptimizeWarning as ow:
        warnings.warn(
          f"{Fore.RED}Curve fitting failed for '{name}' on segment of length "
          f"{len(y)}. SciPy yielded: {Fore.YELLOW}{ow}"
          f"{Fore.RESET}"
        )
      except ValueError as ve:
        warnings.warn(
          f"{Fore.RED}Your vectors may be invalid."
          "Consider filtering for NaNs or invalid dimensions."
          f"Failed for '{name}' on segment of length "
          f"{len(y)}. SciPy yielded: {Fore.YELLOW}{ve}"
          f"{Fore.RESET}"
        )
      except RuntimeError as re:
        warnings.warn(
          f"{Fore.RED}Least-squares fitting failed for '{name}' on segment of length "
          f"{len(y)}. SciPy yielded: {Fore.YELLOW}{re}"
          f"{Fore.RESET}"
        )
      except Exception as e:
        warnings.warn(
          f"{Fore.RED}Unexpected error fitting '{name}' on segment of length "
          f"{len(y)}. SciPy yielded: {Fore.YELLOW}{e}"
          f"{Fore.RESET}"
        )

    # Ensure a valid fallback if no curve fit succeeded
    if best_label is None:
      mean_val = float(np.mean(y_arr))
      best_label = "constant"
      best_params = [mean_val]
      best_error = float(np.mean((y_arr - mean_val) ** 2))

    return best_label, best_params, best_error

  def label(
    self, series: NDArray, breakpoints: list[int]
  ) -> list[dict[str, Any]]:
    """
    Apply `label_segment` across all segments defined by `breakpoints`.

    Args:
        series: The full univariate time series.
        breakpoints: Sorted list of indices where each new segment starts.
                    Indices are in the range [1..len(series)].

    Returns:
        A list of metadata dicts for each segment, where each dict contains:
        - "start": inclusive start index
        - "end": exclusive end index
        - "label": chosen curve name
        - "params": fitted parameters for the curve
        - "error": mean squared error
    """
    labels: list[dict[str, Any]] = []
    start = 0

    for breakpoint in breakpoints + [len(series)]:
      segment = series[start:breakpoint]
      label, params, err = self._label_segment(segment)
      labels.append(
        {
          "start": start,
          "end": breakpoint,
          "label": label,
          "params": params,
          "error": err,
        }
      )
      start = breakpoint

    return labels
