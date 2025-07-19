from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import ruptures as rpt


class Segmenter(ABC):
    """
    Abstract interface for change-point detection algorithms.

    A Segmenter accepts a univariate time series (e.g., price, volume, or returns)
    and produces indices where the series should be split into regimes.
    """

    @abstractmethod
    def fit(self, series: list[float]) -> None:
        """
        Train the segmenter on a univariate series.

        Args:
            series: list of float values representing the time series.
        """

    @abstractmethod
    def predict(self, series: list[float]) -> list[int]:
        """
        Return change-point indices for the given series.

        The returned list contains indices in the range [1..len(series)) where each
        index marks the start of a new segment. For example, an index of 10 indicates
        that series[10] is the first element of a new regime.

        Args:
            series: sequence of float values representing the time series.

        Returns:
            A list of integer breakpoints (change-point indices) in ascending order.
        """


class PELT(Segmenter):
    """
    Change-point detection using the PELT algorithm from the ruptures library.

    Args:
        model: Cost function to use (e.g. "l2", "rbf", "linear").
        pen: Penalty value to control number of change points.
        min_size: Minimum segment length.
        jump: Subsampling step to speed up search (1 means no subsampling).
    """

    def __init__(
        self,
        model: str = "l2",
        pen: float = 10.0,
        min_size: int = 1,
        jump: int = 1,
    ):
        self.model = model
        self.pen = pen
        self.min_size = min_size
        self.jump = jump

    def fit(self, series: Sequence[float]) -> None:
        """
        Fit the PELT algorithm on the raw series.

        Args:
            series: Univariate time series data.
        """
        # Create and fit the PELT instance
        # Stateless: no internal state is stored during fit.
        pass

    def predict(self, series: Sequence[float]) -> list[int]:
        """
        Predict change-point indices for the given series.

        If not already fitted on this series, it will fit first.

        Args:
            series: Univariate time series data.

        Returns:
            Sorted list of breakpoints where new segments begin (excluding the final point).
        """
        # Stateless run: instantiate and fit PELT for this series
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump).fit(
            np.asarray(series)
        )
        # `bkps` includes the final endpoint len(series)
        bkps = algo.predict(pen=self.pen)
        # Slice off the last breakpoint (== len(series))
        return bkps[:-1]
