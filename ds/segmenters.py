import warnings

import numpy as np
import ruptures as rpt
from colorama import Fore


class Ruptures:
    @classmethod
    def pelt(
        cls,
        series: list[float],
        *,
        model: str = "l2",
        pen: float = 10.0,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        algo = rpt.Pelt(
            model=model,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(pen=pen)
        return bkps[:-1]

    @classmethod
    def dynp(
        cls,
        series: list[float],
        *,
        model: str = "l2",
        n_bkps: int = 5,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        algo = rpt.Dynp(
            model=model,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(n_bkps=n_bkps)
        return bkps[:-1]

    @classmethod
    def binseg(
        cls,
        series: list[float],
        *,
        model: str = "l2",
        pen: float = 5,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        algo = rpt.Binseg(
            model=model,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(pen=pen)
        return bkps[:-1]

    @classmethod
    def bottom_up(
        cls,
        series: list[float],
        *,
        model: str = "l2",
        pen: float = 10.0,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        algo = rpt.BottomUp(
            model=model,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(pen=pen)
        return bkps[:-1]

    @classmethod
    def kernel_cpd(
        cls,
        series: list[float],
        *,
        kernel: str = "rbf",
        pen: float = 3.0,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        """
        Either `n_bkps` or `pen` must be provided (if both, n_bkps takes precedence).
        """
        algo = rpt.KernelCPD(
            kernel=kernel,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(
            pen=pen,
        )

        if not bkps:
            warnings.warn(
                f"{Fore.RED}KernelCPD failed to detect any change-points "
                f"for series of length {len(series)}."
                f"{Fore.RESET}"
            )
            return []

        return bkps[:-1]

    @classmethod
    def window(
        cls,
        series: list[float],
        *,
        width: int,
        model: str = "l2",
        pen: float = 10.0,
        min_size: int = 10,
        jump: int = 1,
    ) -> list[int]:
        algo = rpt.Window(
            width=width,
            model=model,
            min_size=min_size,
            jump=jump,
        ).fit(np.asarray(series))
        bkps = algo.predict(pen=pen)
        return bkps[:-1]
