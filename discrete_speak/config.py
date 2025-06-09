from dataclasses import dataclass
from datetime import datetime
from typing import final
from alpaca.data.timeframe import TimeFrame
from typing import Any


@final
@dataclass(frozen=True, kw_only=True)
class Config:
    # Trading

    # Training
    start: datetime
    end: datetime = datetime.now()
    step: Any = TimeFrame.Day
