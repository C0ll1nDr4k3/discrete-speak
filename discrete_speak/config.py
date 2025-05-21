from dataclasses import dataclass
from datetime import datetime
from alpaca.data.timeframe import TimeFrame


@dataclass(frozen=True, kw_only=True)
class Config:
    # Trading

    # Training
    start: datetime
    end: datetime = datetime.now()
    step: TimeFrame
