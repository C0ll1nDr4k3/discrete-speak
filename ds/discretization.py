import copy
from typing import Any, Dict, List, Optional

import numpy as np


class Discretizer:
    CURVE_TYPES = ["linear", "quadratic", "cubic", "exponential", "constant"]
    CURVE_TYPE_MAP = {name: i for i, name in enumerate(CURVE_TYPES)}
    MAX_PARAMS = 4 # Cubic has 4 parameters

    def __init__(self, min_power: int, max_power: int, points_per_magnitude: int):
        self.min_power = min_power
        self.max_power = max_power
        self.points_per_magnitude = points_per_magnitude
        
        discretization_log_num_steps = (
            max_power - min_power
        ) * points_per_magnitude

        pos_range = np.logspace(min_power, max_power, discretization_log_num_steps)
        neg_range = -pos_range[::-1]
        self.discrete_range = np.concatenate((neg_range, [0], pos_range))
        
    def discretize(self, labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map curve parameters to a discrete, logarithmically spaced space.
        """
        discretized_labels = copy.deepcopy(labels)

        for label_info in discretized_labels:
            original_params = label_info.get("params")
            if original_params is None:
                continue

            discretized_params = [
                self.discrete_range[np.argmin(np.abs(self.discrete_range - param))]
                for param in original_params
            ]
            label_info["params"] = discretized_params

        return discretized_labels

    def get_param_index(self, param: float) -> int:
        """Returns the index of the closest discrete parameter value."""
        return int(np.argmin(np.abs(self.discrete_range - param)))

    @property
    def vocab_size(self) -> int:
        return len(self.discrete_range)

    def get_param_value(self, index: int) -> float:
        """Returns the continuous value for a parameter index."""
        if 0 <= index < len(self.discrete_range):
            return self.discrete_range[index]
        raise ValueError(f"Index {index} out of bounds for param range.")

    def get_curve_type(self, index: int) -> str:
        """Returns the curve type string for an index."""
        if 0 <= index < len(self.CURVE_TYPES):
            return self.CURVE_TYPES[index]
        raise ValueError(f"Index {index} out of bounds for curve types.")
