"""Task records shared by scene simulation and optimization.

These are also re-exported from common_utilis for existing scene pickles.
"""

from typing import NamedTuple, Optional

import numpy as np


class TaskInfo(NamedTuple):
    """One robot action, targeting an object when item_id is supplied."""

    task_type: str
    pose: np.ndarray
    item_id: Optional[int]


class TaskConfig(NamedTuple):
    """Task chains and an optional initial robot pose."""

    tasks: list
    pose_init: Optional[np.ndarray]
