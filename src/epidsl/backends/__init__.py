"""Simulation backends. Each turns an EpiModel into trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Trajectories on a common time grid, keyed by compartment name."""

    t: NDArray[np.float64]
    y: dict[str, NDArray[np.float64]]
    metadata: dict[str, object] = field(default_factory=dict)
