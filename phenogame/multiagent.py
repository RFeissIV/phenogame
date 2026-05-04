"""Minimal multi-agent game containers for future grower/weather/market models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MultiAgentDecisionGame:
    """N-player payoff tensor container.

    This is intentionally conservative: it stores and validates multi-agent
    payoff tensors but does not claim to solve general N-player Nash games.
    """

    agents: List[str]
    action_labels: Dict[str, List[str]]
    payoff_tensors: Dict[str, np.ndarray]

    def validate(self) -> None:
        expected_shape = tuple(len(self.action_labels[a]) for a in self.agents)
        for agent in self.agents:
            if agent not in self.payoff_tensors:
                raise ValueError(f"Missing payoff tensor for agent: {agent}")
            arr = np.asarray(self.payoff_tensors[agent], dtype=float)
            if arr.shape != expected_shape:
                raise ValueError(f"Payoff tensor for {agent} has shape {arr.shape}, expected {expected_shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Payoff tensor for {agent} contains non-finite values.")

    def summary(self) -> str:
        self.validate()
        lines = ["Multi-agent decision game", f"  agents: {', '.join(self.agents)}"]
        for agent in self.agents:
            lines.append(f"  {agent} actions: {', '.join(self.action_labels[agent])}")
        lines.append("  solver: not provided; use this as a validated data structure only")
        return "\n".join(lines)
