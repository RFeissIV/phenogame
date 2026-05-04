"""Counterfactual decision testing for phenology games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .pipeline import run_phenology_game


@dataclass
class CounterfactualResult:
    """Result of shifting management/phenology timing and rerunning the game."""

    offsets: List[float]
    recommendations: List[str]
    cce_gaps: List[float]
    certified: List[bool]
    baseline_action: str
    best_offset: float
    best_estimated_payoff: float

    def summary(self) -> str:
        lines = [
            "Counterfactual timing test",
            f"  baseline action: {self.baseline_action}",
            f"  best offset: {self.best_offset:+.1f} days",
            f"  best estimated payoff: {self.best_estimated_payoff:.4f}",
            "",
        ]
        for off, rec, gap, cert in zip(self.offsets, self.recommendations, self.cce_gaps, self.certified):
            lines.append(f"  offset={off:+6.1f} days  rec={rec:12s}  gap={gap:.4f}  cert={'Y' if cert else 'N'}")
        return "\n".join(lines)


def counterfactual_timing_test(
    npn_data: pd.DataFrame,
    *,
    phenophase: str = "Ripe fruits",
    offsets: Optional[List[float]] = None,
    seed: int = 42,
    **game_kwargs,
) -> CounterfactualResult:
    """Ask what would happen if observed phenology/timing shifted earlier/later."""
    game_kwargs.setdefault("iterations", 50)
    offsets = offsets or [-7, -5, -3, 0, 3, 5, 7]
    required = ["Phenophase_Description", "Mean_First_Yes_DOY"]
    missing = [c for c in required if c not in npn_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    recs, gaps, certs, payoffs = [], [], [], []
    baseline_action = "ERROR"
    for off in offsets:
        df = npn_data.copy()
        mask = df["Phenophase_Description"] == phenophase
        df.loc[mask, "Mean_First_Yes_DOY"] = df.loc[mask, "Mean_First_Yes_DOY"] + off
        try:
            result = run_phenology_game(df[df["Phenophase_Description"] == phenophase].copy(), phenophase=phenophase, seed=seed, **game_kwargs)
            recs.append(result.recommended_strategy)
            gaps.append(result.cce_gap)
            certs.append(result.certified)
            payoffs.append(result.certificate.estimated_payoff)
            if off == 0:
                baseline_action = result.recommended_strategy
        except (ValueError, RuntimeError):
            recs.append("ERROR")
            gaps.append(float("nan"))
            certs.append(False)
            payoffs.append(float("-inf"))
    best_idx = max(range(len(payoffs)), key=lambda i: payoffs[i])
    return CounterfactualResult(
        offsets=list(offsets), recommendations=recs, cce_gaps=gaps, certified=certs,
        baseline_action=baseline_action, best_offset=float(offsets[best_idx]),
        best_estimated_payoff=float(payoffs[best_idx]),
    )
