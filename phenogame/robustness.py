"""
phenogame.robustness — Uncertainty quantification for decision certificates.

Provides two complementary analyses:

  1. Bootstrap policy intervals: resample the data, rerun the game,
     measure how stable the recommendation is under sampling uncertainty.

  2. Sensitivity analysis: perturb assumptions (DOY offsets, payoff
     weights, scenario definitions) and measure recommendation stability.

Together these let the package say:

    "PhenoGame does not return a single brittle recommendation; it
     evaluates recommendation stability under sampling uncertainty
     and parameter perturbation."

All outputs are structured dataclasses with summary methods.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pipeline import (
    run_phenology_game, PhenologyGameResult, load_npn_csv,
)
from .steering import crop_steering_certificate, SteeringCertificate


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP POLICY INTERVALS
# ═══════════════════════════════════════════════════════════════

@dataclass
class BootstrapResult:
    """Bootstrap uncertainty quantification for a phenology game.

    Attributes
    ----------
    n_bootstraps : int
        Number of bootstrap resamples completed.
    action_frequency : Dict[str, float]
        Fraction of bootstraps each strategy was recommended.
    modal_action : str
        Most frequently recommended strategy.
    modal_frequency : float
        Fraction of bootstraps recommending the modal action.
    cce_gap_mean : float
        Mean CCE gap across bootstraps.
    cce_gap_ci : Tuple[float, float]
        2.5% / 97.5% percentile interval for CCE gap.
    certification_rate : float
        Fraction of bootstraps where CCE was certified.
    consensus_rate : float
        Fraction of bootstraps where all DPUU criteria agreed.
    payoff_mean : float
        Mean estimated payoff across bootstraps.
    payoff_ci : Tuple[float, float]
        2.5% / 97.5% percentile interval for estimated payoff.
    """
    n_bootstraps: int
    action_frequency: Dict[str, float]
    modal_action: str
    modal_frequency: float
    cce_gap_mean: float
    cce_gap_ci: Tuple[float, float]
    certification_rate: float
    consensus_rate: float
    payoff_mean: float
    payoff_ci: Tuple[float, float]

    def summary(self) -> str:
        lines = [
            f"Bootstrap Policy Interval (n={self.n_bootstraps})",
            f"  Modal action: {self.modal_action} "
            f"({self.modal_frequency:.0%} of bootstraps)",
            "",
            "  Action selection frequency:",
        ]
        for action, freq in sorted(
            self.action_frequency.items(), key=lambda x: -x[1]
        ):
            bar = "#" * int(freq * 40)
            lines.append(f"    {action:12s} {freq:5.1%}  {bar}")
        lines.extend([
            "",
            f"  CCE gap:     {self.cce_gap_mean:.4f} "
            f"[{self.cce_gap_ci[0]:.4f}, {self.cce_gap_ci[1]:.4f}]",
            f"  Certified:   {self.certification_rate:.0%} of bootstraps",
            f"  DPUU consensus: {self.consensus_rate:.0%} of bootstraps",
            f"  Est. payoff: {self.payoff_mean:.4f} "
            f"[{self.payoff_ci[0]:.4f}, {self.payoff_ci[1]:.4f}]",
        ])
        return "\n".join(lines)


def bootstrap_policy_interval(
    npn_data: pd.DataFrame,
    phenophase: str = "Ripe fruits",
    n_bootstraps: int = 200,
    seed: int = 42,
    **game_kwargs,
) -> BootstrapResult:
    """Compute bootstrap intervals for the phenology game recommendation.

    Resamples NPN observations with replacement, reruns the full game
    pipeline, and measures recommendation stability.

    Parameters
    ----------
    npn_data : DataFrame
        USA-NPN data (already filtered by load_npn_csv).
    phenophase : str
        Phenophase to analyze.
    n_bootstraps : int
        Number of bootstrap resamples (default 200).
    seed : int
        Random seed for reproducibility.
    **game_kwargs
        Passed to run_phenology_game (e.g., strategy_offsets).

    Returns
    -------
    BootstrapResult with intervals and action frequencies.
    """
    rng = np.random.default_rng(seed)
    # Robustness bootstraps should be fast enough for CI unless callers override.
    game_kwargs.setdefault("iterations", 50)

    # Get the phenophase subset
    ph_data = npn_data[
        npn_data["Phenophase_Description"] == phenophase
    ].copy()
    n_obs = len(ph_data)

    if n_obs < 2:
        raise ValueError(
            f"Need at least 2 observations for '{phenophase}', got {n_obs}."
        )

    recommendations = []
    cce_gaps = []
    certifieds = []
    consensuses = []
    payoffs = []

    for b in range(n_bootstraps):
        # Resample rows with replacement
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_ph = ph_data.iloc[idx].copy()

        # run_phenology_game only uses the requested phenophase, so pass only
        # the bootstrap subset. This avoids copying unrelated large species tables.
        try:
            result = run_phenology_game(
                boot_ph, phenophase=phenophase,
                seed=seed + b, **game_kwargs,
            )
            recommendations.append(result.recommended_strategy)
            cce_gaps.append(result.cce_gap)
            certifieds.append(result.certified)
            consensuses.append(result.consensus)
            payoffs.append(result.certificate.estimated_payoff)
        except (ValueError, RuntimeError):
            # Skip bootstraps that produce degenerate games
            # (e.g., zero variance from resampling)
            continue

    if len(recommendations) < 10:
        raise RuntimeError(
            f"Only {len(recommendations)} valid bootstraps out of "
            f"{n_bootstraps}. Data may be too sparse."
        )

    # Compute statistics
    action_counts = pd.Series(recommendations).value_counts(normalize=True)
    action_freq = action_counts.to_dict()
    modal = action_counts.index[0]
    modal_freq = float(action_counts.iloc[0])

    gaps = np.array(cce_gaps)
    pays = np.array(payoffs)

    return BootstrapResult(
        n_bootstraps=len(recommendations),
        action_frequency=action_freq,
        modal_action=modal,
        modal_frequency=modal_freq,
        cce_gap_mean=float(gaps.mean()),
        cce_gap_ci=(float(np.percentile(gaps, 2.5)),
                    float(np.percentile(gaps, 97.5))),
        certification_rate=float(np.mean(certifieds)),
        consensus_rate=float(np.mean(consensuses)),
        payoff_mean=float(pays.mean()),
        payoff_ci=(float(np.percentile(pays, 2.5)),
                   float(np.percentile(pays, 97.5))),
    )


# ═══════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SensitivityResult:
    """Sensitivity of recommendation to assumption perturbations.

    Attributes
    ----------
    parameter : str
        Name of the perturbed parameter.
    values : List[float]
        Perturbation values tested.
    recommendations : List[str]
        Recommended strategy at each perturbation.
    cce_gaps : List[float]
        CCE gap at each perturbation.
    certified : List[bool]
        Certification status at each perturbation.
    stable : bool
        True if recommendation is the same at all perturbations.
    tipping_point : Optional[float]
        Value where recommendation changes (None if stable).
    """
    parameter: str
    values: List[float]
    recommendations: List[str]
    cce_gaps: List[float]
    certified: List[bool]
    stable: bool
    tipping_point: Optional[float]

    def summary(self) -> str:
        lines = [
            f"Sensitivity: {self.parameter}",
            f"  Stable: {'YES' if self.stable else 'NO'}",
        ]
        if self.tipping_point is not None:
            lines.append(f"  Tipping point: {self.tipping_point}")
        lines.append("")
        for v, r, g, c in zip(
            self.values, self.recommendations, self.cce_gaps, self.certified
        ):
            lines.append(
                f"  {self.parameter}={v:+6.1f}  "
                f"rec={r:12s}  gap={g:.4f}  cert={'Y' if c else 'N'}"
            )
        return "\n".join(lines)


def sensitivity_analysis(
    npn_data: pd.DataFrame,
    phenophase: str = "Ripe fruits",
    doy_offsets: Optional[List[float]] = None,
    strategy_scale_factors: Optional[List[float]] = None,
    seed: int = 42,
    **game_kwargs,
) -> List[SensitivityResult]:
    """Run sensitivity analysis on the phenology game.

    Perturbs DOY values and strategy definitions, reruns the game,
    and reports whether the recommendation changes.

    Parameters
    ----------
    npn_data : DataFrame
        USA-NPN data (already filtered).
    phenophase : str
        Phenophase to analyze.
    doy_offsets : list of float, optional
        DOY perturbations to test (default: [-7, -3, 0, 3, 7]).
    strategy_scale_factors : list of float, optional
        Multipliers for strategy offset from mean
        (default: [0.25, 0.5, 0.75, 1.0, 1.5]).
    seed : int
        Random seed.

    Returns
    -------
    List of SensitivityResult, one per parameter.
    """
    game_kwargs.setdefault("iterations", 50)

    ph_data = npn_data[npn_data["Phenophase_Description"] == phenophase].copy()
    if len(ph_data) < 2:
        raise ValueError(
            f"Need at least 2 observations for '{phenophase}', got {len(ph_data)}."
        )

    results = []

    # --- DOY perturbation ---
    offsets = doy_offsets or [-7, -3, 0, 3, 7]
    recs, gaps, certs = [], [], []

    for delta in offsets:
        perturbed = ph_data.copy()
        perturbed["Mean_First_Yes_DOY"] = (
            perturbed["Mean_First_Yes_DOY"] + delta
        )
        try:
            r = run_phenology_game(
                perturbed, phenophase=phenophase, seed=seed,
                **game_kwargs,
            )
            recs.append(r.recommended_strategy)
            gaps.append(r.cce_gap)
            certs.append(r.certified)
        except (ValueError, RuntimeError):
            recs.append("ERROR")
            gaps.append(float("nan"))
            certs.append(False)

    unique_recs = set(r for r in recs if r != "ERROR")
    stable = len(unique_recs) <= 1
    tipping = None
    if not stable:
        baseline_rec = recs[offsets.index(0)] if 0 in offsets else recs[0]
        for v, r in zip(offsets, recs):
            if r != baseline_rec and r != "ERROR":
                tipping = v
                break

    results.append(SensitivityResult(
        parameter="doy_offset",
        values=offsets,
        recommendations=recs,
        cce_gaps=gaps,
        certified=certs,
        stable=stable,
        tipping_point=tipping,
    ))

    # --- Strategy scale perturbation ---
    scales = strategy_scale_factors or [0.25, 0.5, 0.75, 1.0, 1.5]
    recs2, gaps2, certs2 = [], [], []

    for scale in scales:
        try:
            r = run_phenology_game(
                ph_data, phenophase=phenophase, seed=seed,
                strategy_offsets={
                    "early": -0.5 * scale,
                    "standard": 0.0,
                    "late": 0.5 * scale,
                },
                **game_kwargs,
            )
            recs2.append(r.recommended_strategy)
            gaps2.append(r.cce_gap)
            certs2.append(r.certified)
        except (ValueError, RuntimeError):
            recs2.append("ERROR")
            gaps2.append(float("nan"))
            certs2.append(False)

    unique_recs2 = set(r for r in recs2 if r != "ERROR")
    stable2 = len(unique_recs2) <= 1
    tipping2 = None
    if not stable2:
        for v, r in zip(scales, recs2):
            if r != "standard" and r != "ERROR":
                tipping2 = v
                break

    results.append(SensitivityResult(
        parameter="strategy_scale",
        values=scales,
        recommendations=recs2,
        cce_gaps=gaps2,
        certified=certs2,
        stable=stable2,
        tipping_point=tipping2,
    ))

    return results


# ═══════════════════════════════════════════════════════════════
# ROBUSTNESS REPORT
# ═══════════════════════════════════════════════════════════════

@dataclass
class RobustnessReport:
    """Combined bootstrap + sensitivity robustness assessment.

    Attributes
    ----------
    bootstrap : BootstrapResult
        Bootstrap policy intervals.
    sensitivity : List[SensitivityResult]
        Sensitivity analysis results.
    robustness_score : float
        Combined robustness score in [0, 1].
        0 = brittle, 1 = fully robust.
    """
    bootstrap: BootstrapResult
    sensitivity: List[SensitivityResult]
    robustness_score: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ROBUSTNESS REPORT",
            "=" * 60,
            "",
            self.bootstrap.summary(),
            "",
        ]
        for s in self.sensitivity:
            lines.append(s.summary())
            lines.append("")
        lines.extend([
            "-" * 60,
            f"  Robustness score: {self.robustness_score:.2f} / 1.00",
            "",
            "  Score components:",
            f"    Bootstrap stability:  {self.bootstrap.modal_frequency:.2f}",
            f"    Certification rate:   {self.bootstrap.certification_rate:.2f}",
            f"    Sensitivity stability: "
            f"{sum(1 for s in self.sensitivity if s.stable) / max(len(self.sensitivity), 1):.2f}",
            "-" * 60,
        ])
        return "\n".join(lines)


def robustness_report(
    npn_data: pd.DataFrame,
    phenophase: str = "Ripe fruits",
    n_bootstraps: int = 200,
    seed: int = 42,
    **game_kwargs,
) -> RobustnessReport:
    """Produce a combined robustness assessment.

    Runs bootstrap policy intervals and sensitivity analysis,
    then computes a combined robustness score.

    Parameters
    ----------
    npn_data : DataFrame
        USA-NPN data (already filtered by load_npn_csv).
    phenophase : str
        Phenophase to analyze.
    n_bootstraps : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    RobustnessReport with score, bootstrap, and sensitivity.

    Example
    -------
    >>> npn = load_npn_csv()
    >>> report = robustness_report(npn, seed=42)
    >>> print(report.summary())
    >>> print(f"Score: {report.robustness_score:.2f}")
    """
    boot = bootstrap_policy_interval(
        npn_data, phenophase=phenophase,
        n_bootstraps=n_bootstraps, seed=seed, **game_kwargs,
    )

    sens = sensitivity_analysis(
        npn_data, phenophase=phenophase, seed=seed, **game_kwargs,
    )

    # Robustness score: weighted average of three components
    boot_stability = boot.modal_frequency
    cert_rate = boot.certification_rate
    sens_stability = (
        sum(1 for s in sens if s.stable) / max(len(sens), 1)
    )

    score = 0.4 * boot_stability + 0.3 * cert_rate + 0.3 * sens_stability

    return RobustnessReport(
        bootstrap=boot,
        sensitivity=sens,
        robustness_score=float(score),
    )
