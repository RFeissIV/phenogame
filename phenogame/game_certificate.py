"""Game audit certificates combining assumptions, robustness, and KG export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ontology import decision_to_jsonld


@dataclass
class GameAuditCertificate:
    """Human/machine-readable audit certificate for a recommendation."""

    result: Any
    assumptions: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)
    robustness: Optional[Any] = None
    counterfactual: Optional[Any] = None
    crop: str = "unknown_crop"
    phenophase: str = "unknown_phenophase"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop": self.crop,
            "phenophase": self.phenophase,
            "recommended_strategy": getattr(self.result, "recommended_strategy", None),
            "cce_gap": getattr(self.result, "cce_gap", None),
            "epsilon": getattr(self.result, "epsilon", None),
            "certified": getattr(self.result, "certified", None),
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "robustness_score": getattr(self.robustness, "robustness_score", None),
            "kg_jsonld": decision_to_jsonld(self.result, crop=self.crop, phenophase=self.phenophase),
        }

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "  PHENOGAME GAME AUDIT CERTIFICATE",
            "=" * 64,
            f"Crop: {self.crop}",
            f"Phenophase: {self.phenophase}",
            f"Recommended strategy: {getattr(self.result, 'recommended_strategy', 'unknown')}",
            f"CCE gap: {getattr(self.result, 'cce_gap', 'n/a')}",
            f"Epsilon: {getattr(self.result, 'epsilon', 'n/a')}",
            f"Certified: {getattr(self.result, 'certified', 'n/a')}",
            "",
            "Assumptions:",
        ]
        for k, v in self.assumptions.items():
            lines.append(f"  - {k}: {v}")
        if self.robustness is not None:
            lines.extend(["", f"Robustness score: {getattr(self.robustness, 'robustness_score', 'n/a')}"])
        if self.counterfactual is not None:
            lines.extend(["", "Counterfactual:", self.counterfactual.summary()])
        lines.extend(["", "Limitations:"])
        for item in self.limitations or ["Conditional on data, payoff construction, and scenario definitions."]:
            lines.append(f"  - {item}")
        lines.append("=" * 64)
        return "\n".join(lines)


def build_game_audit_certificate(result: Any, **kwargs: Any) -> GameAuditCertificate:
    """Convenience constructor for GameAuditCertificate."""
    defaults = {
        "limitations": [
            "This is a decision-support certificate, not a validated agronomic prescription.",
            "EML-tree payoff learning is upstream representation, not a new equilibrium theorem.",
            "Results are conditional on finite actions, finite scenarios, and supplied data quality.",
        ]
    }
    defaults.update(kwargs)
    return GameAuditCertificate(result=result, **defaults)
