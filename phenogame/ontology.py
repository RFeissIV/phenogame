"""Ontology/KG-compatible export utilities for PhenoGame decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DecisionTriple:
    subject: str
    predicate: str
    object: str


def decision_to_triples(result: Any, *, crop: str = "unknown_crop", phenophase: str = "unknown_phenophase") -> List[DecisionTriple]:
    """Export a result object as simple KG triples.

    Designed to be ontology-mappable without forcing one ontology dependency.
    """
    strategy = getattr(result, "recommended_strategy", None) or getattr(result, "baseline_action", "unknown")
    cce_gap = getattr(result, "cce_gap", None)
    epsilon = getattr(result, "epsilon", None)
    node = f"Decision:{crop}:{phenophase}:{strategy}"
    triples = [
        DecisionTriple(node, "hasCrop", crop),
        DecisionTriple(node, "hasPhenophase", phenophase),
        DecisionTriple(node, "recommendsAction", str(strategy)),
    ]
    if cce_gap is not None:
        triples.append(DecisionTriple(node, "hasCCEGap", str(cce_gap)))
    if epsilon is not None:
        triples.append(DecisionTriple(node, "hasEpsilonBound", str(epsilon)))
    if hasattr(result, "certified"):
        triples.append(DecisionTriple(node, "isCertified", str(getattr(result, "certified"))))
    return triples


def decision_to_jsonld(result: Any, *, crop: str = "unknown_crop", phenophase: str = "unknown_phenophase") -> Dict[str, Any]:
    """Export a PhenoGame decision object as lightweight JSON-LD."""
    triples = decision_to_triples(result, crop=crop, phenophase=phenophase)
    strategy = getattr(result, "recommended_strategy", None) or getattr(result, "baseline_action", "unknown")
    obj: Dict[str, Any] = {
        "@context": {
            "pg": "https://example.org/phenogame#",
            "hasCrop": "pg:hasCrop",
            "hasPhenophase": "pg:hasPhenophase",
            "recommendsAction": "pg:recommendsAction",
            "hasCCEGap": "pg:hasCCEGap",
            "hasEpsilonBound": "pg:hasEpsilonBound",
            "isCertified": "pg:isCertified",
        },
        "@id": f"pg:Decision/{crop}/{phenophase}/{strategy}",
        "@type": "pg:DecisionCertificate",
    }
    for t in triples:
        obj[t.predicate] = t.object
    return obj
