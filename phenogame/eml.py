"""
phenogame.eml — EML operator decomposition and tree structures.

Provides export of calibrated response functions in EML (Exp-Minus-Log) form.
All math verified against Odrzywolek (2026) arXiv:2603.21852v2.

Key identity: eml(x, y) = exp(x) - ln(y)
Grammar: S -> 1 | eml(S, S)

With constant 1, this single operator generates the standard scientific-calculator
basis of elementary functions (Odrzywolek 2026, Theorem 1). Complex-domain
computations are required for trigonometric functions; the phenology functions
in this package are real-valued and do not require complex arithmetic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .response import RESPONSE_FUNCTIONS, K_DIRECT


@dataclass
class EMLExport:
    """EML decomposition of a calibrated response function."""
    func_id: str
    func_name: str
    equation: str
    params: Dict[str, float]
    eml_ops: List[str]
    eml_K: int
    eml_note: str
    # The core EML identities used
    identities: List[str]
    grammar: str = "S -> 1 | eml(S, S)"
    operator: str = "eml(x, y) = exp(x) - ln(y)"

    def summary(self) -> str:
        lines = [
            "EML Decomposition",
            f"  Operator: {self.operator}",
            f"  Grammar:  {self.grammar}",
            "",
            f"  Function: {self.func_name}",
            f"  Equation: {self.equation}",
            f"  Calibrated parameters: {self.params}",
            "",
            f"  EML operations: {' -> '.join(self.eml_ops)}",
            f"  Compositional K <= {self.eml_K} (RPN instruction count)",
            f"  Note: {self.eml_note}",
            "",
            "  Core EML identities used:",
        ]
        for identity in self.identities:
            lines.append(f"    {identity}")
        return "\n".join(lines)


# Basic EML identities used for metadata and decomposition summaries.
CORE_IDENTITIES = {
    "exp": "exp(x) = eml(x, 1)                    [K=3]",
    "e":   "e = eml(1, 1)                           [K=3]",
    "ln":  "ln(x) = eml(1, eml(eml(1,x), 1))       [K=7, Eq.5]",
    "zero":"0 = eml(1, eml(eml(1,1), 1))            [K=7]",
    "id":  "x = eml(eml(1, eml(eml(1,x),1)), 1)    [K=9]",
}

# Which identities each operation type requires
OP_TO_IDENTITIES = {
    "sub": ["exp", "ln"],          # x-y uses exp and ln
    "add": ["exp", "ln"],          # x+y = ln(exp(x)*exp(y))
    "mul": ["exp", "ln"],          # x*y = exp(ln(x)+ln(y))
    "div": ["exp", "ln"],          # x/y = exp(ln(x)-ln(y))
    "pow": ["exp", "ln"],          # x^y = exp(y*ln(x))
    "exp": ["exp"],                # direct: eml(x,1)
    "ln":  ["ln"],                 # Eq.5
    "neg": ["exp", "ln", "zero"],  # -x = 0 - x
    "inv": ["exp", "ln"],          # 1/x = exp(-ln(x))
    "sq":  ["exp", "ln"],          # x^2 = exp(2*ln(x))
    "sqrt":["exp", "ln"],          # sqrt(x) = exp(ln(x)/2)
    "logxy":["exp", "ln"],         # log_x(y) = ln(y)/ln(x)
    "half": ["exp", "ln"],
    "double":["exp", "ln"],
}


def export_eml(func_id: str, params: Dict[str, float]) -> EMLExport:
    """Export a calibrated response function as an EML decomposition.

    Parameters
    ----------
    func_id : str
        ID of the response function.
    params : dict
        Calibrated parameter values.

    Returns
    -------
    EMLExport with full EML metadata.
    """
    rf = RESPONSE_FUNCTIONS[func_id]

    # Collect which core identities are needed
    needed = set()
    for op in rf.eml_ops:
        for identity_key in OP_TO_IDENTITIES.get(op, ["exp", "ln"]):
            needed.add(identity_key)

    identities = [CORE_IDENTITIES[k] for k in sorted(needed) if k in CORE_IDENTITIES]

    return EMLExport(
        func_id=rf.id,
        func_name=rf.name,
        equation=rf.equation,
        params=params,
        eml_ops=rf.eml_ops,
        eml_K=rf.eml_K,
        eml_note=rf.eml_note,
        identities=identities,
    )


def reduction_chain() -> str:
    """Return the Odrzywolek Table 2 reduction chain as a string."""
    return (
        "36 buttons (Table 1)\n"
        " -> 7 (Wolfram: pi, e, i, ln, +, *, ^)\n"
        " -> 6 (Calc 3: exp, ln, -x, 1/x, +)\n"
        " -> 4 (Calc 2: exp, ln, -)\n"
        " -> 3 (Calc 0: exp, log_x(y))\n"
        " -> 2 (EML: eml, 1)\n"
        "\n"
        "No further reduction possible: at least one binary operator\n"
        "and one terminal symbol are required."
    )
