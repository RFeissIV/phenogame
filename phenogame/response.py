"""
phenogame.response — 13 EML-reducible crop phenology response functions.

Each function is calibratable from field data. The ``inverse_fn`` attribute
holds a numerical inverse of the response (a math property of the function
class) used for verification of EML decompositions.
EML decompositions and K values verified against Odrzywolek (2026) Table 4.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ─── Table 4 direct-search K values (Odrzywolek 2026) ───
# Used for compositional K estimation of response functions.
K_DIRECT = {
    "sub": 11, "add": 19, "mul": 17, "div": 17, "pow": 25,
    "exp": 3, "ln": 7, "neg": 15, "inv": 15, "sq": 17,
    "sqrt": 43, "logxy": 29, "half": 27, "double": 19,
}


@dataclass
class ResponseFunction:
    """A phenological response function with EML metadata."""
    id: str
    name: str
    equation: str
    description: str
    param_names: List[str]
    param_defaults: Dict[str, float]
    param_bounds: Dict[str, Tuple[float, float]]
    x_label: str
    y_label: str
    fn: Callable  # fn(x, **params) -> y
    inverse_fn: Optional[Callable] = None  # inverse(y, **params) -> x
    eml_ops: List[str] = field(default_factory=list)
    eml_K: int = 0
    eml_note: str = ""
    applicable_scales: List[str] = field(default_factory=list)
    applicable_models: List[str] = field(default_factory=list)


def _compute_K(ops: List[str]) -> int:
    """Compositional K upper bound from Table 4 direct-search values."""
    return sum(K_DIRECT.get(op, 30) for op in ops)


# ═══════════════════════════════════════════════════════════════
# THE 13 RESPONSE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _beta_thermal(x, Tb=0.0, To=26.0, Tc=34.0, alpha=1.7, beta=1.7):
    """Beta-function thermal response (cardinal temperature model)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask = (x > Tb) & (x < Tc)
    xm = x[mask]
    num = np.power(xm - Tb, alpha) * np.power(Tc - xm, beta)
    den = np.power(To - Tb, alpha) * np.power(Tc - To, beta)
    y[mask] = np.clip(num / den, 0, None)
    return y


def _beta_thermal_inv(y, Tb=0.0, To=26.0, Tc=34.0, alpha=1.7, beta=1.7):
    """Numerical inverse: given target rate y, find temperature T.

    Returns the two solutions (ascending and descending limb).
    """
    from scipy.optimize import brentq
    results = []
    f = lambda t: _beta_thermal(t, Tb, To, Tc, alpha, beta) - y
    # Ascending limb
    try:
        results.append(brentq(f, Tb + 1e-6, To))
    except ValueError:
        pass
    # Descending limb
    try:
        results.append(brentq(f, To, Tc - 1e-6))
    except ValueError:
        pass
    return np.array(results)


def _linear_thermal(x, Tb=10.0):
    """Growing degree-days: GDD = max(0, T - Tb)."""
    return np.maximum(0.0, np.asarray(x, dtype=float) - Tb)


def _linear_thermal_inv(y, Tb=10.0):
    """Inverse: T = y + Tb (for y >= 0)."""
    return np.asarray(y, dtype=float) + Tb


def _sigmoid(x, k=0.08, x0=120.0):
    """Logistic sigmoid: sigma(x) = 1 / (1 + exp(-k(x - x0)))."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _sigmoid_inv(y, k=0.08, x0=120.0):
    """Inverse: x = x0 - ln((1-y)/y) / k."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-12, 1 - 1e-12)
    return x0 - np.log((1.0 - y) / y) / k


def _exp_saturation(x, A=5.0, k=0.004):
    """Monomolecular: f(t) = A * (1 - exp(-k*t))."""
    with np.errstate(over="ignore"):
        return A * (1.0 - np.exp(-k * np.asarray(x, dtype=float)))


def _exp_saturation_inv(y, A=5.0, k=0.004):
    """Inverse: t = -ln(1 - y/A) / k."""
    y = np.asarray(y, dtype=float)
    ratio = np.clip(y / A, 0, 1 - 1e-12)
    return -np.log(1.0 - ratio) / k


def _photoperiod(x, Pc=12.0, Po=14.5, n=1.0):
    """Photoperiod response: clamp01((P - Pc)/(Po - Pc))^n."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask = (x > Pc) & (x < Po)
    y[mask] = np.power((x[mask] - Pc) / (Po - Pc), n)
    y[x >= Po] = 1.0
    return y


def _photoperiod_inv(y, Pc=12.0, Po=14.5, n=1.0):
    """Inverse: P = Pc + (Po - Pc) * y^(1/n)."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, 1)
    return Pc + (Po - Pc) * np.power(y, 1.0 / n)


def _vernalization(x, Vsat=45.0):
    """Vernalization: Veff = 1 - (1 - Vd/Vsat)^2."""
    x = np.asarray(x, dtype=float)
    r = np.clip(x / Vsat, 0, 1)
    return 1.0 - np.power(1.0 - r, 2)


def _vernalization_inv(y, Vsat=45.0):
    """Inverse: Vd = Vsat * (1 - sqrt(1 - y))."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, 1)
    return Vsat * (1.0 - np.sqrt(1.0 - y))


def _water_stress(x, wp=0.12, fc=0.33):
    """Water stress factor: Ks = clamp01((theta - wp)/(fc - wp))."""
    x = np.asarray(x, dtype=float)
    return np.clip((x - wp) / (fc - wp), 0, 1)


def _water_stress_inv(y, wp=0.12, fc=0.33):
    """Inverse: theta = wp + y*(fc - wp)."""
    return wp + np.asarray(y, dtype=float) * (fc - wp)


def _n_stress(x, Nmin=0.01, Ncrit=0.04, k=3.5):
    """Nitrogen limitation: fN = 1 - exp(-k*(N-Nmin)/(Ncrit-Nmin))."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask = (x > Nmin) & (x < Ncrit)
    ratio = (x[mask] - Nmin) / (Ncrit - Nmin)
    y[mask] = 1.0 - np.exp(-k * ratio)
    y[x >= Ncrit] = 1.0
    return y


def _n_stress_inv(y, Nmin=0.01, Ncrit=0.04, k=3.5):
    """Inverse: N = Nmin + (Ncrit-Nmin) * (-ln(1-y)/k)."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, 1 - 1e-12)
    return Nmin + (Ncrit - Nmin) * (-np.log(1.0 - y) / k)


def _gaussian(x, To=28.0, sigma=6.0):
    """Gaussian optimum: f(T) = exp(-(T-To)^2 / (2*sigma^2))."""
    x = np.asarray(x, dtype=float)
    return np.exp(-np.power(x - To, 2) / (2.0 * sigma * sigma))


def _gaussian_inv(y, To=28.0, sigma=6.0):
    """Inverse: T = To +/- sigma*sqrt(-2*ln(y)). Returns both roots."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-12, 1)
    offset = sigma * np.sqrt(-2.0 * np.log(y))
    return np.array([To - offset, To + offset])


def _gompertz(x, A=1.0, b=3.0, c=0.03):
    """Gompertz: f(t) = A * exp(-b * exp(-c*t))."""
    with np.errstate(over="ignore"):
        return A * np.exp(-b * np.exp(-c * np.asarray(x, dtype=float)))


def _gompertz_inv(y, A=1.0, b=3.0, c=0.03):
    """Inverse: t = -ln(-ln(y/A)/b) / c."""
    y = np.asarray(y, dtype=float)
    ratio = np.clip(y / A, 1e-12, 1 - 1e-12)
    return -np.log(-np.log(ratio) / b) / c


def _broken_stick(x, Tb=0.0, To=25.0, Tc=35.0):
    """Broken-stick (triangular) thermal response."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    asc = (x > Tb) & (x <= To)
    desc = (x > To) & (x < Tc)
    y[asc] = (x[asc] - Tb) / (To - Tb)
    y[desc] = (Tc - x[desc]) / (Tc - To)
    return y


def _broken_stick_inv(y, Tb=0.0, To=25.0, Tc=35.0):
    """Inverse: returns ascending and descending T. Two roots."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0, 1)
    T_asc = Tb + y * (To - Tb)
    T_desc = Tc - y * (Tc - To)
    return np.array([T_asc, T_desc])


def _beer_lambert(x, RUE=1.2, k=0.5):
    """Beer-Lambert: dW/dt = RUE * (1 - exp(-k*LAI))."""
    with np.errstate(over="ignore"):
        return RUE * (1.0 - np.exp(-k * np.asarray(x, dtype=float)))


def _beer_lambert_inv(y, RUE=1.2, k=0.5):
    """Inverse: LAI = -ln(1 - y/RUE) / k."""
    y = np.asarray(y, dtype=float)
    ratio = np.clip(y / RUE, 0, 1 - 1e-12)
    return -np.log(1.0 - ratio) / k


def _farquhar(x, Vcmax=120.0, Kc=300.0, Ko=300000.0, O2=210000.0, Rd=1.5):
    """Farquhar: A = Vcmax*Ci/(Ci + Kc*(1+O/Ko)) - Rd."""
    x = np.asarray(x, dtype=float)
    Wc = Vcmax * x / (x + Kc * (1.0 + O2 / Ko))
    return np.maximum(0.0, Wc - Rd)


def _farquhar_inv(y, Vcmax=120.0, Kc=300.0, Ko=300000.0, O2=210000.0, Rd=1.5):
    """Inverse Michaelis-Menten: Ci = Keff*(y+Rd)/(Vcmax - y - Rd)."""
    y = np.asarray(y, dtype=float)
    Keff = Kc * (1.0 + O2 / Ko)
    denom = Vcmax - y - Rd
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return Keff * (y + Rd) / denom


# ═══════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════

_BETA_OPS = ["sub", "sub", "pow", "pow", "mul", "mul", "div"]
_SIG_OPS = ["sub", "mul", "neg", "exp", "add", "inv"]
_EXPSAT_OPS = ["mul", "neg", "exp", "sub", "mul"]
_PHOTO_OPS = ["sub", "sub", "div", "pow"]
_VERN_OPS = ["div", "sub", "sq", "sub"]
_WATER_OPS = ["sub", "sub", "div"]
_NSTR_OPS = ["sub", "sub", "div", "mul", "neg", "exp", "sub"]
_GAUSS_OPS = ["sub", "sq", "mul", "div", "neg", "exp"]
_GOMP_OPS = ["mul", "neg", "exp", "mul", "neg", "exp", "mul"]
_BSTICK_OPS = ["sub", "div", "sub", "div"]
_BEER_OPS = ["mul", "neg", "exp", "sub", "mul", "mul"]
_FAQ_OPS = ["div", "add", "mul", "add", "mul", "div", "sub"]

RESPONSE_FUNCTIONS: Dict[str, ResponseFunction] = {}


def _register(rf: ResponseFunction):
    RESPONSE_FUNCTIONS[rf.id] = rf


_register(ResponseFunction(
    id="beta_thermal",
    name="Beta-function thermal response",
    equation="f(T) = [(T-Tb)^a * (Tc-T)^b] / [(To-Tb)^a * (Tc-To)^b]",
    description="Cardinal temperature model. Skewed bell curve.",
    param_names=["Tb", "To", "Tc", "alpha", "beta"],
    param_defaults=dict(Tb=0, To=26, Tc=34, alpha=1.7, beta=1.7),
    param_bounds=dict(Tb=(-10, 15), To=(15, 35), Tc=(25, 50), alpha=(0.5, 5), beta=(0.5, 5)),
    x_label="Temperature (C)", y_label="Development rate (0-1)",
    fn=_beta_thermal, inverse_fn=_beta_thermal_inv,
    eml_ops=_BETA_OPS, eml_K=_compute_K(_BETA_OPS),
    eml_note="2x subtract(11) + 2x power(25) + 2x multiply(17) + divide(17) = K<=123",
    applicable_scales=["BBCH", "Zadoks", "Feekes", "Haun", "Kuperman"],
    applicable_models=["DSSAT-CERES", "APSIM", "WOFOST", "CROPGRO", "GOSSYM"],
))

_register(ResponseFunction(
    id="linear_thermal",
    name="Linear thermal time (GDD)",
    equation="GDD = max(0, T - Tb)",
    description="Growing degree-days. Clamp-and-sum.",
    param_names=["Tb"],
    param_defaults=dict(Tb=10),
    param_bounds=dict(Tb=(-5, 20)),
    x_label="Temperature (C)", y_label="GDD contribution",
    fn=_linear_thermal, inverse_fn=_linear_thermal_inv,
    eml_ops=["sub"], eml_K=_compute_K(["sub"]),
    eml_note="Subtraction K=11. Clamp via |x|=sqrt(x^2) adds ~106.",
    applicable_scales=["BBCH", "Zadoks", "Feekes", "Haun", "COTMAN", "Fehr-Caviness", "Kuperman"],
    applicable_models=["DSSAT-CERES", "APSIM", "WOFOST", "GOSSYM", "COTMAN"],
))

_register(ResponseFunction(
    id="sigmoid",
    name="Logistic sigmoid",
    equation="sigma(x) = 1 / (1 + exp(-k(x - x0)))",
    description="S-curve for emergence, photoperiod, vernalization saturation.",
    param_names=["k", "x0"],
    param_defaults=dict(k=0.08, x0=120),
    param_bounds=dict(k=(0.001, 1.0), x0=(0, 2000)),
    x_label="Accumulated GDD", y_label="Probability / fraction",
    fn=_sigmoid, inverse_fn=_sigmoid_inv,
    eml_ops=_SIG_OPS, eml_K=_compute_K(_SIG_OPS),
    eml_note="subtract(11) + scale(17) + negate(15) + exp(3) + add(19) + reciprocal(15) = K<=80",
    applicable_scales=["BBCH", "Zadoks", "Feekes", "Fehr-Caviness", "COTMAN"],
    applicable_models=["DSSAT-CERES", "APSIM", "WOFOST", "CROPGRO"],
))

_register(ResponseFunction(
    id="exp_saturation",
    name="Exponential saturation",
    equation="f(t) = A * (1 - exp(-k*t))",
    description="Monomolecular growth. Boll maturation, LAI expansion, N uptake.",
    param_names=["A", "k"],
    param_defaults=dict(A=5, k=0.004),
    param_bounds=dict(A=(0.01, 100), k=(1e-6, 1.0)),
    x_label="Thermal time", y_label="Accumulated response",
    fn=_exp_saturation, inverse_fn=_exp_saturation_inv,
    eml_ops=_EXPSAT_OPS, eml_K=_compute_K(_EXPSAT_OPS),
    eml_note="scale(17) + negate(15) + exp(3) + subtract(11) + scale(17) = K<=63",
    applicable_scales=["COTMAN", "Fehr-Caviness", "GOSSYM"],
    applicable_models=["COTMAN", "GOSSYM", "DSSAT-CERES", "CROPGRO"],
))

_register(ResponseFunction(
    id="photoperiod",
    name="Photoperiod response",
    equation="f(P) = clamp01((P - Pc)/(Po - Pc))^n",
    description="Day-length sensitivity for floral induction.",
    param_names=["Pc", "Po", "n"],
    param_defaults=dict(Pc=12, Po=14.5, n=1),
    param_bounds=dict(Pc=(8, 16), Po=(10, 20), n=(0.5, 5)),
    x_label="Photoperiod (hours)", y_label="Photoperiod factor (0-1)",
    fn=_photoperiod, inverse_fn=_photoperiod_inv,
    eml_ops=_PHOTO_OPS, eml_K=_compute_K(_PHOTO_OPS),
    eml_note="2x subtract(11) + divide(17) + power(25) = K<=64",
    applicable_scales=["Fehr-Caviness", "BBCH", "Zadoks"],
    applicable_models=["DSSAT-CERES", "CROPGRO", "APSIM", "WOFOST"],
))

_register(ResponseFunction(
    id="vernalization",
    name="Vernalization accumulation",
    equation="Veff = 1 - (1 - Vd/Vsat)^2",
    description="Cold-day accumulation for floral competence.",
    param_names=["Vsat"],
    param_defaults=dict(Vsat=45),
    param_bounds=dict(Vsat=(5, 120)),
    x_label="Vernalization days", y_label="Vernalization factor (0-1)",
    fn=_vernalization, inverse_fn=_vernalization_inv,
    eml_ops=_VERN_OPS, eml_K=_compute_K(_VERN_OPS),
    eml_note="divide(17) + subtract(11) + square(17) + subtract(11) = K<=56",
    applicable_scales=["Zadoks", "BBCH", "Feekes", "Kuperman"],
    applicable_models=["DSSAT-CERES", "APSIM", "WOFOST"],
))

_register(ResponseFunction(
    id="water_stress",
    name="Water stress factor",
    equation="Ks = clamp01((theta - wp)/(fc - wp))",
    description="Linear soil moisture response.",
    param_names=["wp", "fc"],
    param_defaults=dict(wp=0.12, fc=0.33),
    param_bounds=dict(wp=(0.01, 0.3), fc=(0.1, 0.5)),
    x_label="Soil water (cm3/cm3)", y_label="Stress factor (0-1)",
    fn=_water_stress, inverse_fn=_water_stress_inv,
    eml_ops=_WATER_OPS, eml_K=_compute_K(_WATER_OPS),
    eml_note="2x subtract(11) + divide(17) = K<=39",
    applicable_scales=["COTMAN", "GOSSYM", "Fehr-Caviness"],
    applicable_models=["GOSSYM", "DSSAT-CERES", "APSIM", "WOFOST", "CROPGRO"],
))

_register(ResponseFunction(
    id="n_stress",
    name="Nitrogen limitation",
    equation="fN = 1 - exp(-k * (N-Nmin)/(Ncrit-Nmin))",
    description="Exponential N-response for development rate.",
    param_names=["Nmin", "Ncrit", "k"],
    param_defaults=dict(Nmin=0.01, Ncrit=0.04, k=3.5),
    param_bounds=dict(Nmin=(0, 0.05), Ncrit=(0.01, 0.1), k=(0.5, 10)),
    x_label="Leaf N (g/g)", y_label="N factor (0-1)",
    fn=_n_stress, inverse_fn=_n_stress_inv,
    eml_ops=_NSTR_OPS, eml_K=_compute_K(_NSTR_OPS),
    eml_note="normalize(11+11+17) + scale(17) + negate(15) + exp(3) + subtract(11) = K<=85",
    applicable_scales=["BBCH", "Zadoks", "COTMAN", "GOSSYM"],
    applicable_models=["DSSAT-CERES", "GOSSYM", "APSIM", "WOFOST"],
))

_register(ResponseFunction(
    id="gaussian",
    name="Gaussian optimum",
    equation="f(T) = exp(-(T-To)^2 / (2*sigma^2))",
    description="Symmetric bell for photosynthesis, pollen viability.",
    param_names=["To", "sigma"],
    param_defaults=dict(To=28, sigma=6),
    param_bounds=dict(To=(5, 45), sigma=(1, 20)),
    x_label="Temperature (C)", y_label="Response (0-1)",
    fn=_gaussian, inverse_fn=_gaussian_inv,
    eml_ops=_GAUSS_OPS, eml_K=_compute_K(_GAUSS_OPS),
    eml_note="subtract(11) + square(17) + multiply(17) + divide(17) + negate(15) + exp(3) = K<=80",
    applicable_scales=["Fehr-Caviness", "GOSSYM", "BBCH"],
    applicable_models=["GOSSYM", "CROPGRO", "APSIM"],
))

_register(ResponseFunction(
    id="gompertz",
    name="Gompertz growth",
    equation="f(t) = A * exp(-b * exp(-c*t))",
    description="Asymmetric sigmoid. LAI, dry matter, fruit sizing.",
    param_names=["A", "b", "c"],
    param_defaults=dict(A=1, b=3, c=0.03),
    param_bounds=dict(A=(0.01, 100), b=(0.1, 20), c=(1e-5, 1)),
    x_label="Thermal time", y_label="Growth fraction",
    fn=_gompertz, inverse_fn=_gompertz_inv,
    eml_ops=_GOMP_OPS, eml_K=_compute_K(_GOMP_OPS),
    eml_note="scale(17) + negate(15) + exp(3) + scale(17) + negate(15) + exp(3) + scale(17) = K<=87",
    applicable_scales=["Fehr-Caviness", "COTMAN", "Keller-Baglioni"],
    applicable_models=["GOSSYM", "CROPGRO", "APSIM"],
))

_register(ResponseFunction(
    id="broken_stick",
    name="Broken-stick thermal",
    equation="f(T) = min((T-Tb)/(To-Tb), (Tc-T)/(Tc-To), 1)",
    description="Triangular/trapezoidal thermal response.",
    param_names=["Tb", "To", "Tc"],
    param_defaults=dict(Tb=0, To=25, Tc=35),
    param_bounds=dict(Tb=(-10, 15), To=(15, 35), Tc=(25, 50)),
    x_label="Temperature (C)", y_label="Development rate (0-1)",
    fn=_broken_stick, inverse_fn=_broken_stick_inv,
    eml_ops=_BSTICK_OPS, eml_K=_compute_K(_BSTICK_OPS),
    eml_note="Each ramp: subtract+divide (K~28). Piecewise min adds ~106. K<=162 upper bound.",
    applicable_scales=["BBCH", "Zadoks", "Feekes", "Kuperman"],
    applicable_models=["APSIM", "DSSAT-CERES", "WOFOST"],
))

_register(ResponseFunction(
    id="beer_lambert",
    name="Beer-Lambert / Monteith RUE",
    equation="dW/dt = RUE * (1 - exp(-k*LAI))",
    description="Light interception to biomass.",
    param_names=["RUE", "k"],
    param_defaults=dict(RUE=1.2, k=0.5),
    param_bounds=dict(RUE=(0.1, 5), k=(0.1, 2)),
    x_label="LAI", y_label="Biomass rate",
    fn=_beer_lambert, inverse_fn=_beer_lambert_inv,
    eml_ops=_BEER_OPS, eml_K=_compute_K(_BEER_OPS),
    eml_note="Same structure as exp saturation + extra multiply. K<=80",
    applicable_scales=["BBCH", "Zadoks", "Fehr-Caviness", "COTMAN"],
    applicable_models=["DSSAT-CERES", "APSIM", "WOFOST", "GOSSYM", "CROPGRO"],
))

_register(ResponseFunction(
    id="farquhar",
    name="Farquhar photosynthesis",
    equation="A = Vcmax*Ci/(Ci + Kc*(1+O/Ko)) - Rd",
    description="Biochemical CO2 assimilation (Michaelis-Menten).",
    param_names=["Vcmax", "Kc", "Ko", "O2", "Rd"],
    param_defaults=dict(Vcmax=120, Kc=300, Ko=300000, O2=210000, Rd=1.5),
    param_bounds=dict(Vcmax=(10, 400), Kc=(50, 1000), Ko=(1e4, 1e6), O2=(1e4, 5e5), Rd=(0, 10)),
    x_label="Ci (umol/mol)", y_label="Assimilation rate",
    fn=_farquhar, inverse_fn=_farquhar_inv,
    eml_ops=_FAQ_OPS, eml_K=_compute_K(_FAQ_OPS),
    eml_note="Michaelis-Menten rational function. K<=117",
    applicable_scales=["BBCH", "COTMAN", "GOSSYM"],
    applicable_models=["GOSSYM", "APSIM", "CROPGRO"],
))


def list_functions() -> List[str]:
    """Return all registered function IDs."""
    return list(RESPONSE_FUNCTIONS.keys())


def get_function(func_id: str) -> ResponseFunction:
    """Retrieve a response function by ID."""
    if func_id not in RESPONSE_FUNCTIONS:
        raise KeyError(f"Unknown function '{func_id}'. Available: {list_functions()}")
    return RESPONSE_FUNCTIONS[func_id]


def functions_for_scale(scale: str) -> List[ResponseFunction]:
    """Return all response functions applicable to a given phenological scale."""
    return [rf for rf in RESPONSE_FUNCTIONS.values() if scale in rf.applicable_scales]


def functions_for_model(model: str) -> List[ResponseFunction]:
    """Return all response functions used by a given crop model."""
    return [rf for rf in RESPONSE_FUNCTIONS.values() if model in rf.applicable_models]
