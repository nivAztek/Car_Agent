"""Canonical mapping + normalization for spec-table labels (verified on all 8 auto.co.il articles).

>>> normalize_specs({"מחיר": "₪168,888", '0 ל-100 קמ"ש (שניות)': "7.3"})
({'price_ils': 168888, 'accel_0_100_s': 7.3}, {})
"""

from __future__ import annotations

import re

# label (exact, whitespace-collapsed) -> (canonical_key, kind)
SPEC_LABEL_MAP: dict[str, tuple[str, str]] = {
    "מחיר": ("price_ils", "int"),
    "החזר חודשי": ("monthly_payment_ils", "int"),
    'צריכת דלק - ממוצעת (ק"מ/ליטר)': ("fuel_km_per_l", "float"),
    'טווח נסיעה חשמלי WLTP (ק"מ)': ("range_wltp_km", "int"),
    'טווח נסיעה בפועל (ק"מ)': ("range_real_km", "int"),
    '0 ל-100 קמ"ש (שניות)': ("accel_0_100_s", "float"),
    "הספק טעינה מהירה DC (קילוואט)": ("dc_charge_kw", "int"),
    "יתרונות": ("pros", "list"),
    "חסרונות": ("cons", "list"),
}

# Fuzzy fallbacks for label variants (checked when the exact match fails), in priority order.
SPEC_LABEL_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^מחיר"), "price_ils", "int"),
    (re.compile(r"החזר"), "monthly_payment_ils", "int"),
    (re.compile(r"צריכת\s*דלק"), "fuel_km_per_l", "float"),
    (re.compile(r"צריכת\s*(חשמל|אנרגיה)"), "energy_kwh_per_100km", "float"),
    (re.compile(r"טווח.*WLTP", re.I), "range_wltp_km", "int"),
    (re.compile(r"טווח.*בפועל"), "range_real_km", "int"),
    (re.compile(r"0\s*ל-?\s*100"), "accel_0_100_s", "float"),
    (re.compile(r"טעינה.*DC|DC.*טעינה", re.I), "dc_charge_kw", "int"),
    (re.compile(r"הספק"), "power_hp", "int"),
    (re.compile(r"מומנט"), "torque_nm", "int"),
    (re.compile(r"תא\s*מטען"), "trunk_l", "int"),
    (re.compile(r"סוללה"), "battery_kwh", "float"),
    (re.compile(r"^יתרונות"), "pros", "list"),
    (re.compile(r"^חסרונות"), "cons", "list"),
]

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("&quot;", '"').replace("\xa0", " ")).strip()


def parse_number(raw: str, kind: str = "float") -> int | float | None:
    match = _NUM_RE.search(raw.replace("\xa0", " "))
    if not match:
        return None
    value = float(match.group(0).replace(",", ""))
    return int(round(value)) if kind == "int" else value


def parse_list(raw: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,،;]", raw.replace("&#x27;", "'")) if p.strip()]


def resolve_label(label: str) -> tuple[str, str] | None:
    label = clean_label(label)
    if label in SPEC_LABEL_MAP:
        return SPEC_LABEL_MAP[label]
    for pattern, key, kind in SPEC_LABEL_PATTERNS:
        if pattern.search(label):
            return key, kind
    return None


def normalize_specs(raw_specs: dict[str, str]) -> tuple[dict[str, object], dict[str, str]]:
    """Return (normalized, unknown). `unknown` labels are logged and passed to the LLM extractor as text."""
    normalized: dict[str, object] = {}
    unknown: dict[str, str] = {}
    for label, raw in raw_specs.items():
        resolved = resolve_label(label)
        if resolved is None:
            unknown[clean_label(label)] = raw
            continue
        key, kind = resolved
        value = parse_list(raw) if kind == "list" else parse_number(raw, kind)
        if value is not None and key not in normalized:
            normalized[key] = value
    return normalized, unknown


def infer_powertrain(norm: dict[str, object]) -> str | None:
    """Heuristic from table fields only. PHEV pages report a WLTP-weighted fuel figure (Lynk & Co 01 = 111.1 km/l),
    so > 40 km/l means PHEV. 'ice' is weak evidence: the table cannot distinguish ICE from (mild) hybrids."""
    if "range_wltp_km" in norm or "dc_charge_kw" in norm:
        return "ev"
    fuel = norm.get("fuel_km_per_l")
    if isinstance(fuel, (int, float)):
        return "phev" if fuel > 40 else "ice"
    return None
