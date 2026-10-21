"""Section-aware chunking: merge tiny sections, split long ones on paragraph boundaries (with overlap),
one chunk per FAQ pair, plus one synthetic bilingual `car_card` chunk rendered from the CarRecord.

Sizes are in characters (Hebrew text; ~1,100 chars is roughly 250-350 tokens).
"""

from __future__ import annotations

import re

from autochat.ingestion.schemas import CarRecord, Chunk, ParsedArticle, Section

MIN_CHARS = 150  # smaller sections are merged into the previous one
MAX_CHARS = 1600  # larger sections are split
TARGET_CHARS = 1100  # packing target when splitting
OVERLAP_MAX_CHARS = 400  # carry the last paragraph into the next part if it is at most this long

# --------------------------------------------------------------------------- classification
# Heading rules, checked in order (first match wins). Verified against the headings of all 8 articles.
HEADING_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("overview", re.compile(r"^פתיח")),
    ("verdict", re.compile(r"יתרונות|חסרונות|סיכום|שורה תחתונה|השורה התחתונה|לסיכום|תמורה למחיר")),
    ("value", re.compile(r"מחיר|עולה|כמה עולה|אבזור|כוללת|כולל")),
    ("efficiency_range", re.compile(r"טווח|צריכ|טעינ")),
    ("trunk", re.compile(r"מטען|החלק הפרקטי")),
    ("cockpit_tech", re.compile(r"סביבת נהג|סביבת הנהג|הנהג מסתדר|קבלת פנים|מולטימדיה|מסך|תא הנהג")),
    ("interior_space", re.compile(r"נוסעים|מאחור|אחורי|מרווח|מושבים")),
    ("design", re.compile(r"עיצוב|נראות|נראה בחוץ|היפה")),
    ("performance", re.compile(r"מנוע|ביצוע|תאוצה|נתונים|ממדים|שיאים|כוח")),
    ("ride_handling", re.compile(r"נוחות|התנהגות|נוח|בולמים|מתלים|כביש|שבילים|שטח")),
    ("equipment_safety", re.compile(r"בטיחות")),
    ("overview", re.compile(r"הקשר|פרטים|מהו הדגם|המותג והדגם|היצע|התחדש|נותר כשהיה|מה זה")),
]

# Body keywords for headings that say nothing ("שוברת השיאים") and for size-split parts with no heading.
TEXT_KEYWORDS: dict[str, re.Pattern[str]] = {
    "performance": re.compile(r'תאוצה|כ"ס|כוחות סוס|הספק|מומנט|מנוע|תיבת|הילוכ|קמ"ש'),
    "ride_handling": re.compile(r"נוחות|שיכוך|מתלים|בולמ|היגוי|הגה|פניות|אחיזה|התנהגות"),
    "cockpit_tech": re.compile(r"מסך|מולטימדיה|לוח שעונים|אפליקצ|ממשק|כפתור|מערכת המידע"),
    "interior_space": re.compile(r"מרווח|ברכיים|ראש|מושב אחורי|ספסל|נוסעים"),
    "trunk": re.compile(r"מטען|ליטר"),
    "efficiency_range": re.compile(r'טווח|צריכ|טעינ|קוט"ש|סוללה|ק"מ לליטר'),
    "design": re.compile(r"עיצוב|קווים|חרטום|פנסים|חישוקים|מראה"),
    "value": re.compile(r'מחיר|₪|ש"ח|שקל|תמורה|אבזור|מתחרים'),
    "equipment_safety": re.compile(r"בטיחות|כריות|מערכות עזר|בלימה אוטונומית"),
}

# User-facing aspects (match UserProfile.priorities / CarRecord.aspect_scores).
ASPECT_KEYWORDS: dict[str, re.Pattern[str]] = {
    "performance": re.compile(r'ביצוע|תאוצה|כ"ס|הספק|מנוע|מומנט'),
    "comfort": re.compile(r"נוחות|נוח|שיכוך|רעש|בידוד|שקט"),
    "practicality": re.compile(r"מטען|מרווח|אחסון|תאי אחסון|נוסעים|ליטר"),
    "tech": re.compile(r"מסך|מולטימדיה|דיגיטל|אפליקצ|מערכת|טכנולוג"),
    "quality": re.compile(r"איכות|חומרים|גימור|הרכבה|יוקרה|פרימיום"),
    "value": re.compile(r'מחיר|תמורה|₪|ש"ח|אבזור|זול|יקר'),
    "efficiency": re.compile(r'צריכ|טווח|טעינ|קוט"ש|ק"מ לליטר|חסכונ'),
    "fun": re.compile(r"כיף|הנאה|ספורטיב|מהנה|חיוך|דינמי"),
}
SECTION_TO_ASPECT = {
    "performance": "performance",
    "ride_handling": "comfort",
    "interior_space": "practicality",
    "trunk": "practicality",
    "cockpit_tech": "tech",
    "value": "value",
    "efficiency_range": "efficiency",
}


def classify_section(heading: str, text: str, kind: str) -> str:
    for section_type, pattern in HEADING_RULES:
        if pattern.search(heading):
            return section_type
    counts = {st: len(p.findall(text)) for st, p in TEXT_KEYWORDS.items()}
    best, hits = max(counts.items(), key=lambda kv: kv[1])
    if hits >= 2:
        return best
    return "overview" if kind == "intro" else "general"


def detect_aspects(heading: str, text: str, section_type: str) -> list[str]:
    found = {a for a, p in ASPECT_KEYWORDS.items() if p.search(heading) or len(p.findall(text)) >= 2}
    if section_type in SECTION_TO_ASPECT:
        found.add(SECTION_TO_ASPECT[section_type])
    return sorted(found)


# --------------------------------------------------------------------------- sizing
def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    return [p for p in parts if p.strip()]


def _explode(paragraphs: list[str]) -> list[str]:
    """Break single paragraphs longer than MAX_CHARS into sentence groups."""
    out: list[str] = []
    for p in paragraphs:
        if len(p) <= MAX_CHARS:
            out.append(p)
            continue
        buf = ""
        for sentence in _split_sentences(p):
            if buf and len(buf) + len(sentence) + 1 > TARGET_CHARS:
                out.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            out.append(buf)
    return out


def split_paragraphs(paragraphs: list[str]) -> list[list[str]]:
    """Greedy pack paragraphs into parts of about TARGET_CHARS, overlapping one short paragraph."""
    paragraphs = _explode(paragraphs)
    if sum(len(p) for p in paragraphs) <= MAX_CHARS:
        return [paragraphs]
    parts: list[list[str]] = []
    current: list[str] = []
    size = 0
    for p in paragraphs:
        if current and size + len(p) > TARGET_CHARS:
            parts.append(current)
            carry = current[-1] if len(current[-1]) <= OVERLAP_MAX_CHARS and len(current) > 1 else None
            current = [carry] if carry else []
            size = len(carry) if carry else 0
        current.append(p)
        size += len(p)
    if current:
        # avoid a tiny tail: fold it into the previous part
        if parts and size < MIN_CHARS * 2:
            parts[-1].extend(x for x in current if x not in parts[-1])
        else:
            parts.append(current)
    return parts


def merge_small(sections: list[Section]) -> list[Section]:
    merged: list[Section] = []
    for s in sections:
        if merged and len(s.text) < MIN_CHARS and len(merged[-1].text) + len(s.text) <= MAX_CHARS:
            prev = merged[-1]
            paragraphs = [*prev.paragraphs, s.heading, *s.paragraphs]
            merged[-1] = Section(heading=prev.heading, kind=prev.kind, paragraphs=paragraphs)
        else:
            merged.append(s)
    return merged


# --------------------------------------------------------------------------- headers + cards
def _fmt_int(n: int | float | None) -> str:
    return f"{int(n):,}" if n is not None else ""


def make_header(car: CarRecord, heading: str) -> str:
    name = " ".join(p for p in (car.make, car.model, str(car.model_year) if car.model_year else "") if p)
    facts = ", ".join(
        p for p in (car.body_type, car.powertrain, f"₪{_fmt_int(car.price_ils)}" if car.price_ils else None) if p
    )
    return " | ".join(p for p in (name, heading, facts) if p)


def _spec_lines(car: CarRecord) -> list[tuple[str, str, str]]:
    """(label_he, label_en, value) for the fields present on the record."""
    rows = [
        ("מחיר", "Price", f"₪{_fmt_int(car.price_ils)}" if car.price_ils else None),
        ("החזר חודשי", "Monthly payment", f"₪{_fmt_int(car.monthly_payment_ils)}" if car.monthly_payment_ils else None),
        ("הנעה", "Powertrain", car.powertrain),
        ("מרכב", "Body type", car.body_type),
        ("סגמנט", "Segment", car.segment),
        ("הנעה לגלגלים", "Drivetrain", car.drivetrain),
        ("תיבת הילוכים", "Transmission", car.transmission),
        ("מושבים", "Seats", car.seats),
        ('הספק (כ"ס)', "Power (hp)", car.power_hp),
        ('מומנט (נ"מ)', "Torque (Nm)", car.torque_nm),
        ('0-100 קמ"ש (שניות)', "0-100 km/h (s)", car.accel_0_100_s),
        ('מהירות מרבית (קמ"ש)', "Top speed (km/h)", car.top_speed_kmh),
        ('סוללה (קוט"ש)', "Battery (kWh)", car.battery_kwh),
        ('טווח WLTP (ק"מ)', "WLTP range (km)", car.range_wltp_km),
        ('טווח בפועל במבחן (ק"מ)', "Real-world range in test (km)", car.range_real_km),
        ("טעינה מהירה DC (קילוואט)", "DC fast charging (kW)", car.dc_charge_kw),
        ('צריכת דלק (ק"מ לליטר)', "Fuel economy (km/l)", car.fuel_km_per_l),
        ('צריכת חשמל (קוט"ש ל-100 ק"מ)', "Energy use (kWh/100km)", car.energy_kwh_per_100km),
        ("תא מטען (ליטר)", "Trunk (l)", car.trunk_l),
        ('אורך (מ"מ)', "Length (mm)", car.length_mm),
        ('בסיס גלגלים (מ"מ)', "Wheelbase (mm)", car.wheelbase_mm),
    ]
    return [(he, en, str(v)) for he, en, v in rows if v not in (None, "")]


def car_card_texts(car: CarRecord) -> tuple[str, str]:
    lines = _spec_lines(car)
    review = "מבחן ארוך טווח" if car.review_type == "long_term" else "מבחן דרכים"
    he = [f"כרטיס רכב: {car.make_he or car.make} {car.model_he or car.model} ({review}, {car.review_date or ''})"]
    he += [f"{label}: {value}" for label, _, value in lines]
    if car.pros_he:
        he.append("יתרונות: " + ", ".join(car.pros_he))
    if car.cons_he:
        he.append("חסרונות: " + ", ".join(car.cons_he))
    if car.verdict_he:
        he.append("שורה תחתונה: " + car.verdict_he)
    review_en = "long-term report" if car.review_type == "long_term" else "road test"
    en = [f"Car card: {car.make} {car.model} ({review_en}, {car.review_date or ''})"]
    en += [f"{label}: {value}" for _, label, value in lines]
    if car.pros_en:
        en.append("Pros: " + ", ".join(car.pros_en))
    if car.cons_en:
        en.append("Cons: " + ", ".join(car.cons_en))
    if car.verdict_en:
        en.append("Verdict: " + car.verdict_en)
    if not car.price_ils:
        en.append("Note: the review does not state a price.")
    return "\n".join(he), "\n".join(en)


# --------------------------------------------------------------------------- main
def _base(car: CarRecord) -> dict:
    return {
        "car_id": car.car_id,
        "url": car.url,
        "make": car.make,
        "model": car.model,
        "model_year": car.model_year,
        "body_type": car.body_type,
        "powertrain": car.powertrain,
        "price_ils": car.price_ils,
        "review_date": car.review_date,
    }


def build_chunks(article: ParsedArticle, car: CarRecord) -> list[Chunk]:
    chunks: list[Chunk] = []
    base = _base(car)
    order = 0
    sections = merge_small(article.sections)
    only_intro = len(sections) == 1 and sections[0].kind == "intro"
    for s_idx, section in enumerate(sections):
        parts = split_paragraphs(section.paragraphs)
        for p_idx, paragraphs in enumerate(parts):
            text = "\n".join(paragraphs)
            heading = section.heading
            if only_intro and len(parts) > 1:  # no headings at all (long-term report, Elantra N)
                heading = f"{article.title} ({p_idx + 1}/{len(parts)})"
            section_type = classify_section(section.heading if not only_intro else "", text, section.kind)
            if only_intro and p_idx == 0:
                section_type = "overview"
            chunks.append(
                Chunk(
                    id=f"{car.car_id}:{s_idx:02d}:{p_idx}",
                    chunk_type="section",
                    section_type=section_type,
                    aspects=detect_aspects(heading, text, section_type),
                    order=order,
                    heading=heading,
                    header=make_header(car, heading),
                    text_he=text,
                    **base,
                )
            )
            order += 1
    for f_idx, item in enumerate(article.faq):
        text = f"שאלה: {item.q}\nתשובה: {item.a}"
        chunks.append(
            Chunk(
                id=f"{car.car_id}:faq:{f_idx:02d}",
                chunk_type="faq",
                section_type="faq",
                aspects=detect_aspects(item.q, item.a, "faq"),
                order=order,
                heading=item.q,
                header=make_header(car, item.q),
                text_he=text,
                **base,
            )
        )
        order += 1
    card_he, card_en = car_card_texts(car)
    chunks.append(
        Chunk(
            id=f"{car.car_id}:card",
            chunk_type="car_card",
            section_type="car_card",
            aspects=sorted(car.aspect_scores),
            order=order,
            heading="כרטיס רכב",
            heading_en="Car card",
            header=make_header(car, "Car card / כרטיס רכב"),
            text_he=card_he,
            text_en=card_en,
            **base,
        )
    )
    return chunks
