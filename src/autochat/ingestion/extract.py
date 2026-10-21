"""LLM structuring of clean article text + deterministic evidence verification + merge with the spec table.

Rules (see copilot-instructions "Structured beats generated"):
* Spec-table values always win over LLM values.
* Every LLM number must come with a verbatim Hebrew quote that (a) is found in the article and (b) contains the
  number. Otherwise the value is dropped (recorded in `CarRecord.dropped`).
* The LLM never sees HTML. It gets the parsed text plus the already-known table facts.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import AsyncRetrying

from autochat.ingestion.schemas import AspectScore, CarExtraction, CarRecord, NumFact, ParsedArticle
from autochat.ingestion.spec_labels import infer_powertrain, normalize_specs
from autochat.llm import RETRY_KW, chat_llm

log = logging.getLogger(__name__)

PROMPT_VERSION = "extract-v1"

SYSTEM_PROMPT = """You structure an Israeli Hebrew car road-test article (auto.co.il) into JSON.
Rules:
- Use ONLY the article text. If a value is not stated, return null. Never guess or use outside knowledge.
- For every number you return, `evidence_he` MUST be a short exact copy (5-25 words) of the article sentence
  fragment that contains that number, character-for-character, not paraphrased or translated.
- Convert units only when asked by the field description (e.g. torque kgm -> Nm); the quote still shows the original.
- `make`/`model` in English (Latin script) as the brand markets them; `make_he`/`model_he` as written in the article.
- powertrain: ev = battery electric only; phev = plug-in hybrid; hybrid = full self-charging hybrid;
  mhev = mild hybrid (48V); ice = petrol/diesel only. Quote the evidence in `powertrain_evidence_he`.
- aspect_scores reflect the REVIEWER's opinion (1 poor .. 5 excellent), each with a verbatim supporting quote;
  leave score null when the article does not discuss the aspect.
- If KNOWN pros/cons are given, copy them exactly into pros_he/cons_he and translate them in the same order.
- verdict_he: 2-3 sentences summarising the reviewer's conclusion in Hebrew; verdict_en: its English translation.
"""

# Fields that are NumFact on the extraction and numbers on the CarRecord: name -> int|float
NUM_FIELDS: dict[str, type] = {
    "seats": int,
    "power_hp": int,
    "torque_nm": int,
    "accel_0_100_s": float,
    "top_speed_kmh": int,
    "price_ils": int,
    "battery_kwh": float,
    "range_wltp_km": int,
    "range_real_km": int,
    "dc_charge_kw": int,
    "fuel_km_per_l": float,
    "energy_kwh_per_100km": float,
    "trunk_l": int,
    "length_mm": int,
    "wheelbase_mm": int,
}
TABLE_ONLY_FIELDS = ("monthly_payment_ils",)
CATEGORICAL_FIELDS = ("body_type", "segment", "drivetrain", "transmission", "trim", "model_year")

HE_NUMBER_WORDS = {
    "שניים": 2,
    "שתי": 2,
    "שני": 2,
    "שלוש": 3,
    "שלושה": 3,
    "ארבע": 4,
    "ארבעה": 4,
    "חמש": 5,
    "חמישה": 5,
    "שש": 6,
    "שישה": 6,
    "שבע": 7,
    "שבעה": 7,
    "שמונה": 8,
    "תשע": 9,
    "תשעה": 9,
}

# --------------------------------------------------------------------------- evidence verification
_PUNCT_RE = re.compile(r"[\"'״׳“”‘’`.,:;!?()\[\]{}\-–—/\\|*]+")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text.replace("\xa0", " "))).strip().lower()


def quote_in_article(quote: str | None, article_norm: str, min_coverage: float = 0.8) -> bool:
    """True when the quote appears verbatim (after normalization), or >= min_coverage of its tokens do."""
    if not quote:
        return False
    q = normalize_text(quote)
    if not q:
        return False
    if q in article_norm:
        return True
    tokens = [t for t in q.split() if len(t) > 1]
    if not tokens:
        return False
    vocab = set(article_norm.split())
    return sum(t in vocab for t in tokens) / len(tokens) >= min_coverage


def numbers_in(quote: str) -> list[float]:
    nums = [float(m.replace(",", "")) for m in _NUM_RE.findall(quote)]
    if "אלף" in quote or "אלפי" in quote:
        nums += [n * 1000 for n in nums]
    for word, n in HE_NUMBER_WORDS.items():
        if re.search(rf"(?<![א-ת]){word}(?![א-ת])", quote):
            nums.append(float(n))
    return nums


def number_in_quote(value: float, quote: str, field: str) -> bool:
    candidates = numbers_in(quote)
    if field == "torque_nm":  # article may state kgm
        candidates += [c * 9.81 for c in candidates]
    if field.endswith("_mm"):  # dimensions are often written in cm (ס"מ) or metres
        candidates += [c * 10 for c in candidates] + [c * 1000 for c in candidates if c < 10]
    return any(abs(c - value) <= max(0.02 * abs(value), 0.05) for c in candidates)


def verify_num(fact: NumFact, field: str, article_norm: str) -> tuple[float | None, str | None]:
    """Return (value, reason_if_dropped)."""
    if fact.value is None:
        return None, None
    if not quote_in_article(fact.evidence_he, article_norm):
        return None, f"{fact.value}: evidence not found in article ({fact.evidence_he!r})"
    if not number_in_quote(fact.value, fact.evidence_he or "", field):
        return None, f"{fact.value}: number not in evidence ({fact.evidence_he!r})"
    return fact.value, None


# --------------------------------------------------------------------------- LLM call
def build_messages(article: ParsedArticle, table: dict[str, object], pros: list[str], cons: list[str]) -> list:
    known = {"spec_table": table, "known_pros": pros, "known_cons": cons}
    user = (
        f"URL: {article.url}\nTITLE: {article.title}\nPUBLISHED: {article.review_date}\n"
        f"KNOWN FACTS (from the article's spec table, already verified):\n{json.dumps(known, ensure_ascii=False)}\n\n"
        f"ARTICLE TEXT:\n{article.full_text()}"
    )
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(user)]


async def llm_extract(
    article: ParsedArticle, table: dict[str, object], pros: list[str], cons: list[str]
) -> CarExtraction:
    structured = chat_llm("extract").with_structured_output(CarExtraction)
    messages = build_messages(article, table, pros, cons)
    async for attempt in AsyncRetrying(**RETRY_KW):
        with attempt:
            result = await structured.ainvoke(messages)
    if not isinstance(result, CarExtraction):
        result = CarExtraction.model_validate(result)
    return result


# --------------------------------------------------------------------------- merge
def _aliases(*names: str | None) -> list[str]:
    out: dict[str, None] = {}
    for name in names:
        if name:
            out.setdefault(re.sub(r"\s+", " ", name).strip().lower(), None)
    return list(out)


def _resolve_powertrain(
    table_pt: str | None, ex: CarExtraction, article_norm: str
) -> tuple[str | None, str | None, str | None]:
    """Return (powertrain, source, evidence)."""
    llm_pt = ex.powertrain
    evidence_ok = quote_in_article(ex.powertrain_evidence_he, article_norm)
    if table_pt in ("ev", "phev"):
        return table_pt, "table", None
    if table_pt == "ice":
        if llm_pt in ("hybrid", "mhev") and evidence_ok:
            return llm_pt, "llm", ex.powertrain_evidence_he
        return "ice", "table", None
    if llm_pt:
        return llm_pt, "llm", ex.powertrain_evidence_he if evidence_ok else None
    return None, None, None


def merge(article: ParsedArticle, ex: CarExtraction) -> CarRecord:
    table, _unknown = normalize_specs(article.specs_raw)
    article_norm = normalize_text(article.full_text() + "\n" + "\n".join(article.specs_raw.values()))
    sources: dict[str, str] = {}
    evidence: dict[str, str] = {}
    dropped: dict[str, str] = {}
    values: dict[str, object] = {}

    for field, typ in NUM_FIELDS.items():
        if field in table:
            values[field] = typ(table[field])
            sources[field] = "table"
            continue
        value, reason = verify_num(getattr(ex, field), field, article_norm)
        if value is not None:
            values[field] = int(round(value)) if typ is int else round(float(value), 2)
            sources[field] = "llm"
            evidence[field] = getattr(ex, field).evidence_he or ""
        elif reason:
            dropped[field] = reason
    for field in TABLE_ONLY_FIELDS:
        if field in table:
            values[field] = int(table[field])  # type: ignore[call-overload]
            sources[field] = "table"

    for field in CATEGORICAL_FIELDS:
        value = getattr(ex, field)
        if value is not None:
            values[field] = value
            sources[field] = "llm"

    powertrain, pt_source, pt_evidence = _resolve_powertrain(infer_powertrain(table), ex, article_norm)
    if powertrain:
        values["powertrain"] = powertrain
        sources["powertrain"] = pt_source or "llm"
        if pt_evidence:
            evidence["powertrain"] = pt_evidence

    # pros/cons: dedicated pros-cons table -> spec-table lists -> LLM
    table_pros = article.pros or list(table.get("pros", []) or [])  # type: ignore[arg-type]
    table_cons = article.cons or list(table.get("cons", []) or [])  # type: ignore[arg-type]
    pros_he, cons_he = (table_pros or ex.pros_he), (table_cons or ex.cons_he)
    sources["pros"] = "table" if table_pros else "llm"
    sources["cons"] = "table" if table_cons else "llm"
    pros_en = ex.pros_en if len(ex.pros_en) == len(pros_he) else []
    cons_en = ex.cons_en if len(ex.cons_en) == len(cons_he) else []

    aspect_scores: dict[str, int] = {}
    for aspect, score in ex.aspect_scores:
        assert isinstance(score, AspectScore)
        if score.score is None:
            continue
        if quote_in_article(score.evidence_he, article_norm):
            aspect_scores[aspect] = score.score
            evidence[f"aspect.{aspect}"] = score.evidence_he or ""
        else:
            dropped[f"aspect.{aspect}"] = f"{score.score}: evidence not found ({score.evidence_he!r})"

    long_term = "long-term" in article.slug or "ארוך" in article.title
    review_type = "long_term" if long_term else ex.review_type

    record = CarRecord(
        id=article.slug,
        car_id=article.slug,
        url=article.url,
        title=article.title,
        review_date=article.review_date,
        review_type=review_type,
        make=ex.make,
        model=ex.model,
        make_he=ex.make_he,
        model_he=ex.model_he,
        aliases=_aliases(
            ex.make,
            ex.model,
            f"{ex.make} {ex.model}",
            ex.model_he,
            f"{ex.make_he} {ex.model_he}" if ex.make_he and ex.model_he else None,
            article.slug.replace("-", " "),
        ),
        pros_he=pros_he,
        cons_he=cons_he,
        pros_en=pros_en,
        cons_en=cons_en,
        verdict_he=ex.verdict_he,
        verdict_en=ex.verdict_en,
        aspect_scores=aspect_scores,
        tags=list(dict.fromkeys(ex.tags)),
        field_sources=sources,  # type: ignore[arg-type]
        evidence=evidence,
        dropped=dropped,
        adapter=article.adapter,
        source_sha=article.source_sha,
        **values,  # type: ignore[arg-type]
    )
    if dropped:
        log.info("%s: dropped %d unverified LLM values: %s", article.slug, len(dropped), sorted(dropped))
    return record


async def extract_car(article: ParsedArticle) -> tuple[CarRecord, CarExtraction]:
    table, unknown = normalize_specs(article.specs_raw)
    if unknown:
        log.warning("%s: unknown spec labels passed to LLM as text: %s", article.slug, unknown)
    pros = article.pros or list(table.get("pros", []) or [])  # type: ignore[arg-type]
    cons = article.cons or list(table.get("cons", []) or [])  # type: ignore[arg-type]
    table_facts = {k: v for k, v in table.items() if k not in ("pros", "cons")}
    table_facts |= {f"raw:{k}": v for k, v in unknown.items()}
    extraction = await llm_extract(article, table_facts, pros, cons)
    return merge(article, extraction), extraction
