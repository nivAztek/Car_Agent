"""Offline ingestion tests: run against the cached HTML in data/raw (no network, no LLM, no Cosmos)."""

from __future__ import annotations

from functools import cache

import pytest

from autochat.ingestion.chunk import build_chunks, classify_section, split_paragraphs
from autochat.ingestion.extract import merge, normalize_text, number_in_quote, quote_in_article, verify_num
from autochat.ingestion.pipeline import content_sha
from autochat.ingestion.schemas import CarExtraction, CarRecord, NumFact, ParsedArticle
from autochat.ingestion.scrape import load_urls, raw_path, slug_of
from autochat.ingestion.sources import parse_html
from autochat.ingestion.spec_labels import infer_powertrain, normalize_specs, parse_number

URLS = {slug_of(u): u for u in load_urls()}
pytestmark = pytest.mark.skipif(
    len(URLS) < 8 or not all(raw_path(s).exists() for s in URLS),
    reason="cached HTML missing (run `dev.ps1 ingest --stage scrape`)",
)


@cache
def article(slug: str) -> ParsedArticle:
    return parse_html(raw_path(slug).read_text(encoding="utf-8"), URLS[slug], slug)


def stub_car(slug: str, **kw: object) -> CarRecord:
    a = article(slug)
    return CarRecord(id=slug, car_id=slug, url=a.url, title=a.title, make="X", model="Y", **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- spec table
def test_parse_number() -> None:
    assert parse_number("168,888 ₪", "int") == 168888
    assert parse_number("7.3 שניות") == 7.3
    assert parse_number("אין") is None


def test_mg_s6_spec_table() -> None:
    table, unknown = normalize_specs(article("mg-s6").specs_raw)
    assert not unknown
    assert table["price_ils"] == 168888
    assert table["accel_0_100_s"] == 7.3
    assert infer_powertrain(table) == "ev"


def test_lynk_is_phev_from_weighted_fuel_figure() -> None:
    table, _ = normalize_specs(article("link-and-co-01-2026").specs_raw)
    assert infer_powertrain(table) == "phev"


def test_no_unknown_labels_on_any_page() -> None:
    for slug in URLS:
        assert normalize_specs(article(slug).specs_raw)[1] == {}, slug


# --------------------------------------------------------------------------- parser
@pytest.mark.parametrize("slug", sorted(URLS))
def test_every_page_parses_with_site_adapter(slug: str) -> None:
    a = article(slug)
    assert a.adapter == "auto_co_il"
    assert a.title and a.review_date
    assert a.body_chars > 3000


def test_faq_and_pros_cons() -> None:
    assert len(article("mg-s6").faq) == 10
    assert article("mg-s6").pros and article("mg-s6").cons
    assert len(article("link-and-co-01-2026").faq) == 10


def test_kia_long_term_has_no_spec_table() -> None:
    assert article("long-term-report-kia-ev9-4").specs_raw == {}


def test_content_sha_is_stable() -> None:
    assert content_sha(article("mg-s6")) == content_sha(article("mg-s6").model_copy())


# --------------------------------------------------------------------------- evidence verification
def test_quote_in_article_tolerates_punctuation() -> None:
    norm = normalize_text('הטווח בפועל עומד על 430 ק"מ, וזה לא רע.')
    assert quote_in_article("הטווח בפועל עומד על 430 קמ", norm)
    assert not quote_in_article("הטווח הוא 600 קילומטר במבחן", norm)


@pytest.mark.parametrize(
    ("value", "quote", "field", "ok"),
    [
        (245, 'הספק של 245 כ"ס', "power_hp", True),
        (350, 'מומנט של 35.7 קג"מ', "torque_nm", True),  # kgm -> Nm
        (4710, 'אורכו 471 ס"מ', "length_mm", True),  # cm -> mm
        (5, "מרווח לרגליים סביר", "seats", False),
        (7, "שבעה מושבים", "seats", True),
    ],
)
def test_number_in_quote(value: float, quote: str, field: str, ok: bool) -> None:
    assert number_in_quote(value, quote, field) is ok


def test_verify_num_rejects_invented_quote() -> None:
    norm = normalize_text("המכונית מהירה מאוד")
    value, reason = verify_num(NumFact(value=300, evidence_he='מהירות מרבית 300 קמ"ש'), "top_speed_kmh", norm)
    assert value is None and reason


def test_merge_table_beats_llm_and_drops_unverified() -> None:
    a = article("mg-s6")
    ex = CarExtraction(
        make="MG",
        model="S6",
        review_type="road_test",
        powertrain="ice",
        verdict_he="x",
        verdict_en="x",
        price_ils=NumFact(value=1, evidence_he="מחיר 1"),
        power_hp=NumFact(value=999, evidence_he="המצאה מוחלטת 999"),
    )
    car = merge(a, ex)
    assert car.price_ils == 168888 and car.field_sources["price_ils"] == "table"
    assert car.powertrain == "ev"  # table-inferred EV wins over the LLM
    assert car.power_hp is None and "power_hp" in car.dropped


# --------------------------------------------------------------------------- chunking
def test_classify_section() -> None:
    assert classify_section("פתיח", "מחיר מחיר מחיר", "intro") == "overview"
    assert classify_section("איך תא המטען שלו?", "", "h2") == "trunk"
    assert classify_section("סביבת נהג, נוסעים", "", "h2") == "cockpit_tech"
    assert classify_section("שוברת השיאים", 'תאוצה של 3.8 שניות, 400 כ"ס ומנוע', "h2") == "performance"
    assert classify_section("כותרת", "טקסט", "h2") == "general"


def test_split_paragraphs_respects_max_and_overlaps() -> None:
    paras = [f"פסקה {i} " + "מילה " * 60 for i in range(12)]
    parts = split_paragraphs(paras)
    assert len(parts) > 1
    assert all(sum(len(p) for p in part) <= 1600 for part in parts)
    assert parts[0][-1] == parts[1][0]  # one-paragraph overlap


@pytest.mark.parametrize("slug", sorted(URLS))
def test_build_chunks(slug: str) -> None:
    a = article(slug)
    chunks = build_chunks(a, stub_car(slug, price_ils=100000, powertrain="ev"))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.car_id == slug and c.url == a.url for c in chunks)
    assert sum(c.chunk_type == "car_card" for c in chunks) == 1
    assert sum(c.chunk_type == "faq" for c in chunks) == len(a.faq)
    sections = [c for c in chunks if c.chunk_type == "section"]
    assert len(sections) >= 4  # single-section bodies (Kia, Elantra) get split
    assert max(len(c.text_he) for c in sections) <= 1800
    assert "₪100,000" in chunks[0].header


def test_car_card_is_bilingual() -> None:
    card = build_chunks(article("mg-s6"), stub_car("mg-s6", price_ils=168888, pros_he=["תמורה"], pros_en=["Value"]))[-1]
    assert "₪168,888" in card.text_he and "Price: ₪168,888" in card.text_en
    assert "Pros: Value" in card.text_en
