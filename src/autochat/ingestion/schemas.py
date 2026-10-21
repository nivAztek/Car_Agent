"""Pydantic models for every ingestion boundary (parsed article, LLM extraction, CarRecord, Chunk)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from autochat.names import car_name_keys

BodyType = Literal["hatchback", "sedan", "suv", "crossover", "wagon", "coupe", "convertible", "mpv", "pickup"]
Powertrain = Literal["ev", "ice", "hybrid", "phev", "mhev"]
Drivetrain = Literal["fwd", "rwd", "awd"]
Transmission = Literal["manual", "automatic", "dct", "cvt", "single_speed"]
Segment = Literal["mini", "small", "compact", "midsize", "large", "luxury"]
ReviewType = Literal["road_test", "long_term"]
Tag = Literal["family", "enthusiast", "city", "commuter", "long_trips", "offroad_light", "luxury", "budget", "tech"]
FieldSource = Literal["table", "llm", "derived"]

SECTION_TYPES = (
    "overview",
    "value",
    "design",
    "cockpit_tech",
    "interior_space",
    "trunk",
    "performance",
    "efficiency_range",
    "ride_handling",
    "equipment_safety",
    "verdict",
    "faq",
    "car_card",
    "general",
)


# --------------------------------------------------------------------------- parse
class Section(BaseModel):
    heading: str
    kind: str = Field(description="intro | h2 | h3 | bold | md")
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


class FaqItem(BaseModel):
    q: str
    a: str


class ParsedArticle(BaseModel):
    slug: str
    url: str
    adapter: str
    title: str
    subtitle: str | None = None
    review_date: str | None = None  # ISO date
    specs_raw: dict[str, str] = Field(default_factory=dict)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    faq: list[FaqItem] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    source_sha: str = ""

    @property
    def body_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)

    def full_text(self) -> str:
        parts = [self.title, self.subtitle or ""]
        parts += [f"## {s.heading}\n{s.text}" for s in self.sections]
        parts += [f"שאלה: {f.q}\nתשובה: {f.a}" for f in self.faq]
        return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- LLM extraction
class NumFact(BaseModel):
    value: float | None = Field(None, description="Number stated in the article, in the requested unit; null if absent")
    evidence_he: str | None = Field(
        None,
        description="Short VERBATIM Hebrew quote from the article that contains this number; null if value is null",
    )


class AspectScore(BaseModel):
    score: int | None = Field(None, ge=1, le=5, description="1=poor ... 5=excellent, per the reviewer's opinion")
    evidence_he: str | None = Field(None, description="Short VERBATIM Hebrew quote supporting the score")


class AspectScores(BaseModel):
    performance: AspectScore = Field(default_factory=AspectScore)
    comfort: AspectScore = Field(default_factory=AspectScore)
    practicality: AspectScore = Field(default_factory=AspectScore)
    tech: AspectScore = Field(default_factory=AspectScore)
    quality: AspectScore = Field(default_factory=AspectScore)
    value: AspectScore = Field(default_factory=AspectScore)
    efficiency: AspectScore = Field(default_factory=AspectScore)
    fun: AspectScore = Field(default_factory=AspectScore)


class CarExtraction(BaseModel):
    """What the LLM fills from clean article text. Numbers must carry a verbatim evidence quote."""

    make: str = Field(description="Manufacturer in English, e.g. 'MG', 'Hyundai', 'Lynk & Co', 'GAC'")
    model: str = Field(description="Model name in English, e.g. 'S6', 'Elantra N', '01', 'Aion HT'")
    make_he: str | None = Field(None, description="Manufacturer as written in Hebrew in the article")
    model_he: str | None = Field(None, description="Model as written in Hebrew in the article")
    trim: str | None = Field(None, description="Tested trim/version if stated")
    model_year: int | None = Field(None, description="Model year if stated or clearly implied")
    review_type: ReviewType = Field(description="long_term for a long-term report, else road_test")
    body_type: BodyType | None = None
    segment: Segment | None = None
    powertrain: Powertrain | None = Field(None, description="ev, ice, hybrid (full HEV), phev (plug-in), mhev")
    powertrain_evidence_he: str | None = None
    drivetrain: Drivetrain | None = None
    transmission: Transmission | None = None
    seats: NumFact = Field(default_factory=NumFact)
    power_hp: NumFact = Field(default_factory=NumFact, description='Total system power in hp (כ"ס)')
    torque_nm: NumFact = Field(default_factory=NumFact, description='Torque in Nm (קג"מ x 9.81 if given in kgm)')
    accel_0_100_s: NumFact = Field(default_factory=NumFact)
    top_speed_kmh: NumFact = Field(default_factory=NumFact)
    price_ils: NumFact = Field(default_factory=NumFact, description="Price in Israeli shekels (₪) of the tested car")
    battery_kwh: NumFact = Field(default_factory=NumFact)
    range_wltp_km: NumFact = Field(default_factory=NumFact)
    range_real_km: NumFact = Field(default_factory=NumFact, description="Range measured/estimated in the test")
    dc_charge_kw: NumFact = Field(default_factory=NumFact)
    fuel_km_per_l: NumFact = Field(default_factory=NumFact, description="Fuel economy in km per litre")
    energy_kwh_per_100km: NumFact = Field(default_factory=NumFact)
    trunk_l: NumFact = Field(default_factory=NumFact, description="Trunk volume in litres (seats up)")
    length_mm: NumFact = Field(default_factory=NumFact)
    wheelbase_mm: NumFact = Field(default_factory=NumFact)
    pros_he: list[str] = Field(default_factory=list, description="Reviewer's pros, short Hebrew phrases")
    cons_he: list[str] = Field(default_factory=list, description="Reviewer's cons, short Hebrew phrases")
    pros_en: list[str] = Field(default_factory=list, description="English translation of pros_he, same order")
    cons_en: list[str] = Field(default_factory=list, description="English translation of cons_he, same order")
    verdict_he: str = Field(description="2-3 sentence Hebrew summary of the reviewer's verdict")
    verdict_en: str = Field(description="English translation of verdict_he")
    aspect_scores: AspectScores = Field(default_factory=AspectScores)
    tags: list[Tag] = Field(default_factory=list, description="Who the car suits, per the review")


# --------------------------------------------------------------------------- outputs
class CarRecord(BaseModel):
    """One structured record per car (Cosmos `cars`, id == car_id). Metadata filtering runs on these fields."""

    id: str
    car_id: str
    doc_type: Literal["car"] = "car"
    url: str
    title: str
    review_date: str | None = None
    review_type: ReviewType = "road_test"
    make: str
    model: str
    make_he: str | None = None
    model_he: str | None = None
    aliases: list[str] = Field(default_factory=list)
    name_keys: list[str] = Field(
        default_factory=list, description="Derived exact-match keys for the name resolver (autochat.names)"
    )
    trim: str | None = None
    model_year: int | None = None
    body_type: BodyType | None = None
    segment: Segment | None = None
    powertrain: Powertrain | None = None
    drivetrain: Drivetrain | None = None
    transmission: Transmission | None = None
    seats: int | None = None
    price_ils: int | None = None
    monthly_payment_ils: int | None = None
    power_hp: int | None = None
    torque_nm: int | None = None
    accel_0_100_s: float | None = None
    top_speed_kmh: int | None = None
    battery_kwh: float | None = None
    range_wltp_km: int | None = None
    range_real_km: int | None = None
    dc_charge_kw: int | None = None
    fuel_km_per_l: float | None = None
    energy_kwh_per_100km: float | None = None
    trunk_l: int | None = None
    length_mm: int | None = None
    wheelbase_mm: int | None = None
    pros_he: list[str] = Field(default_factory=list)
    cons_he: list[str] = Field(default_factory=list)
    pros_en: list[str] = Field(default_factory=list)
    cons_en: list[str] = Field(default_factory=list)
    verdict_he: str = ""
    verdict_en: str = ""
    aspect_scores: dict[str, int] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict, description="field -> verbatim Hebrew quote")
    dropped: dict[str, str] = Field(default_factory=dict, description="LLM values rejected (no verified evidence)")
    adapter: str = ""
    source_sha: str = ""

    @model_validator(mode="after")
    def _derive_name_keys(self) -> CarRecord:
        self.name_keys = car_name_keys(self.model_dump(include={"make", "model", "make_he", "model_he", "aliases"}))
        return self


class Chunk(BaseModel):
    """Retrieval unit (Cosmos `chunks`, PK car_id). Filter fields are denormalized from the CarRecord."""

    id: str
    car_id: str
    doc_type: Literal["chunk"] = "chunk"
    chunk_type: Literal["section", "faq", "car_card"]
    section_type: str
    aspects: list[str] = Field(default_factory=list)
    order: int
    heading: str
    heading_en: str = ""
    header: str
    text_he: str
    text_en: str = ""
    url: str
    make: str
    model: str
    model_year: int | None = None
    body_type: str | None = None
    powertrain: str | None = None
    price_ils: int | None = None
    review_date: str | None = None

    def embed_text(self) -> str:
        return "\n".join(p for p in (self.header, self.text_he, self.text_en) if p)
