---
name: autochat-ingestion
description: "Scrape, parse, structure, chunk, translate, embed and load the 8 auto.co.il Hebrew road-test articles into Cosmos DB. Use when working on src/autochat/ingestion, when a page parses with missing fields, when adding metadata fields for filtering, or when re-running ingestion."
---

# autochat-ingestion

Turns 8 unstructured Hebrew articles into:
- `cars`: one structured **CarRecord** per car, used for metadata filtering and exact spec answers.
- `chunks`: section-level bilingual chunks with embeddings and denormalized filter fields.

## Scripts (run from repo root)

| Script | Purpose |
|---|---|
| `scripts/probe_articles.py [--slug S] [--refresh] [--quiet]` | Download (cached) + report title, date, spec table, pros/cons, FAQ, sections. Writes `data/raw/_probe_summary.json`. **Run first, and after any parser change.** |
| `scripts/inspect_dom.py <slug>` | Dump headings and the `<p>`-heavy containers of a cached page. Use when a page yields 0 sections or the layout changes. |
| `scripts/spec_labels.py` | `SPEC_LABEL_MAP`, `normalize_specs()`, `infer_powertrain()`. Run it to normalize every cached page and list unknown labels. Copy into `src/autochat/ingestion/spec_labels.py`. |

```powershell
.venv\Scripts\python.exe .github/skills/autochat-ingestion/scripts/probe_articles.py --quiet
.venv\Scripts\python.exe .github/skills/autochat-ingestion/scripts/spec_labels.py
```

## Verified page facts (probed 2026-10-16)

| slug | specs | sections | FAQ | date | notes |
|---|---|---|---|---|---|
| citroen-c3-2026 | 6 | 7 | 0 | 2026-01-27 | ICE, 20 km/l |
| audi-rs3-fl | 5 | 8 | 0 | 2026-01-22 | No fuel row, so the **LLM must set powertrain** (ICE) |
| long-term-report-kia-ev9-4 | **0** | 1 (bold-subheads/size split) | 0 | 2026-01-22 | **No spec table.** Long-term report #4, so all fields come from LLM extraction. `review_type=long_term` |
| mg-s6 | 8 | 12 | 10 | 2026-01-13 | EV, WLTP 530 / real 430, DC 144 kW |
| hyundai-elantra-n-manual | 6 | 1 (needs split) | 10 | 2026-01-15 | ICE, manual. Body has no H2, so split by size |
| aion-ht | 8 | 9 | 0 | 2026-01-12 | EV, WLTP 445 / real 370, DC 163 kW |
| link-and-co-01-2026 | 6 | 7 | 10 | 2026-01-14 | **PHEV**: 111.1 "km/l" is a WLTP-weighted figure, not real consumption |
| genesis-gv80-2026 | 6 | 9 | 0 | 2026-01-08 | ICE, 8.1 km/l |

## Selectors (selectolax **lexbor** backend)

```python
from selectolax.lexbor import LexborHTMLParser  # selectolax.parser raises ImportError in 1.x
title      = meta og:title                           # e.g. "MG S6 – מבחן דרכים"
summary    = meta og:description
date       = tree.css_first("time").attributes["datetime"]
body       = tree.css("div.sections__items .text-wrapper")   # walk in order; recurse into div/span/figure
section    = new on h2/h3, or on <p> whose text is <=90 chars and >=80% inside <strong>/<b>
specs      = .vehicle-table__item / .table-item -> .table-item__title + .table-item__value
pros/cons  = also in specs as "יתרונות"/"חסרונות" (comma separated); .pros-cons-table__col has longer text
faq        = div.faq__container details.spollers__item -> summary (Q) + .spollers__body (A)
boilerplate H2 to drop: הצטרפו לוואטסאפ | כתבות אחרונות | כתבות באותם נושאים | מותגים מתחרים | המומחה האובייקטיבי
```

Fetching: use a browser UA plus `Accept-Language: he-IL`. Fetch tools without a UA get redirected to `fls.doubleclick.net`, so treat any final URL off `auto.co.il` as an error.

## Pipeline stages (`python -m autochat.ingestion.pipeline [--stage X] [--force]`)

1. **scrape**: fetch to `data/raw/<slug>.html`.
2. **parse**: produce `ParsedArticle{slug,url,title,subtitle,date,specs_raw,pros,cons,faq[],sections[]}` in `data/processed/parsed/<slug>.json`.
3. **structure**:
   - Deterministic: `normalize_specs()` + `infer_powertrain()` give **table facts** (source=`table`).
   - LLM (fast deployment, `with_structured_output(CarExtraction)`, temperature 0). Input: title, subtitle, all section text, and the table facts marked "already known, do not change". Output:
     `make, model, trim, model_year, body_type, segment, powertrain, drivetrain, transmission, seats, power_hp, torque_nm, battery_kwh, trunk_l, verdict_he, verdict_en, tags[], aspect_scores{performance,comfort,practicality,tech,quality,value,efficiency,fun: {score 1-5, evidence_he}}`. Each numeric field also carries `evidence_he` or is null.
   - Merge: table wins and LLM fills gaps. Record per-field `source` in `extraction_meta`. Write `data/processed/cars.json`.
4. **chunk**: sections become chunks.
   - Merge sections under 120 tokens into the next one. Split sections over 600 tokens on paragraph boundaries with 1-paragraph overlap. This is how the Elantra and Kia single-section bodies get split.
   - Each FAQ pair becomes its own chunk (`section_type=faq`).
   - Add one synthetic `car_card` chunk per car, rendered from the CarRecord (specs + pros/cons + verdict). This is the best hit for spec questions.
   - `section_type` is classified by heading keywords. Map: `price|מחיר|תמורה` → value, `ביצוע|מנוע|תאוצה` → performance, `נוחות|התנהגות|כביש` → ride_handling, `מושב|אחור|מרווח` → interior_space, `מטען` → trunk, `נהג|תפעול|מולטימדיה|מסך` → cockpit_tech, `עיצוב` → design, `טווח|טעינה|צריכ` → efficiency_range, `אבזור|בטיחות` → equipment_safety, `יתרונות|חסרונות|סיכום|שורה תחתונה` → verdict, otherwise `general`.
5. **translate**: HE→EN per chunk (fast model, batches of 5, keep numbers and model names verbatim). Cached by content hash in `data/processed/cache/translations.json`.
6. **embed**: embed `header + "\n" + text_he + "\n" + text_en`, batch 16, `dimensions=EMBED_DIMENSIONS`. Cached by (deployment, dims, text) hash in `data/processed/cache/embeddings.jsonl`.
7. **load**: bootstrap containers (see the `cosmos-emulator` skill), upsert `cars` and `chunks`, and delete stale chunk ids of re-ingested cars.

Implementation notes (as built):
- CLI flags: `--stage`, `--force` (re-run the LLM extraction), `--refresh` (re-download), `--only <slug...>`, `--url <url...>`.
- The raw LLM extraction is cached in `data/processed/extractions/<slug>.json` with `prompt_version`. Changing the merge rules re-merges without calling the LLM. Bumping `extract.PROMPT_VERSION` or changed content (manifest `content_sha256` of the parsed article, not the raw HTML) triggers re-extraction.
- Chunk sizes are measured in chars: merge sections under 150 chars into the previous section, split sections over 1,600 chars at about 1,100 chars. Section type comes from heading rules first, then body-keyword counts (≥2 hits), so the creative Audi headings classify correctly. The intro (`פתיח`) is always `overview`.
- Unit-aware evidence checks: kgm→Nm, cm/m→mm, Hebrew number words. LLM values without a matching quote go to `CarRecord.dropped`; this is expected for guessed `seats`.
- Result on the 8 articles: 110 chunks (72 section, 30 FAQ, 8 car cards), all translated and embedded.

## Chunk document shape (Cosmos `chunks`)

```json
{"id":"mg-s6:03:0","car_id":"mg-s6","doc_type":"chunk","section_type":"ride_handling",
 "heading":"איך הנוחות וההתנהגות של MG S6?","header":"MG S6 2026 | ... | suv, ev, ₪168,888",
 "text_he":"...","text_en":"...","url":"https://www.auto.co.il/...",
 "make":"MG","model":"S6","model_year":2026,"body_type":"suv","powertrain":"ev","price_ils":168888,
 "embedding":[...1536 floats...]}
```

## Adding sources (flexible ingestion)

**Decision: use deterministic parsing first and the LLM for structuring only. Never use an LLM as the HTML parser.** Raw HTML is about 50–100k tokens per page, costly, non-deterministic, and invites invented numbers. The LLM adds value where it already runs: turning clean text into `CarExtraction` with evidence quotes.

Verified on the 8 cached pages (2026-10-17):

| Extractor | Body text | Spec-table values (price etc.) | Sections | Date |
|---|---|---|---|---|
| `AutoCoIlAdapter` (site selectors) | 53.9k chars | ✅ deterministic (0 on Kia) | ✅ h2/h3/bold | ✅ |
| `trafilatura` 2.3 (generic) | 65.3k chars (×1.21, includes FAQ and captions), 0 boilerplate | ❌ **lost** (div-based table, e.g. ₪168,888 missing) | markdown headings | ✅ |

Design (`src/autochat/ingestion/sources/`):
- `SourceAdapter` protocol: `matches(url) -> bool`, `parse(html, url) -> ParsedArticle`. A registry is checked in order.
- `AutoCoIlAdapter`: the verified selectors above. It yields `specs_raw` (table facts, `source=table`).
- `GenericAdapter`: `trafilatura.extract(html, output_format="markdown", include_tables=True, favor_recall=True)` plus `extract_metadata` (title, date). Sections are split on markdown headings, with size-split as fallback. `specs_raw={}`, so every field comes from LLM extraction with evidence (same path as Kia EV9).
- **Quality gate**: if a site adapter returns 0 sections or under 1,500 body chars, log `layout drift` and re-parse with `GenericAdapter`. A site redesign degrades the data instead of breaking ingestion.
- **Add a URL**: append it to `data/urls.txt` (or `dev.ps1 ingest --url <u>`). `data/processed/manifest.json` keeps `{url, slug, content_sha256, adapter, stages_done}`. Only new or changed URLs go through LLM, translate and embed, and are upserted to Cosmos. Each car is its own partition, so no other data is touched. The API rebuilds BM25 on startup (or `POST /admin/reload`).
- **Nothing is hard-coded to 8 cars**: the `analyze_turn` car alias list, the UI catalog and `/cars` are read from the `cars` container. `car_id` = URL slug.
- Out of scope for the POC: deduping several reviews of the same model (future: `model_key = make|model|year` linking records), JS-rendered sites (would need Playwright), and robots/ToS checks per new domain.

## Done when
- `probe_articles.py --quiet` shows sections > 0 for all 8 pages.
- `cars.json` has 8 records. Every record has `price_ils` except Kia EV9 (LLM, possibly null), plus a `powertrain` and `body_type`.
- `chunks.jsonl` has about 120–250 chunks, each with `text_en` and an embedding of the configured length.
- Unit tests pass on cached HTML: spec normalization, section count > 1 after splitting, FAQ count for mg-s6 = 10.
