---
applyTo: "src/autochat/ingestion/**,tests/**/test_*ingest*.py,tests/**/test_*parse*.py"
description: "Ingestion pipeline rules: scraping, parsing, extraction, chunking for auto.co.il articles"
---
# Ingestion rules

Load the `autochat-ingestion` skill for selectors, the label map, and scripts. Key points:

- **Fetch** with a browser User-Agent and `follow_redirects=True`. Abort if the final URL leaves `auto.co.il` (a known ad-tracker redirect). Cache to `data/raw/<slug>.html` and never re-download when the cache exists, unless `--refresh` is passed.
- **Parse with `selectolax.lexbor.LexborHTMLParser`.** The legacy `selectolax.parser` raises ImportError in selectolax 1.x.
- Body = `div.sections__items .text-wrapper`, walked in document order. A new section starts at `h2`/`h3` **or** at a short bold-only `<p><strong>`. Some pages (Kia EV9 long-term, Elantra N) have no H2s, so fall back to size-based splitting.
- Drop boilerplate headings: "הצטרפו לוואטסאפ", "כתבות אחרונות", "כתבות באותם נושאים", "מותגים מתחרים", "המומחה האובייקטיבי".
- Spec table = `.table-item__title` / `.table-item__value`. Normalize through `SPEC_LABEL_MAP`. Log unknown labels and never drop them silently.
- FAQ = `div.faq__container details.spollers__item` (`summary` = question, `.spollers__body` = answer). Only some pages have it.
- Publish date = first `<time datetime>`.
- The pipeline must be **idempotent and resumable**. Each stage writes to `data/processed/` and skips work whose output exists, unless `--force` is passed. Chunk ids are deterministic: `f"{car_id}:{section_idx:02d}:{part}"`.
- LLM extraction uses `with_structured_output(PydanticModel)` with temperature 0 on the fast deployment. Pass the known table values in so the LLM doesn't re-guess them. Require an `evidence` quote for each scored aspect.
- Prefix every chunk with a context header: `"{make} {model} {year} | {section heading} | {body_type}, {powertrain}, ₪{price}"`.
- **Source adapters**: parsing goes through `ingestion/sources/` (`AutoCoIlAdapter`, then the `GenericAdapter` trafilatura fallback). A quality gate falls back to generic when there are 0 sections or under 1,500 chars. Never send raw HTML to an LLM. Never hard-code the car list: aliases and catalog come from the `cars` container. New URLs are processed incrementally through `data/processed/manifest.json` (content hash).
- The bare `--` rule for `dev.ps1` applies here too: `dev.ps1 ingest --url https://...`.
