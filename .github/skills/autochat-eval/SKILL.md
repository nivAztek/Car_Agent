---
name: autochat-eval
description: "Evaluate the autochat chatbot with MLflow: golden set (facts, opinions, metadata filters, comparisons, multi-turn preference inference, hallucination traps), deterministic golden scorer, Foundry-backed MLflow LLM judges (relevance, retrieval groundedness, language, citations), run comparison in the local MLflow UI, and example transcripts for the README. Use after changing retrieval, prompts or the graph, and before submission."
---

# autochat-eval

## Scripts (repo root)

| Script | Purpose |
|---|---|
| `scripts/run_eval.py [--api URL] [--only types/ids] [--judge] [--no-mlflow] [--min-pass 0.8]` | Runs `eval/golden.jsonl` turn by turn through `runtime.run_turn` (in-process) or `/chat/invoke` (`--api`). Each turn is an MLflow trace (session = case thread). Then it runs **one MLflow run** in experiment `autochat-eval`: params (deployments, mode), metrics (`pass_rate`, `pass_rate_<type>`, `latency_p50_s`), the `eval_results.json` artifact, and `mlflow.genai.evaluate` on the last-turn trace of each case. Also writes `eval/results/<ts>.json`. Exits 1 below `--min-pass`. |
| `scripts/selftest_scoring.py` | Offline test of `score_case` plus the MLflow `golden` scorer through `genai.evaluate` (temp sqlite store, no LLM or DB). Run after editing `run_eval.py`. |
| `../autochat-agent/scripts/chat_cli.py --script X --out Y.md` | Records a transcript for the README from a scripted conversation. |

## Scorers (MLflow)
| Scorer | Kind | Notes |
|---|---|---|
| `golden` → `check_<name>` + `golden_pass` | code (always) | Wraps `score_case`: retrieval_all/any (cited car ids), contains_any/all, not_contains, abstained, no_sources, profile_*. |
| `relevance_to_query` | LLM judge (`--judge`) | Foundry via `openai:/<MLFLOW_JUDGE_DEPLOYMENT or FOUNDRY_CHAT_DEPLOYMENT>`. Skipped (no assessment) on abstain cases: a correct refusal is "irrelevant" to the question. |
| `retrieval_groundedness` | LLM judge (`--judge`) | Judged against the top-level `plan_evidence` RETRIEVER span (catalog selections + final context). Skipped on abstain cases and turns without retrieval. |
| `same_language` | `Guidelines` judge on the answer text only | Reply in the user's language (he/en); car names, numbers and ₪ don't count. |
| `grounded_or_abstains` | `Guidelines` judge on the answer text only | Every figure has `[n]` (a short verdict restating cited points is fine), or the answer says the car/fact isn't covered. Never invents specs. |

LLM judges run only when every row has a trace (`--api` against a server with tracing off falls back to code scorers).

## Comparing runs
`scripts/dev.ps1 mlflow` → experiment **autochat-eval** → select runs → Compare. Change one thing per run (prompt, `k_per_car`, RRF weights, model) and note it in the run name or description. Click a failing row to open its full trace.

## Golden set (`eval/golden.jsonl`, 26 cases)
Expected values come from the **real spec tables** (probed 2026-10-16):

| Car | price ₪ | 0-100 s | other |
|---|---|---|---|
| citroen-c3-2026 | 114,990 | 9.5 | 20 km/l, ICE |
| audi-rs3-fl | 528,000 | 3.8 | ICE (from text) |
| long-term-report-kia-ev9-4 | — (no table) | — | EV, long-term report |
| mg-s6 | 168,888 | 7.3 | EV, WLTP 530 / real 430, DC 144 |
| hyundai-elantra-n-manual | 224,900 | 5.8 | 10.3 km/l, manual |
| aion-ht | 179,990 | 7.1 | EV, WLTP 445 / real 370, DC 163 |
| link-and-co-01-2026 | 198,900 | 7.7 | PHEV |
| genesis-gv80-2026 | 575,990 | 6.2 | 8.1 km/l |

Case types: `fact` (8, incl. a typo name and a bare Hebrew make), `opinion` (3), `filter` (4, exercise metadata filtering), `compare` (2), `multi_turn` (4, exercise preference inference and reference resolution incl. "the second one"), `abstain` (5, hallucination traps incl. same-make "Audi Q5", and off-topic).

**Baseline (2026-10-21, `--judge`, gpt-5.5 judge):** golden 26/26; `grounded_or_abstains` 1.0, `relevance_to_query` 1.0, `same_language` 1.0, `retrieval_groundedness` 0.95 (one inferred verdict in a Hebrew comparison).

Abstention cases expect phrases such as "not covered", "don't have information", "לא נסקר", or "אין לי מידע". If the prompt wording changes, update `ABSTAIN_PATTERNS` in `run_eval.py` to match.

## Targets for the POC
- fact ≥ 7/8, filter ≥ 3/4, abstain 5/5 (non-negotiable: no hallucinated specs), overall ≥ 80%.
- With `--judge`: `grounded_or_abstains` ≥ 0.9, `retrieval_groundedness` ≥ 0.9 on non-abstain cases, `same_language` = 1.0.

## When a case fails
0. Open the case's trace from the eval run in MLflow. The RETRIEVER span shows which chunks were retrieved; the `generate` span shows the exact prompt.
1. Check the `cited` car ids in the results JSON. A wrong or missing car is a retrieval problem: check `analyze_turn` car resolution, the partition fan-out, and BM25 tokens.
2. Right car but the number is missing: check that the `car_card` chunk was included, and the prompt's citation rules.
3. Answered a trap: strengthen the "only CONTEXT" rule and make `analyze_turn` flag `out_of_corpus_cars`.
4. Don't edit golden expectations to make tests pass unless the source article really says otherwise. Re-check with `probe_articles.py`.

## README example interaction
Record 2 transcripts (EN family-EV flow, HE enthusiast flow) with `chat_cli.py --script eval/transcripts/<name>.txt --out eval/transcripts/<name>.md` (scripts: `family_en.txt`, `sporty_he.txt`) and paste excerpts into README.md.
