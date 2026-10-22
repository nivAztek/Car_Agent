"""Run the autochat golden set, score it, and log everything to MLflow (local sqlite store).

Usage (repo root):
    .venv\\Scripts\\python.exe .github/skills/autochat-eval/scripts/run_eval.py                       # in-process graph
    .venv\\Scripts\\python.exe .github/skills/autochat-eval/scripts/run_eval.py --api http://localhost:8000
    .venv\\Scripts\\python.exe .github/skills/autochat-eval/scripts/run_eval.py --only fact,abstain --judge
    ... --no-mlflow                                                     # plain console + JSON only

Golden format (eval/golden.jsonl): {id, type, lang, turns[], expect{...}}
  expect keys: car_ids (all must be cited), car_ids_any (>=1 cited), must_contain_any, must_contain_all,
               must_not_contain, abstain (bool), no_sources (bool), profile{budget_max_ils_range, powertrain_pref}

MLflow (default on): every turn is traced (session = thread_id); after the run, `mlflow.genai.evaluate`
scores the last-turn trace of each case with the deterministic `golden` scorer (one Feedback per check +
`golden_pass`) and, with --judge, Foundry-backed judges (RelevanceToQuery, RetrievalGroundedness,
language + citation Guidelines). View: scripts/dev.ps1 mlflow -> experiment "autochat-eval".
Also writes eval/results/<timestamp>.json. Exit code 1 if pass rate < --min-pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

ABSTAIN_PATTERNS = [
    r"not (covered|included|in (the|our) (reviews|articles|data|catalog))",
    r"don'?t have (information|data)",
    r"no (information|data|review)",
    r"only (cover|have|discuss)",
    r"outside (the|my|what i) (scope|reviews|cover)",
    r"(don'?t|do not) have (a |any )?(road[- ]test )?(review|information|data|content)",
    r"(i'?m|i am) (here|only able) to help with car",
    r"(do|does) not (cover|include|mention|predict|forecast|say|state|give|provide)",
    r"(can'?t|cannot|unable to) (help|answer|predict|provide|create)",
    r"(isn'?t|aren'?t|not) covered",
    r"לא (נסקר|מכוסה|מופיע|כלול|נכלל)",
    r"אין (לי )?(מידע|נתונים|סקירה|מבחן|ביקורת|ביקורות)",
    r"אינו (נסקר|מופיע)",
    r"רק על",
    r"(אינן|לא) מכסות",
]


def norm(text: str) -> str:
    """Lowercase and drop thousands separators so '168,888' == '168888' == '168 888'."""
    return re.sub(r"(?<=\d)[,\s](?=\d{3})", "", text.lower())


def cited_car_ids(sources: list[dict]) -> set[str]:
    return {s.get("car_id") for s in sources if isinstance(s, dict) and s.get("car_id")}


def score_case(case: dict, out: dict) -> dict:
    exp, answer = case["expect"], out.get("answer", "")
    a = norm(answer)
    cars = cited_car_ids(out.get("sources", []))
    checks: dict[str, bool] = {}
    if "car_ids" in exp:
        checks["retrieval_all"] = set(exp["car_ids"]) <= cars
    if "car_ids_any" in exp:
        checks["retrieval_any"] = bool(set(exp["car_ids_any"]) & cars)
    if "must_contain_any" in exp:
        checks["contains_any"] = any(norm(s) in a for s in exp["must_contain_any"])
    if "must_contain_all" in exp:
        checks["contains_all"] = all(norm(s) in a for s in exp["must_contain_all"])
    if "must_not_contain" in exp:
        checks["not_contains"] = not any(norm(s) in a for s in exp["must_not_contain"])
    if exp.get("abstain"):
        plain = re.sub(r"[*_`]", "", answer).replace("\u2019", "'")  # **not** / isn’t
        checks["abstained"] = any(re.search(p, plain, re.I) for p in ABSTAIN_PATTERNS)
    if exp.get("no_sources"):
        checks["no_sources"] = not out.get("sources")
    if "profile" in exp:
        prof = out.get("profile") or {}
        rng = exp["profile"].get("budget_max_ils_range")
        if rng:
            b = prof.get("budget_max_ils")
            checks["profile_budget"] = b is not None and rng[0] <= b <= rng[1]
        for pt in exp["profile"].get("powertrain_pref", []):
            checks[f"profile_pt_{pt}"] = pt in (prof.get("powertrain_pref") or [])
    return {
        "id": case["id"],
        "type": case["type"],
        "lang": case["lang"],
        "checks": checks,
        "passed": all(checks.values()) if checks else True,
        "cited": sorted(cars),
        "answer": answer,
    }


def make_golden_scorer():
    """MLflow code scorer wrapping score_case: one Feedback per check + overall `golden_pass`."""
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer(name="golden")
    def golden(outputs, expectations) -> list[Feedback]:
        out = outputs if isinstance(outputs, dict) else {"answer": str(outputs or "")}
        case = {
            "id": expectations.get("case_id", "?"),
            "type": expectations.get("case_type", "?"),
            "lang": expectations.get("lang", "?"),
            "expect": expectations.get("expect", {}),
        }
        res = score_case(case, out)
        failed = [k for k, v in res["checks"].items() if not v]
        fbs = [Feedback(name=f"check_{k}", value=bool(v)) for k, v in res["checks"].items()]
        fbs.append(
            Feedback(
                name="golden_pass",
                value=res["passed"],
                rationale="failed: " + ", ".join(failed) if failed else "all checks passed",
            )
        )
        return fbs

    return golden


def build_scorers(judge: bool) -> list:
    scorers = [make_golden_scorer()]
    if judge:
        from mlflow.genai.scorers import Guidelines, RelevanceToQuery, RetrievalGroundedness, scorer

        from autochat.observability import judge_model_uri  # type: ignore[import-not-found]

        model = judge_model_uri()

        def answering_cases_only(judge_fn, name: str, *, needs_retrieval: bool = False):
            """Relevance/groundedness are meaningless when the correct reply is a refusal ("Tesla isn't
            covered" is judged irrelevant to "Tesla range?"). Returning None logs no assessment, so
            abstain cases are excluded from the mean instead of counted as failures or errors."""

            @scorer(name=name)
            def _scoped(trace, expectations):
                if expectations.get("case_type") == "abstain" or expectations.get("expect", {}).get("abstain"):
                    return None
                if needs_retrieval and not trace.search_spans(span_type="RETRIEVER"):
                    return None
                return judge_fn(trace=trace)

            return _scoped

        def on_answer_text(judge_fn, name: str):
            """Guidelines judges see the whole root-span output (sources JSON with English headings,
            planner steps), which made Hebrew answers fail `same_language`. Judge the answer text only."""

            @scorer(name=name)
            def _answer_only(inputs, outputs):
                answer = outputs.get("answer", "") if isinstance(outputs, dict) else str(outputs or "")
                message = inputs.get("message", "") if isinstance(inputs, dict) else str(inputs or "")
                return judge_fn(inputs={"message": message}, outputs=answer)

            return _answer_only

        scorers += [
            answering_cases_only(RelevanceToQuery(model=model), "relevance_to_query"),
            answering_cases_only(RetrievalGroundedness(model=model), "retrieval_groundedness", needs_retrieval=True),
            on_answer_text(
                Guidelines(
                    name="same_language",
                    model=model,
                    guidelines="The prose of the response is written in the same natural language as the "
                    "user's last message (Hebrew or English). Car names, model codes, numbers, units and "
                    "currency symbols such as ₪ do not count. Whether the response fulfils or declines the "
                    "request is irrelevant to this guideline; judge only the language.",
                ),
                "same_language",
            ),
            on_answer_text(
                Guidelines(
                    name="grounded_or_abstains",
                    model=model,
                    guidelines="Either (a) every specific figure about a reviewed car (price, range, acceleration, "
                    "power, size, consumption, charging) carries a [n] citation, and each opinion attributed to "
                    "the review is cited where it is stated in detail; a short opening verdict or summary that "
                    "restates points cited later in the same response needs no citation of its own; or (b) the "
                    "response declines, saying the car or information is not covered by the reviews; a decline "
                    "needs no citations. The response never states specs for a car it says is not covered.",
                ),
                "grounded_or_abstains",
            ),
        ]
    return scorers


def expectations_for(case: dict) -> dict:
    return {"case_id": case["id"], "case_type": case["type"], "lang": case["lang"], "expect": case["expect"]}


DATASET_NAME = "autochat-golden"


def sync_golden_dataset(golden_path: str):
    """Create/update the MLflow evaluation dataset `autochat-golden` from golden.jsonl (idempotent: records
    are de-duplicated by inputs). Each record: inputs {case_id, message (last turn), turns}, expectations.
    Visible in the MLflow UI under the experiment's Datasets tab. Returns the dataset or None on failure."""
    import mlflow
    from mlflow.genai.datasets import create_dataset, get_dataset, search_datasets

    cases = [json.loads(line) for line in Path(golden_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [
        {
            "inputs": {"case_id": c["id"], "message": c["turns"][-1], "turns": c["turns"]},
            "expectations": expectations_for(c),
        }
        for c in cases
    ]
    try:
        exp = mlflow.get_experiment_by_name("autochat-eval")
        found = search_datasets(
            experiment_ids=[exp.experiment_id] if exp else None, filter_string=f"name = '{DATASET_NAME}'"
        )
        if found:
            dataset = get_dataset(dataset_id=found[0].dataset_id)
        else:
            dataset = create_dataset(
                name=DATASET_NAME,
                experiment_id=exp.experiment_id if exp else None,
                tags={"source": "eval/golden.jsonl", "purpose": "autochat golden set"},
            )
        dataset.merge_records(records)
        print(f"MLflow dataset '{DATASET_NAME}': {len(records)} cases synced")
        return dataset
    except Exception as exc:  # noqa: BLE001 - the dataset is an extra; never fail the eval over it
        print(f"MLflow dataset sync skipped: {type(exc).__name__}: {str(exc)[:200]}")
        return None


async def run_case_in_process(graph, case: dict) -> dict:
    """Uses the traced runtime wrapper (one MLflow trace per turn, session = thread_id)."""
    from autochat.agent.runtime import run_turn  # type: ignore[import-not-found]

    thread = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    out: dict = {}
    for turn in case["turns"]:
        out = await run_turn(graph, thread_id=thread, message=turn, user="eval")
    return out


async def run_case_api(api: str, case: dict) -> dict:
    import httpx

    thread = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    out: dict = {}
    async with httpx.AsyncClient(timeout=240) as client:
        for turn in case["turns"]:
            r = await client.post(f"{api}/chat/invoke", json={"thread_id": thread, "message": turn})
            r.raise_for_status()
            out = r.json()
    return out


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--api")
    ap.add_argument("--golden", default=str(REPO_ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--only", help="comma-separated case types or ids")
    ap.add_argument("--judge", action="store_true", help="add Foundry-backed MLflow LLM judges")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracing/evaluation")
    ap.add_argument("--min-pass", type=float, default=0.8)
    args = ap.parse_args()
    if args.judge and args.no_mlflow:
        ap.error("--judge requires MLflow")
    return await run_cases(args)


async def run_cases(args: argparse.Namespace) -> int:
    use_mlflow = False
    if not args.no_mlflow:
        from autochat.observability import setup_mlflow  # type: ignore[import-not-found]

        use_mlflow = setup_mlflow("autochat-eval")
        if not use_mlflow:
            print("AUTOCHAT_TRACING=false -> running without MLflow")

    cases = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        sel = set(args.only.split(","))
        cases = [c for c in cases if c["type"] in sel or c["id"] in sel]

    results = []
    outputs: dict[str, dict] = {}

    async def evaluate(runner) -> None:
        for case in cases:
            t = time.perf_counter()
            try:
                out = await runner(case)
                outputs[case["id"]] = out
                res = score_case(case, out)
                res["trace_id"] = out.get("trace_id")
            except Exception as exc:  # noqa: BLE001
                res = {
                    "id": case["id"],
                    "type": case["type"],
                    "lang": case["lang"],
                    "checks": {"error": False},
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            res["latency_s"] = round(time.perf_counter() - t, 2)
            results.append(res)
            failed = [k for k, v in res["checks"].items() if not v]
            print(f"{'PASS' if res['passed'] else 'FAIL'}  {res['id']:<26} {res['latency_s']:>6}s  {failed or ''}")

    if args.api:
        await evaluate(lambda c: run_case_api(args.api.rstrip("/"), c))
    else:
        from autochat.agent.runtime import get_runtime  # type: ignore[import-not-found]

        async with get_runtime() as (graph, _deps):
            await evaluate(lambda c: run_case_in_process(graph, c))

    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["passed"])
    print("\ntype          pass/total")
    for t, v in sorted(by_type.items()):
        print(f"{t:<14}{sum(v):>3}/{len(v)}")
    rate = sum(r["passed"] for r in results) / max(len(results), 1)
    print(f"OVERALL       {rate:.0%}")

    out_dir = REPO_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {path.relative_to(REPO_ROOT)}")

    # Scoring runs after the event loop closes (see __main__): MLflow judges run in worker threads and
    # must not be started from inside a running loop.
    _PENDING.update(args=args, cases=cases, outputs=outputs, results=results, rate=rate, stamp=stamp)
    _PENDING["use_mlflow"] = use_mlflow
    return 0 if rate >= args.min_pass else 1


_PENDING: dict = {}


def log_to_mlflow(
    args, cases: list[dict], outputs: dict[str, dict], results: list[dict], rate: float, stamp: str
) -> None:
    """Evaluate captured traces with MLflow scorers inside one run (params + metrics + artifacts)."""
    import os
    import time

    import mlflow

    def fetch_trace(trace_id: str | None) -> object | None:
        """In API mode the server exports traces asynchronously; `flush=True` only flushes *this* process."""
        if not trace_id:
            return None
        for _ in range(20 if args.api else 1):
            try:
                trace = mlflow.get_trace(trace_id, flush=True)
            except Exception:  # noqa: BLE001 - not exported yet
                trace = None
            if trace is not None and trace.info.state.value == "OK" and trace.data.spans:
                return trace
            time.sleep(0.5)
        return None

    rows = []
    for case in cases:
        out = outputs.get(case["id"])
        if out is None:
            continue  # runner crashed; already counted as failed in results
        trace = fetch_trace(out.get("trace_id"))
        row = {"expectations": expectations_for(case)}
        if trace is not None:
            row["trace"] = trace
        else:  # e.g. API server running with tracing off -> score on inputs/outputs only
            row["inputs"] = {"message": case["turns"][-1]}
            row["outputs"] = {k: out.get(k) for k in ("answer", "sources", "profile", "intent")}
        rows.append(row)
    if not rows:
        print("MLflow: nothing to evaluate")
        return
    has_traces = all("trace" in r for r in rows)
    if args.judge and has_traces:
        # Defaults (10 rows x 10 scorers in parallel) hit Foundry rate limits and sqlite write locks:
        # every judge then times out after 300 s. Bounded concurrency finishes in a few minutes.
        os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "4")
        os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")
    scorers = build_scorers(args.judge and has_traces)
    if args.judge and not has_traces:
        print("MLflow: some cases have no trace -> LLM judges skipped (retrieval judges need RETRIEVER spans)")
    dataset = sync_golden_dataset(args.golden)
    with mlflow.start_run(run_name=f"eval-{stamp}"):
        if dataset is not None:
            try:
                mlflow.log_input(dataset, context="eval")
            except Exception as exc:  # noqa: BLE001
                print(f"MLflow: could not link dataset to run: {type(exc).__name__}: {str(exc)[:150]}")
        mlflow.log_params(
            {
                "golden": Path(args.golden).name,
                "cases": len(cases),
                "only": args.only or "all",
                "mode": "api" if args.api else "in_process",
                "judge": args.judge,
                "chat_deployment": os.environ.get("FOUNDRY_CHAT_DEPLOYMENT", ""),
                "fast_deployment": os.environ.get("FOUNDRY_FAST_DEPLOYMENT", ""),
            }
        )
        mlflow.log_metric("pass_rate", rate)
        for t in sorted({r["type"] for r in results}):
            v = [r["passed"] for r in results if r["type"] == t]
            mlflow.log_metric(f"pass_rate_{t}", sum(v) / len(v))
        lat = [r["latency_s"] for r in results if "latency_s" in r]
        if lat:
            mlflow.log_metric("latency_p50_s", sorted(lat)[len(lat) // 2])
        mlflow.log_dict({"results": results}, "eval_results.json")
        res = mlflow.genai.evaluate(data=rows, scorers=scorers)
    keys = sorted(k for k in res.metrics if k.endswith("/mean") and not k.startswith("check_"))
    print("MLflow:", {k: round(float(res.metrics[k]), 3) for k in keys})
    print(f"MLflow run {res.run_id} -> scripts/dev.ps1 mlflow (experiment autochat-eval)")


if __name__ == "__main__":
    code = asyncio.run(main())
    if _PENDING.get("use_mlflow"):
        p = _PENDING
        log_to_mlflow(p["args"], p["cases"], p["outputs"], p["results"], p["rate"], p["stamp"])
    raise SystemExit(code)
