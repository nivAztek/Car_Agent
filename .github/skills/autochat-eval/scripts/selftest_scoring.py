"""Offline self-test for run_eval scoring (no LLM / DB needed).

Also runs the MLflow `golden` scorer through mlflow.genai.evaluate on static outputs, using a throwaway
sqlite store in a temp dir (skip with --no-mlflow).

Usage:
    .venv\\Scripts\\python.exe .github/skills/autochat-eval/scripts/selftest_scoring.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import expectations_for, make_golden_scorer, norm, score_case  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]


def mlflow_golden_selftest() -> None:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    import logging

    import mlflow

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    # ignore_cleanup_errors: MLflow keeps the sqlite file open for the process lifetime (locked on Windows)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        mlflow.set_tracking_uri(f"sqlite:///{Path(tmp, 'selftest.db').as_posix()}")
        mlflow.set_experiment("selftest")
        case = {
            "id": "t",
            "type": "fact",
            "lang": "en",
            "expect": {"car_ids": ["mg-s6"], "must_contain_any": ["168,888"]},
        }
        rows = [
            {
                "inputs": {"message": "price?"},
                "expectations": expectations_for(case),
                "outputs": {"answer": "The MG S6 costs ₪168,888 [1].", "sources": [{"n": 1, "car_id": "mg-s6"}]},
            },
            {
                "inputs": {"message": "price?"},
                "expectations": expectations_for(case),
                "outputs": {"answer": "About 170k.", "sources": [{"n": 1, "car_id": "aion-ht"}]},
            },
        ]
        res = mlflow.genai.evaluate(data=rows, scorers=[make_golden_scorer()])
        got = res.metrics.get("golden_pass/mean")
        assert got is not None and abs(float(got) - 0.5) < 1e-9, res.metrics
        assert "check_retrieval_all/mean" in res.metrics, res.metrics
    print("selftest OK - MLflow golden scorer (golden_pass/mean=0.5 on 1 pass + 1 fail)")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    assert norm("₪168,888") == "₪168888"
    assert norm("168 888") == "168888"

    case = {"id": "t", "type": "fact", "lang": "en", "expect": {"car_ids": ["mg-s6"], "must_contain_any": ["168,888"]}}
    good = {"answer": "The MG S6 costs ₪168,888 [1].", "sources": [{"n": 1, "car_id": "mg-s6"}]}
    bad = {"answer": "It costs about 170k.", "sources": [{"n": 1, "car_id": "aion-ht"}]}
    assert score_case(case, good)["passed"]
    assert not score_case(case, bad)["passed"]

    trap = {"id": "x", "type": "abstain", "lang": "he", "expect": {"abstain": True}}
    assert score_case(trap, {"answer": "הרכב הזה לא נסקר בכתבות שברשותי.", "sources": []})["passed"]
    assert not score_case(trap, {"answer": "Its range is 513 km.", "sources": []})["passed"]

    prof_case = {
        "id": "p",
        "type": "multi_turn",
        "lang": "en",
        "expect": {"profile": {"budget_max_ils_range": [180000, 220000], "powertrain_pref": ["ev"]}},
    }
    assert score_case(
        prof_case, {"answer": "", "sources": [], "profile": {"budget_max_ils": 200000, "powertrain_pref": ["ev"]}}
    )["passed"]

    golden = REPO_ROOT / "eval" / "golden.jsonl"
    cases = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate golden ids"
    for c in cases:
        assert {"id", "type", "lang", "turns", "expect"} <= c.keys(), c["id"]
    print(f"selftest OK - {len(cases)} golden cases valid")
    if "--no-mlflow" not in sys.argv:
        mlflow_golden_selftest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
