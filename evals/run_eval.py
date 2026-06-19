"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    q_text = question["question"]
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Run gold SQL to get expected rows
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)
    if not gold_ok:
        return {
            "question": q_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "gold_error": gold_err,
            "agent_error": None,
            "iterations": 0,
            "correct_at_iteration": {},
            "final_correct": False,
            "skipped": True,
        }

    # Call the agent
    try:
        response = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id, "tags": {"phase": "5", "eval": "baseline"}},
            timeout=120.0,
        )
        response.raise_for_status()
        agent_result = response.json()
    except Exception as e:
        return {
            "question": q_text,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "gold_error": None,
            "agent_error": str(e),
            "iterations": 0,
            "correct_at_iteration": {},
            "final_correct": False,
            "skipped": False,
        }

    history = agent_result.get("history", [])
    total_iterations = agent_result.get("iterations", 0)

    # Check correctness at each iteration by replaying history
    correct_at_iteration = {}
    iteration = 0

    for entry in history:
        if entry["node"] in ("generate_sql", "revise"):
            iteration += 1
            last_sql = entry["sql"]
            pred_ok, pred_rows, _ = run_sql(db_id, last_sql)
            correct_at_iteration[iteration] = matches(gold_rows, pred_rows) if pred_ok else False

    final_correct = correct_at_iteration.get(total_iterations, False)

    return {
        "question": q_text,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_sql": agent_result.get("sql", ""),
        "agent_error": None,
        "iterations": total_iterations,
        "correct_at_iteration": correct_at_iteration,
        "final_correct": final_correct,
        "skipped": False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results with per-iteration pass rates."""
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    evaluated = total - skipped

    max_iter = max(
        (r["iterations"] for r in results if not r.get("skipped")),
        default=3,
    )
    max_iter = max(max_iter, 3)

    iter_correct = {i: 0 for i in range(1, max_iter + 1)}

    for r in results:
        if r.get("skipped"):
            continue
        correct_at = r.get("correct_at_iteration", {})
        final_iter = r["iterations"]
        final_result = r["final_correct"]

        for i in range(1, max_iter + 1):
            if i in correct_at:
                iter_correct[i] += 1 if correct_at[i] else 0
            elif i > final_iter:
                iter_correct[i] += 1 if final_result else 0

    iter_pass_rate = {
        f"iter_{i}": round(iter_correct[i] / evaluated, 3) if evaluated > 0 else 0.0
        for i in range(1, max_iter + 1)
    }

    final_correct = sum(1 for r in results if r.get("final_correct"))

    return {
        "total": total,
        "skipped": skipped,
        "evaluated": evaluated,
        "final_correct": final_correct,
        "final_pass_rate": round(final_correct / evaluated, 3) if evaluated > 0 else 0.0,
        "per_iteration_pass_rate": iter_pass_rate,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()