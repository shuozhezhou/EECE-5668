"""Run the benchmark for one model under one or more conditions.

Traces are appended to <out-dir>/<model-tag>/<condition>.jsonl as each episode
finishes, so an interrupted run (e.g. a Colab disconnect) resumes where it stopped.

Example (Colab, with a vLLM server on port 8000):
    python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --conditions C0 C1 \
        --out-dir /content/drive/MyDrive/EECE5668/results --workers 16
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.market import DEFAULT_SNAPSHOT, SnapshotMarket
from agent.runner import CONDITIONS, run_episode

CODE_DIR = Path(__file__).resolve().parent


def model_tag(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def load_done(path: Path) -> set[str]:
    """IDs already finished (episodes that ended in an error are retried)."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off by a crash
        if not str(rec.get("status", "")).startswith("error"):
            done.add(rec["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name as served (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--conditions", nargs="+", default=["C0", "C1"], choices=CONDITIONS)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--questions", default=str(CODE_DIR / "questions" / "questions_v1.jsonl"))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--out-dir", default=str(CODE_DIR.parent / "results"))
    parser.add_argument("--workers", type=int, default=8, help="Concurrent episodes")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke tests)")
    parser.add_argument("--ids", nargs="*", default=None, help="Only these question IDs")
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    from openai import OpenAI  # imported here so the rest of the package works without it

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)
    market = SnapshotMarket(args.snapshot_dir)
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[: args.limit]

    out_dir = Path(args.out_dir) / model_tag(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        path = out_dir / f"{condition}.jsonl"
        done = load_done(path)
        todo = [q for q in questions if q["id"] not in done]
        print(f"[{condition}] {len(done)} done, {len(todo)} to run -> {path}", flush=True)
        if not todo:
            continue
        lock = threading.Lock()
        started, n_err = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a", encoding="utf-8") as f:
            futures = {
                pool.submit(run_episode, q, condition, client, args.model, market,
                            args.max_tool_rounds, 0.0, args.max_tokens): q
                for q in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                trace = fut.result()
                n_err += str(trace["status"]).startswith("error")
                with lock:
                    f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  errors={n_err}  {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
