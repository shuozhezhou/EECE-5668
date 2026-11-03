"""Build colab/run_experiments.ipynb from the cell sources below.

Keeping the notebook as code makes it easy to review and diff. Rebuild with:
    py code/tools_dev/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CELLS: list[tuple[str, str]] = [
    ("markdown", """# EECE 5668: run the benchmark on Colab

This notebook does five things:
1. Mounts Google Drive.
2. Pulls the latest code from GitHub.
3. Restores the market snapshot from Drive.
4. Serves an open-weight model with vLLM.
5. Runs the agent conditions.

Results go to `MyDrive/EECE5668/results/<model>/<condition>.jsonl`. They are written
after every episode, so a disconnect loses nothing: re-run the cells and the run resumes.

**Runtime:** use a GPU runtime (Runtime → Change runtime type). An L4 (24 GB) fits
7B–8B models in bf16; use an A100 for 14B, or for 32B with AWQ quantization."""),
    ("code", """# 1. Drive and paths
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, pathlib
DRIVE = pathlib.Path('/content/drive/MyDrive/EECE5668')
REPO = pathlib.Path('/content/EECE-5668')
SNAPSHOT = '2026-11-02'
RESULTS = DRIVE / 'results'
for p in [DRIVE / 'data' / 'snapshots', RESULTS, DRIVE / 'logs']:
    p.mkdir(parents=True, exist_ok=True)
print(sorted(os.listdir(DRIVE)))"""),
    ("code", """# 2. Latest code from GitHub
import shutil
os.chdir('/content')  # never stand inside the directory we may delete
if REPO.exists() and not (REPO / '.git').exists():
    shutil.rmtree(REPO)  # left over from running cell 3 first; clone fresh
if (REPO / '.git').exists():
    !git -C {REPO} pull --ff-only
else:
    !git clone https://github.com/shuozhezhou/EECE-5668.git {REPO}
!git -C {REPO} log --oneline -3"""),
    ("code", """# 3. Market snapshot: canonical copy on Drive, linked into the repo
import shutil, zipfile
snap_drive = DRIVE / 'data' / 'snapshots' / SNAPSHOT
if not (snap_drive / 'prices.parquet').exists():
    # First run: unpack the bundle uploaded from the local machine
    with zipfile.ZipFile(DRIVE / 'upload_to_drive_code.zip') as z:
        prefix = f'code/data/snapshots/{SNAPSHOT}/'
        for name in z.namelist():
            if name.replace('\\\\', '/').startswith(prefix) and not name.endswith('/'):
                target = snap_drive / pathlib.Path(name.replace('\\\\', '/')).name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
snap_repo = REPO / 'code' / 'data' / 'snapshots' / SNAPSHOT
snap_repo.parent.mkdir(parents=True, exist_ok=True)
if not snap_repo.exists():
    shutil.copytree(snap_drive, snap_repo)  # local copy: faster than reading through the Drive mount
print(sorted(os.listdir(snap_repo)))"""),
    ("code", """# 4. Dependencies (vLLM takes ~3 min on a fresh runtime)
!pip install -q vllm openai tabulate 2>&1 | tail -2
%cd {REPO}/code
!python tests/check_questions.py | tail -1
!python tests/test_pipeline.py | tail -1"""),
    ("code", """# 5. Start the vLLM server in the background
# Tool-call parser: 'hermes' for Qwen2.5/Qwen3, 'llama3_json' for Llama 3.x.
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
TOOL_PARSER = 'hermes'
MAX_LEN = 16384

import time, requests
!pkill -f "vllm serve" || true
log_path = DRIVE / 'logs' / f"vllm_{MODEL.split('/')[-1]}.log"
server = subprocess.Popen(
    ['vllm', 'serve', MODEL, '--enable-auto-tool-choice', '--tool-call-parser', TOOL_PARSER,
     '--max-model-len', str(MAX_LEN), '--gpu-memory-utilization', '0.90', '--port', '8000'],
    stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
    start_new_session=True)  # otherwise interrupting any later cell also kills the server
for i in range(180):
    try:
        if requests.get('http://localhost:8000/v1/models', timeout=2).ok:
            print(f'server ready after {i * 5}s'); break
    except requests.RequestException:
        pass
    if server.poll() is not None:
        print('server exited; see log:'); print(open(log_path).read()[-3000:]); break
    time.sleep(5)"""),
    ("code", """# 6. Smoke test: 5 questions under C1, then show one trace
assert requests.get('http://localhost:8000/v1/models', timeout=5).ok, 'vLLM server is not up; re-run cell 5'
!python run_experiment.py --model {MODEL} --conditions C1 --limit 5 --out-dir /content/smoke --workers 5
import json
tag = MODEL.split('/')[-1]
trace = json.loads(open(f'/content/smoke/{tag}/C1.jsonl').readline())
print(trace['status'], '| tool calls:', trace['n_tool_calls'])
for m in trace['messages'][1:]:
    print(f"--- {m['role']}:", (m.get('content') or '')[:400], m.get('tool_calls', ''))"""),
    ("code", """# 7. Full run (resumable). 200 questions per condition.
assert requests.get('http://localhost:8000/v1/models', timeout=5).ok, 'vLLM server is not up; re-run cell 5'
!python run_experiment.py --model {MODEL} --conditions C0 C0F C1 C2 --out-dir {RESULTS} --workers 16"""),
    ("code", """# 8. Score everything under results/ (all models run so far)
!python -m eval.summarize --results {RESULTS}"""),
    ("code", """# 9. Stop the server before switching models (or disconnect the runtime to save compute units)
server.terminate()"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "L4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = REPO / "colab" / "run_experiments.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
