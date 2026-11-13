#!/usr/bin/env bash
# Full pipeline. Each stage was executed individually in the original run; this chained script itself
# has NOT been executed end-to-end [TEST REQUIRED]. Expected total time on 1 vCPU: ~12-15 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
python scripts/00_env_check.py          | tee results/logs/00_env_check.log
python scripts/01_validate_baseline.py  | tee results/logs/01_validate_baseline.log
python scripts/02_run_experiments.py    | tee results/logs/02_run_experiments.log   # ~9 min (LR fits dominate)
python scripts/03_analyze.py            | tee results/logs/03_analyze.log
python scripts/04_efficiency.py         | tee results/logs/04_efficiency.log
python scripts/05_make_latex.py
(cd paper && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex)
