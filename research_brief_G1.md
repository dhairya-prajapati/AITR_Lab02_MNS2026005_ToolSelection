# G1 One-page research brief

| Field | Content |
|---|---|
| Review topic and key papers | Tool selection in multi-tool LLM agents. Verified-by-review references used: MetaTool (Huang et al. 2024), ToolLLM (Qin 2023), AnyTool (Du 2024), Gorilla (Patil 2023), RestGPT (Song 2023), survey (Qu 2025). **Student to confirm each against the primary source** (only MetaTool's data/licence were checked in this project). |
| Research gap | Review §4/§6/§7 *argue* that (i) library scale degrades selection, (ii) fine-tuned selectors fail on unseen tools, (iii) accuracy hides cost. Source evidence: review Sections 3, 4, 6, 7. Not measured in the review → INFERRED claims. |
| Research question | **RQ1** accuracy vs library size (10→199); **RQ2** accuracy on tools with no training queries and whether exemplars help; **RQ3** whether a cost-adjusted score (EAS) changes method ranking, and how effective confidence abstention is. |
| Study pathway | B (controlled comparison) + C (one isolated change: exemplar-augmented tool documents). |
| Dataset / measurement | MetaTool ToolE single-tool queries, 199 tools, MIT licence, commit 35e81bb. Per-tool 60/20/20 split after text de-dup; 5 seeds; unseen-tool and out-of-library protocols in `configs/protocol.json`. |
| Baseline / comparators | BM25 (validated vs `rank_bm25`), TF-IDF-desc, Hybrid(k=5), TF-IDF+LR, Random. All code in `src/methods.py`; no external weights. |
| Primary metric and control | Top-1 accuracy (closed set); EAS_c (open world). Held constant: cleaned data, splits per seed, tokeniser, tie-breaking. |
| Computing and risks | 1 vCPU, 3 GB RAM; full run ≈12 min. Risks: template-like LLM-generated queries; near-duplicate tools; no LLM evaluated (stated limitation). Fallback if dataset unreachable: none needed (files vendored in `data/`). |
| First executable milestone | `scripts/00_env_check.py` (ENV OK) and `scripts/01_validate_baseline.py` (BASELINE VALIDATION PASSED) — both executed; logs in `results/logs/`. |
