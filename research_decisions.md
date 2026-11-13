# Research Decisions & Provenance Log (`research_decisions.md`)

**Student Name:** Dhairya Ashvinbhai Prajapati

**Enrollment / Roll Number:** MNS2026005

**Programme / Department:** Networks Systems and Computing, India

**Course:** AI Tools for Research (Post-Mid-Sem Implementation Study)

**Topic ID & Title:** T12 — Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems

**Submission Folder:** `AITR_Lab02_MNS2026005_ToolSelection`

## 1. One-Page Research Brief (Gate G1 Deliverable)

| Field | Completed Project Information | 
| ----- | ----- | 
| **Review Topic & Key Verified Papers** | **Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems (Topic T12).** Verified original references from the completed review paper (`MNS2026005_DHAIRYA PRAJAPATI.pdf`): | 

1. Huang et al. (2024), *MetaTool Benchmark (`ToolE`)* (`10.48550/arxiv.2310.03128`)

2. Li et al. (2023), *API-Bank* (`10.48550/arxiv.2304.08244`)

3. Patil et al. (2023), *Gorilla* (`10.48550/arxiv.2305.15334`)

4. Qin et al. (2023), *ToolLLM* (`10.48550/arxiv.2307.16789`)

5. Qu et al. (2025), *Tool Learning with Large Language Models: A Survey* (`10.1007/s11704-024-40678-2`) |
   | **Research Gap** | In Sections 5, 6, and 7 of our literature review, we identified two coupled evaluation gaps: (1) standard tool-selection benchmarks evaluate agents using unweighted top-1 binary accuracy (Equation 1), implicitly assuming that every user prompt requires an external API call; and (2) agents frequently fail upstream *tool-use awareness*—invoking external tools when internal pre-trained knowledge suffices, or hallucinating calls that incur billing, latency, or safety costs (Huang et al., 2024 \[3\]; Patil et al., 2023 \[6\]). While our review formulated the Efficiency-Adjusted Selection Score ($\text{EAS}$, Equation 2) theoretically, it remained empirically untested alongside an upstream awareness gate. |
   | **Research Questions & Hypotheses** | **RQ1 (Upstream Awareness Gating):** How does prepending an explicit Stage 0 tool-use awareness gate before top-$k$ retrieval affect the False-Positive Invocation Rate ($\text{FPIR}$), False-Negative Abstention Rate ($\text{FNAR}$), and overall binary selection accuracy on a mixed query workload?

**RQ2 (Cost-Sensitive Evaluation via EAS):** How do system rankings and performance margins shift when evaluated under our proposed Efficiency-Adjusted Selection Score ($\text{EAS}$, Equation 2) compared to unweighted Binary Accuracy (Equation 1)?

**RQ3 (Candidate Pool & Penalty Sensitivity):** How sensitive are 2-stage awareness-gated pipelines to retriever pool size ($k \in \{1, 3, 5, 10\}$), gating threshold ($\tau \in [0.25, 0.48]$), and false-positive invocation cost ($c_{\text{fp}} \in [0.0, 1.5]$)? |
| **Study Pathway** | **Pathway C (Targeted Modification) + Pathway B (Controlled Comparative Study):** Reproduce 1-stage forced retrieval baselines (`BM25`, Dense `all-MiniLM-L6-v2`, and Hybrid Reciprocal Rank Fusion) over the 199-tool MetaTool catalog, then introduce an isolated Stage 0 Awareness Gate and evaluate across controlled penalty schedules. |
| **Dataset & Measurement Setup** | Official **MetaTool (`ToolE`)** repository (`HowieHwong/MetaTool`, `master` branch, MIT License): `dataset/plugin_des.json` ($199$ verified tool descriptions) and `dataset/data/all_clean_data.csv` ($20,544$ clean single-tool queries). Frozen evaluation partition of $N=300$ instances (`seed=42`): $200$ positive tool-requiring queries spanning $107$ unique ground-truth tools + $100$ deterministic no-tool CS/ML conceptual control queries (`ground_truth_tool = "NONE"`). |
| **Baselines & Comparators** | 1. `BM25_Forced_1Stage`: 1-stage lexical retrieval forcing top-1 tool selection.

2. `Dense_Forced_1Stage`: 1-stage dense retrieval (`all-MiniLM-L6-v2`) forcing top-1 tool selection.

3. `Hybrid_Forced_1Stage`: 1-stage Reciprocal Rank Fusion (BM25 + Dense) without awareness gating.

4. `BM25_Awareness_2Stage`, `Dense_Awareness_2Stage`, and `Hybrid_Awareness_2Stage` (Proposed): 2-stage pipelines with Stage 0 awareness gating ($\tau = 0.36$). |
   | **Primary Metrics & Controls** | Standard Binary Accuracy (Equation 1), Efficiency-Adjusted Selection Score ($\text{EAS}$, Equation 2 with frozen penalties $c_{\text{fp}}=0.50, c_{\text{fn}}=0.75, c_{\text{wrong}}=1.00$), Awareness Accuracy, False-Positive Invocation Rate ($\text{FPIR}$), False-Negative Abstention Rate ($\text{FNAR}$), and Selection Accuracy@$k$ ($k \in \{1, 3, 5, 10\}$). |
   | **Computing & Risks** | Executed deterministically on Linux (`Linux-6.6.159+-x86_64-with-glibc2.39`, Python `3.13.18`, `torch 2.11.0+cpu`, `sentence-transformers 5.7.0`, `rank-bm25`, `scikit-learn 1.6.1`). Full $N=300$ evaluation and ablation suite completes in $<20$ seconds on CPU with zero paid API key dependencies. |
   | **First Executable Milestone** | Completed and verified `scripts/env_sanity_check.py` (`results/env_verification_log.json`) and `src/data_loader.py` (`results/module_9_1_data_summary.json`). |

## 2. Chronological Decision & Evidence Ledger (Prompts 01–18)

| Decision ID | Phase / Prompt | Decision Made & Scientific Rationale | Evidence Label | Verified Source / Output File | 
| ----- | ----- | ----- | ----- | ----- | 
| **D01** | Phase I (Prompt 01–02) | Connect implementation study directly to Section 5 (Equations 1 & 2) and Section 7 (Joint Awareness & Selection + Risk/Cost-Aware Evaluation) of the completed literature review. | `[VERIFIED]` | `MNS2026005_DHAIRYA PRAJAPATI.pdf` (pp. 5–7) | 
| **D02** | Phase I (Prompt 03–05) | Select **MetaTool (`ToolE`)** (`HowieHwong/MetaTool`) as the primary executable benchmark rather than multi-container WebArena (which requires heavy Docker server hosting), ensuring 100% deterministic local/Colab reproducibility. | `[VERIFIED]` | `configs/experiment_config.yaml` | 
| **D03** | Phase II (Prompt 06–07) | Freeze evaluation split at $N=300$ ($200$ tool-required + $100$ no-tool control) with `seed=42` and lock default $\text{EAS}$ penalty weights ($c_{\text{correct}}=0.0, c_{\text{fp}}=0.50, c_{\text{fn}}=0.75, c_{\text{wrong}}=1.00$) prior to viewing test results. | `[VERIFIED]` | `src/data_loader.py`, `configs/experiment_config.yaml` | 
| **D04** | Phase III (Prompt 08) | Configure Linux Python `3.13.18` environment (`torch 2.11.0+cpu`, `sentence_transformers 5.7.0`) and execute 5-tool sanity check (`WeatherAPI` top-1 cosine score `0.5182`, toy Eq. 1 = `0.3333`, toy Eq. 2 = `-0.3333`). | `[VERIFIED]` | `results/env_verification_log.json` | 
| **D05** | Phase III (Prompt 09.1 & 11) | Resolve `HTTPError 404` on MetaTool GitHub download by correcting branch from `main` to `master` and path to `dataset/plugin_des.json`. Resolve interactive Jupyter/Colab `NameError: name '__file__' is not defined` by adding `os.getcwd()` fallback. | `[VERIFIED]` | `src/data_loader.py`, `results/module_9_1_data_summary.json` | 
| **D06** | Phase III (Prompt 09.2–10) | Validate 1-stage `Dense_Forced_1Stage` baseline: achieves `0.6350` top-1 accuracy on the 200 positive queries, but `1.0000` FPIR on the 100 no-tool queries, reducing overall Binary Accuracy to `0.4233` and $\text{EAS}$ to `-0.1550`. | `[VERIFIED]` | `results/baseline_metrics.json`, `baseline_validation.md` | 
| **D07** | Phase IV (Prompt 12–14) | Evaluate 2-stage Awareness-Gated pipelines (`BM25`, `Dense`, and `Hybrid RRF`) plus deep ablations across $k \in \{1, 3, 5, 10\}$, $\tau \in [0.25, 0.48]$, and $c_{\text{fp}} \in [0.0, 1.5]$. Confirm `Hybrid_Awareness_2Stage` reaches `0.7533` Top-1 Accuracy (`0.8833` Top-5) and `+0.5333` Top-1 $\text{EAS}$ (`+0.7933` Top-5). | `[VERIFIED]` | `results/results.csv`, `results/ablation_topk_and_tau.csv` | 
| **D08** | Phase V–VI (Prompt 15–18) | Compile final manuscript (`paper/main.tex` / `paper/overleaf_main.tex`) with verified 14-entry BibTeX (`paper/verified.bib`), embedded figures (`fig1`–`fig3`), and complete human-AI collaboration audit (`ai_assistance_log.md`). | `[VERIFIED]` | `paper/overleaf_main.tex`, `ai_assistance_log.md` | 
