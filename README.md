# Evaluating Tool-Selection Accuracy in Multi-Tool Agentic AI Systems

**Submission Folder:** `AITR_Lab02_MNS2026005_ToolSelection`  
**Student Name:** Dhairya Ashvinbhai Prajapati  
**Enrollment / Roll Number:** MNS2026005  
**Department:** Networks Systems and Computing, India  
**Course:** AI Tools for Research — Post-Mid-Sem Implementation Study (Topic ID: T12)  

---

## 1. Overview & Research Questions
This repository contains the complete code, frozen dataset partition, experimental logs, publication figures, provenance documents, and LaTeX manuscript for our implementation paper extending the completed literature review (`MNS2026005_DHAIRYA PRAJAPATI.pdf`).

We investigate two core limitations of current multi-tool agent evaluations:
1. **Upstream Tool-Use Awareness Failure:** Standard benchmarks assume every user prompt requires an external API call, ignoring whether the agent knows *when* to abstain and answer from internal knowledge (Huang et al., 2024 [MetaTool]).
2. **Risk-Blind Binary Evaluation:** Standard top-1 binary accuracy ($\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}[t_i = t_i^*]$, Equation 1) treats harmless abstentions and costly or unsafe false-positive invocations equally.

We implement a **Two-Stage Awareness-Gated Hybrid Tool Selector** (`Hybrid_Awareness_2Stage`) over the official **MetaTool (`ToolE`)** catalog ($199$ API plugins, $N=300$ mixed evaluation split) and evaluate it under both **Binary Accuracy (Equation 1)** and our **Efficiency-Adjusted Selection Score ($\text{EAS}$, Equation 2)**:
$$\text{EAS} = \frac{1}{N}\sum_{i=1}^{N}\left(\mathbb{I}[t_i = t_i^*] - c_i\right)$$

---

## 2. Project Directory Structure
```text
AITR_Lab02_MNS2026005_ToolSelection/
├── README.md                              # Setup, commands, and reproduction guide
├── requirements.txt                       # Verified Python dependency versions
├── configs/
│   └── experiment_config.yaml             # Frozen seed (42), split sizes, and EAS penalties
├── data/
│   ├── README.md                          # MetaTool dataset access, license, and schema notes
│   └── processed/
│       ├── tool_catalog.json              # 199 verified MetaTool API plugin descriptions
│       └── eval_dataset_n300.json         # Frozen N=300 split (200 positive + 100 no-tool)
├── src/
│   ├── __init__.py
│   ├── seed_utils.py                      # Deterministic seed control (Python, NumPy, PyTorch)
│   ├── data_loader.py                     # Downloads MetaTool & builds N=300 partition
│   ├── retriever.py                       # BM25, Dense (MiniLM-L6-v2), and Hybrid RRF engine
│   └── awareness_selector.py              # Stage 0 Awareness Gate + 2-stage selector
├── scripts/
│   ├── env_sanity_check.py                # Prompt 08 environment & toy EAS verification
│   ├── evaluate_baseline.py               # Prompt 09-10 1-stage baseline evaluation runner
│   └── run_experiments.py                 # Prompt 12-14 full comparison, ablations & plots
├── results/
│   ├── env_verification_log.json          # Genuine environment check console output
│   ├── module_9_1_data_summary.json       # Verified dataset loader summary (199 tools)
│   ├── baseline_metrics.json              # Verified 1-stage dense baseline metrics
│   ├── baseline_predictions.csv           # Instance-level baseline predictions (300 rows)
│   ├── results.csv                        # Main comparative metrics table (6 methods)
│   ├── ablation_topk_and_tau.csv          # Top-k pool size and threshold (tau) ablation data
│   └── figures/
│       ├── fig1_accuracy_vs_eas.png       # Figure 1: Binary Accuracy vs. EAS bar chart
│       ├── fig2_eas_sensitivity.png       # Figure 2: EAS penalty sensitivity curves (c_fp)
│       └── fig3_topk_and_tau_ablation.png # Figure 3: Top-k recall & threshold trade-off
├── paper/
│   ├── overleaf_main.tex                  # Complete LaTeX manuscript source
│   ├── verified.bib                       # Audited 14-reference BibTeX file
│   ├── fig1_accuracy_vs_eas.pdf           # Vector PDF of Figure 1 for Overleaf
│   ├── fig2_eas_sensitivity.pdf           # Vector PDF of Figure 2 for Overleaf
│   └── fig3_topk_and_tau_ablation.pdf     # Vector PDF of Figure 3 for Overleaf
├── research_decisions.md                  # Gate G1 One-Page Brief & decision log
├── baseline_validation.md                 # Gate G2 Baseline reproduction & discrepancy report
└── ai_assistance_log.md                   # Gate G4 Human-AI collaboration & debugging audit
```

---

## 3. Reproducible Setup & Execution Commands

Verified on **Linux (`Linux-6.6.159+-x86_64-with-glibc2.39`)**, **Python `3.13.18`**, CPU execution (`torch 2.11.0+cpu`, `sentence-transformers 5.7.0`, `scikit-learn 1.6.1`, `pandas 2.2.3`, `numpy 2.1.3`).

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install verified dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Step 1 (Gate G1/G2): Run Environment & Toy EAS Sanity Check
python3 scripts/env_sanity_check.py

# 4. Step 2 (Gate G2): Download MetaTool Dataset & Build Frozen N=300 Split (seed=42)
python3 src/data_loader.py

# 5. Step 3 (Gate G2): Run & Validate 1-Stage Forced Dense Baseline
python3 scripts/evaluate_baseline.py

# 6. Step 4 (Gate G3/G4): Run Full 6-Method Comparison, Ablations, and Plot Generation
python3 scripts/run_experiments.py
```

*(Note for Google Colab / Jupyter Users: All scripts include the `"__file__" in globals() else os.getcwd()` path guard so they can be executed either via `!python3 scripts/run_experiments.py` or pasted directly into notebook cells.)*

---

## 4. Summary of Verified Experimental Results ($N = 300$, `seed = 42`)

| Method | Stage 0 Gate | Awareness Acc | FPIR $\downarrow$ | FNAR $\downarrow$ | Top-1 Acc (Eq. 1) | Top-5 Acc | EAS Top-1 (Eq. 2) | EAS Top-5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BM25_Forced_1Stage` | No | `0.6667` | `1.0000` | `0.0000` | `0.3467` | `0.4667` | `-0.2700` | `-0.0300` |
| `Dense_Forced_1Stage` (Baseline) | No | `0.6667` | `1.0000` | `0.0000` | `0.4233` | `0.5667` | `-0.1550` | `+0.2000` |
| `Hybrid_Forced_1Stage` | No | `0.6667` | `1.0000` | `0.0000` | `0.4600` | `0.6000` | `-0.0467` | `+0.2667` |
| `BM25_Awareness_2Stage` | Yes ($\tau=0.36$) | `0.8800` | `0.1200` | `0.1200` | `0.6167` | `0.7467` | `+0.2733` | `+0.5333` |
| `Dense_Awareness_2Stage` | Yes ($\tau=0.36$) | `0.9467` | `0.0600` | `0.0500` | `0.7133` | `0.8467` | `+0.4567` | `+0.7233` |
| **`Hybrid_Awareness_2Stage` (Proposed)** | **Yes ($\tau=0.36$)** | **`0.9533`** | **`0.0500`** | **`0.0450`** | **`0.7533`** | **`0.8833`** | **`+0.5333`** | **`+0.7933`** |

---

## 5. License & Citation
- **Code & Documentation License:** MIT License.
- **Benchmark Attribution:** MetaTool (`ToolE`) dataset by Huang et al. (ICLR 2024, `arXiv:2310.03128`).