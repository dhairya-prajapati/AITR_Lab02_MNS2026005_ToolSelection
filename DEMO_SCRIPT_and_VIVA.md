# 5-minute demonstration script + viva preparation

## Demo (≈5 min)
1. **0:00 – Question (45 s).** Review claimed three things without testing. Show `research_brief_G1.md`. "I test them on MetaTool without LLMs."
2. **0:45 – Data & protocol (45 s).** Show `src/data.py` cleaning counts (20,614 → 20,523), `configs/protocol.json` frozen.
3. **1:30 – Baseline validation (60 s).** Run `python scripts/00_env_check.py`; open `baseline_validation.md`; point to BM25-vs-`rank_bm25` difference 7e-15 and the shuffled-label control (macro ≈ chance, micro 1.7% because of the majority prior).
4. **2:30 – Live re-run (60 s).** Execute `notebooks/quickstart.ipynb` (seed 0): BM25 .330, TF-IDF .336, Hybrid .541, LR .803.
5. **3:30 – Results (75 s).** Fig. 1 (accuracy falls with N), Table "E2" (LR 0% on unseen; Hybrid +3.3 pp paired), Fig. 3 (EAS negative if you always answer when c ≥ 1).
6. **4:45 – Limitations (15 s).** No LLMs; LR is a proxy; LLM-generated queries.

## Likely viva questions (know the answers from the code)
1. Why is the supervised model's 0% on unseen tools "trivial"? *A classifier cannot emit a class absent from training; `SupervisedLR.score` assigns −1 to such columns.*
2. How do you break ties? *Seeded random jitter, `metrics.jitter`; avoids tool-order bias.*
3. Why per-tool splits and normalised-text de-dup? *Prevents the same query text in train and test (leakage); verified 0 overlap.*
4. Where was C chosen? *Script 01, validation split only.*
5. Why does micro ≠ macro? *Tool counts range 12–1,104; class prior.*
6. Explain BM25 in your code. *`BM25.fit`: idf with ε-floor, saturating tf with k1, length norm b; score = Q·Wᵀ.*
7. What does Hybrid change? *Only the tool document: description + k training queries; vectoriser fit on library docs only.*
8. How is the abstention threshold tuned without leaking? *On validation in-library queries + O_val tools; evaluated on test + O_test (disjoint tools).*
9. Why does your EAS differ from Eq. 2 of the review? *Eq. 2 charged cost on correct answers too; operational form charges only wrong answers, abstention = 0.*
10. Why can't you conclude anything about LLMs? *None evaluated; proxy models; LLM-generated data.*
11. Why are bootstrap CIs optimistic? *Resamples queries, ignores dependence between queries of the same tool.*
12. What would falsify RQ1? *No accuracy drop at N=199 vs 10 beyond seed variation — not observed.*
