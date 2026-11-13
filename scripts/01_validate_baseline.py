"""G2 baseline verification: reference-implementation agreement, controls, leakage audit, LR-C selection (validation only)."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from src.data import load_clean, split_queries
from src.methods import BM25, TfidfRetriever, SupervisedLR, RandomBaseline, TOKEN, tool_doc
from src.metrics import closed_set_metrics

cfg = json.load(open(ROOT / "configs/protocol.json"))
df, desc, rep = load_clean(); tools = sorted(df.Tool.unique()); tidx = {t: i for i, t in enumerate(tools)}
out = {"data_report": rep}; seed = 0
d = split_queries(df, seed); tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
out["split_sizes"] = {"train": len(tr), "val": len(va), "test": len(te)}

# --- 1. leakage audit
nt = lambda x: set(x.norm)
out["leak_norm_text_overlap"] = {"train&test": len(nt(tr) & nt(te)), "train&val": len(nt(tr) & nt(va)), "val&test": len(nt(va) & nt(te))}
out["tools_in_test_not_in_train"] = len(set(te.Tool) - set(tr.Tool))
assert sum(out["leak_norm_text_overlap"].values()) == 0 and out["tools_in_test_not_in_train"] == 0

# --- 2. BM25 vs reference rank_bm25 (same tokenizer / stopwords / k1 / b / epsilon)
an = CountVectorizer(stop_words="english", token_pattern=TOKEN).build_analyzer()
ref = BM25Okapi([an(tool_doc(t, desc)) for t in tools], k1=1.5, b=0.75, epsilon=0.25)
mine = BM25().fit(tools, desc)
sample = te.Query.sample(300, random_state=0).tolist()
S_mine = mine.score(sample); S_ref = np.array([ref.get_scores(an(q)) for q in sample])
out["bm25_vs_rank_bm25_max_abs_diff"] = float(np.abs(S_mine - S_ref).max())
assert out["bm25_vs_rank_bm25_max_abs_diff"] < 1e-8, "BM25 implementation disagrees with reference"

# --- 3. controls
gold = te.Tool.map(tidx).values
out["random_baseline"] = closed_set_metrics(RandomBaseline(seed).fit(tools, desc).score(te.Query), gold)
out["chance_1_over_N"] = 1 / len(tools)
shuf = SupervisedLR(C=10, max_iter=cfg["lr"]["max_iter"], shuffle_labels=True).fit(tools, desc, tr)
out["lr_label_shuffled_control"] = closed_set_metrics(shuf.score(te.Query), gold)
k0 = TfidfRetriever(0).fit(tools, desc, tr); h0 = TfidfRetriever(5, seed=0).fit(tools, desc, tr)
out["tfidf_k0_equals_desc_only"] = bool(np.allclose(k0.score(sample), TfidfRetriever(0).fit(tools, desc).score(sample)))

# --- 4. LR regularisation chosen on VALIDATION only (never test)
rows = []
for C in cfg["lr"]["C_grid_for_validation_only"]:
    t0 = time.time(); m = SupervisedLR(C=C, max_iter=cfg["lr"]["max_iter"]).fit(tools, desc, tr)
    r = closed_set_metrics(m.score(va.Query), va.Tool.map(tidx).values)
    rows.append({"C": C, "val_top1": r["top1"], "fit_s": time.time() - t0}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/lr_C_selection_val.csv", index=False)
bestC = max(rows, key=lambda r: r["val_top1"])["C"]; out["lr_selected_C"] = bestC
json.dump({"lr_C": bestC}, open(ROOT / "configs/selected_hparams.json", "w"))

# --- 5. closed-set baseline numbers on seed-0 test (sanity read-out only; full study in script 02)
for m in (BM25().fit(tools, desc), k0, h0, SupervisedLR(C=bestC, max_iter=cfg["lr"]["max_iter"]).fit(tools, desc, tr)):
    out[f"seed0_test_{m.name}"] = closed_set_metrics(m.score(te.Query), gold)
json.dump(out, open(ROOT / "results/baseline_validation.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
print("BASELINE VALIDATION PASSED")
