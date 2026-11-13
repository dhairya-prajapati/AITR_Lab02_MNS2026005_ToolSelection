"""Builds every table / figure in the paper from results/*.csv (no numbers are typed by hand)."""
import sys, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src.data import load_clean, split_queries

R = ROOT / "results"; F = R / "figures"; PF = ROOT / "paper/figures"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"Random": "gray", "BM25": "tab:blue", "TFIDF-desc": "tab:cyan", "Hybrid(k=5)": "tab:green", "TFIDF+LR": "tab:red"}
MK = {"Random": "x", "BM25": "o", "TFIDF-desc": "s", "Hybrid(k=5)": "^", "TFIDF+LR": "D"}
ms = lambda s: f"{s.mean()*100:.1f} ± {s.std(ddof=1)*100:.1f}"
def save(name):
    plt.savefig(F / f"{name}.png", dpi=200); plt.savefig(F / f"{name}.pdf"); plt.close()

# ---------- E1
e1 = pd.read_csv(R / "e1_closed_set.csv")
t = e1.groupby(["N", "method"]).agg(top1=("top1", ms), top3=("top3", ms), mrr=("mrr", lambda s: f"{s.mean():.3f}"), macro=("macro_top1", ms)).reset_index()
t.to_csv(R / "table_e1_scaling.csv", index=False)
g = e1.groupby(["method", "N"]).top1.agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(4.4, 3.1))
for m, gg in g.groupby("method"):
    ax.errorbar(gg.N, gg["mean"] * 100, yerr=gg["std"] * 100, label=m, color=COL[m], marker=MK[m], ms=4, capsize=2, lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("Tool-library size N"); ax.set_ylabel("Top-1 accuracy (%)"); ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); save("fig_e1_scaling")

# ---------- E2
e2 = pd.read_csv(R / "e2_unseen_tools.csv"); e2n = pd.read_csv(R / "e2_name_ablation.csv"); e2k = pd.read_csv(R / "e2_exemplar_k_sweep.csv")
t2 = e2.pivot_table(index="method", columns="subset", values="top1", aggfunc=[ms] if False else (lambda s: ms(s))); t2.to_csv(R / "table_e2_unseen.csv")
tn = pd.concat([e2, e2n]).pivot_table(index="method", columns="subset", values="top1", aggfunc=lambda s: ms(s)); tn.to_csv(R / "table_e2_with_name_ablation.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.0)); meths = ["BM25", "TFIDF-desc", "Hybrid(k=5)", "TFIDF+LR"]; w = .38
for j, sub in enumerate(["seen", "unseen"]):
    v = [e2[(e2.method == m) & (e2.subset == sub)].top1 for m in meths]
    ax.bar(np.arange(4) + (j - .5) * w, [x.mean() * 100 for x in v], w, yerr=[x.std(ddof=1) * 100 for x in v], capsize=2, label=f"{sub} tools", color=["tab:blue", "tab:orange"][j])
ax.set_xticks(range(4)); ax.set_xticklabels(meths, fontsize=7); ax.set_ylabel("Top-1 accuracy (%), N=199"); ax.legend(fontsize=7); save("fig_e2_unseen")
tk = e2k.groupby(["k", "subset"]).agg(top1=("top1", ms), mrr=("mrr", lambda s: f"{s.mean():.3f}")).reset_index(); tk.to_csv(R / "table_e2_k_sweep.csv", index=False)
gk = e2k.groupby(["k", "subset"]).top1.agg(["mean", "std"]).reset_index(); fig, ax = plt.subplots(figsize=(4.4, 3.0))
for sub, c in (("seen", "tab:blue"), ("unseen", "tab:orange")):
    gg = gk[gk.subset == sub]; ax.errorbar(gg.k, gg["mean"] * 100, yerr=gg["std"] * 100, marker="o", ms=4, capsize=2, color=c, label=f"{sub} tools")
ax.set_xlabel("Exemplar queries k per seen tool in tool document"); ax.set_ylabel("Top-1 accuracy (%)"); ax.legend(fontsize=7); save("fig_e2_k_sweep")

# ---------- E3
e3 = pd.read_csv(R / "e3_abstention.csv"); e3a = pd.read_csv(R / "e3_auroc.csv")
a = e3a.groupby("method").agg(auroc=("auroc_test", lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"), acc_in=("acc_in_lib_test", ms)).reset_index(); a.to_csv(R / "table_e3_auroc.csv", index=False)
p5 = e3[e3.p_out == .5]
t3 = p5.groupby(["method", "cost"]).agg(eas_answer_all=("eas_answer_all", lambda s: f"{s.mean():.3f}"), eas_abstain=("eas_abstain", lambda s: f"{s.mean():.3f}"),
     delta=("eas_abstain", "count")).reset_index().drop(columns="delta")
d3 = p5.assign(gain=p5.eas_abstain - p5.eas_answer_all).groupby(["method", "cost"]).gain.agg(["mean", "std"]).reset_index(); d3.to_csv(R / "table_e3_abstention_gain.csv", index=False)
t3.to_csv(R / "table_e3_eas_p05.csv", index=False)
cov = p5.groupby(["method", "cost"]).agg(coverage_in=("coverage_in", "mean"), coverage_out=("coverage_out", "mean"), sel_acc_in=("sel_acc_in", "mean")).reset_index(); cov.to_csv(R / "table_e3_coverage.csv", index=False)
sens = e3.groupby(["method", "p_out", "cost"]).agg(eas_abstain=("eas_abstain", "mean"), eas_answer_all=("eas_answer_all", "mean")).reset_index(); sens.to_csv(R / "table_e3_sensitivity_pout.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, col, ttl in ((axs[0], "eas_answer_all", "Always answer"), (axs[1], "eas_abstain", "Abstain (val-tuned threshold)")):
    for m, gg in p5.groupby("method"):
        s = gg.groupby("cost")[col].agg(["mean", "std"]); ax.errorbar(s.index, s["mean"], yerr=s["std"], label=m, color=COL[m], marker=MK[m], ms=4, capsize=2)
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("Cost c of a wrong selection"); ax.set_title(ttl, fontsize=9)
axs[0].set_ylabel(r"EAS$_c$ (p$_{out}$=0.5)"); axs[0].legend(fontsize=7); save("fig_e3_eas")

# ---------- E4 error analysis (pooled over seeds, N=199)
df, desc, _ = load_clean(); ALL = sorted(df.Tool.unique()); P = pd.read_csv(R / "predictions_closed_199.csv"); P["ok"] = P.gold == P.pred
P["gold_t"] = P.gold.map(dict(enumerate(ALL))); P["pred_t"] = P.pred.map(dict(enumerate(ALL))); P["len"] = P.qid.map(df.Query.str.split().str.len())
ea = {}
for m in ["BM25", "Hybrid(k=5)", "TFIDF+LR"]:
    q = P[P.method == m]; err = q[~q.ok]
    pairs = err.groupby(["gold_t", "pred_t"]).size().sort_values(ascending=False)
    ea[m] = {"n_err": int(len(err)), "top10_confusions": [[a, b, int(n)] for (a, b), n in pairs.head(10).items()],
             "share_errors_in_top20_pairs": float(pairs.head(20).sum() / len(err)),
             "most_frequent_predicted_when_wrong": err.pred_t.value_counts().head(5).to_dict()}
    ntr = d_tr = split_queries(df, 0).query("split=='train'").Tool.value_counts()
    pt = q.groupby("gold_t").ok.mean(); ea[m]["spearman_acc_vs_trainsize"] = float(spearmanr(pt.index.map(ntr).values, pt.values)[0])
    bins = pd.cut(q.len, [0, 8, 12, 18, 25, 40, 200]); ea[m]["acc_by_len_bin"] = {str(k): float(v) for k, v in q.groupby(bins, observed=True).ok.mean().items()}
    ea[m]["macro_vs_micro_top1"] = [float(pt.mean()), float(q.ok.mean())]
# misclassified examples for the LR model, seed 0
ex = P[(P.method == "TFIDF+LR") & (~P.ok) & (P.seed == 0)].sample(8, random_state=0)
ea["examples_LR_errors_seed0"] = [{"query": df.loc[r.qid, "Query"], "gold": r.gold_t, "pred": r.pred_t} for r in ex.itertuples()]
# paired bootstrap seed 0
rng = np.random.default_rng(0); piv = P[P.seed == 0].pivot(index="qid", columns="method", values="ok").astype(float); idx = rng.integers(0, len(piv), (2000, len(piv)))
for a_, b_ in (("Hybrid(k=5)", "BM25"), ("TFIDF+LR", "Hybrid(k=5)"), ("BM25", "TFIDF-desc")):
    diff = (piv[a_].values - piv[b_].values)[idx].mean(1); ea[f"bootstrap_seed0_{a_}_minus_{b_}"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
json.dump(ea, open(R / "error_analysis.json", "w"), indent=2)
# per-tool accuracy vs support figure
fig, ax = plt.subplots(figsize=(4.4, 3.0)); ntr = split_queries(df, 0).query("split=='train'").Tool.value_counts()
for m in ["BM25", "TFIDF+LR"]:
    pt = P[P.method == m].groupby("gold_t").ok.mean(); ax.scatter(pt.index.map(ntr), pt.values * 100, s=8, alpha=.6, color=COL[m], label=m)
ax.set_xscale("log"); ax.set_xlabel("Training queries available for the tool"); ax.set_ylabel("Per-tool top-1 (%)"); ax.legend(fontsize=7); save("fig_e4_pertool")
for f in F.glob("fig_*.pdf"): shutil.copy(f, PF / f.name)
print(json.dumps(ea, indent=1)[:6000]); print("ANALYSIS DONE")
