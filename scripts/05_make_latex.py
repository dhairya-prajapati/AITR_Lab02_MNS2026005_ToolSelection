"""Generates paper/numbers.tex (macros) and paper/tables_generated.tex from results/*. The paper prose uses only these macros."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
R = ROOT / "results"; W = {"10": "Nten", "25": "Ntwentyfive", "50": "Nfifty", "100": "Nhundred", "199": "Nall"}
MC = {"Random": "Rand", "BM25": "BM", "TFIDF-desc": "TFd", "Hybrid(k=5)": "Hyb", "TFIDF+LR": "LR", "BM25+name": "BMn", "TFIDF-desc+name": "TFn", "Hybrid(k=5)+name": "Hybn"}
CW = {0.0: "czero", 0.5: "chalf", 1.0: "cone", 2.0: "ctwo", 5.0: "cfive"}; PW = {0.1: "pTen", 0.3: "pThirty", 0.5: "pFifty"}
KW = {0: "kzero", 1: "kone", 3: "kthree", 5: "kfive", 10: "kten", 20: "ktwenty"}
M = {}
def put(n, v): M[n] = v
pc = lambda x: f"{100 * x:.1f}"
# data + validation
bv = json.load(open(R / "baseline_validation.json")); dr = bv["data_report"]
put("DrawRows", f"{dr['raw_rows']:,}"); put("DafterDup", f"{dr['after_exact_dup_removal']:,}"); put("DambigRemoved", str(dr["ambiguous_norm_queries_removed"]))
put("DafterAmbig", f"{dr['after_ambiguity_removal']:,}"); put("DfinalRows", f"{dr['after_norm_dedup']:,}"); put("Dtools", str(dr["n_tools"]))
put("DminQ", str(dr["min_q_per_tool"])); put("DmaxQ", f"{dr['max_q_per_tool']:,}"); put("DmedQ", f"{dr['median_q_per_tool']:.0f}")
for k in ("train", "val", "test"): put("Dsplit" + k, f"{bv['split_sizes'][k]:,}")
put("VbmDiff", f"{bv['bm25_vs_rank_bm25_max_abs_diff']:.1e}".replace("e-", r"\times 10^{-") + "}")
put("VrandTop", pc(bv["random_baseline"]["top1"])); put("VchanceTop", pc(bv["chance_1_over_N"]))
put("VshufTop", pc(bv["lr_label_shuffled_control"]["top1"])); put("VshufMacro", pc(bv["lr_label_shuffled_control"]["macro_top1"]))
put("VselC", str(bv["lr_selected_C"]))
lc = pd.read_csv(R / "lr_C_selection_val.csv")
for r in lc.itertuples(): put(f"VvalC{int(r.C)}".replace("1", "I").replace("0", "O"), pc(r.val_top1))
# E1
e1 = pd.read_csv(R / "e1_closed_set.csv"); g = e1.groupby(["N", "method"])
for (N, m), gg in g:
    for met, mn in (("top1", "Top"), ("top3", "Three"), ("mrr", "Mrr"), ("macro_top1", "Macro")):
        put(f"E{MC[m]}{W[str(N)]}{mn}", f"{gg[met].mean():.3f}" if met == "mrr" else pc(gg[met].mean()))
        put(f"E{MC[m]}{W[str(N)]}{mn}Sd", f"{gg[met].std(ddof=1):.3f}" if met == "mrr" else pc(gg[met].std(ddof=1)))
# E2
e2 = pd.concat([pd.read_csv(R / "e2_unseen_tools.csv"), pd.read_csv(R / "e2_name_ablation.csv")])
for (m, s), gg in e2.groupby(["method", "subset"]):
    put(f"T{MC[m]}{s}", pc(gg.top1.mean())); put(f"T{MC[m]}{s}Sd", pc(gg.top1.std(ddof=1)))
def paired(a, b, sub):
    x = e2[(e2.method == a) & (e2.subset == sub)].sort_values("seed").top1.values; y = e2[(e2.method == b) & (e2.subset == sub)].sort_values("seed").top1.values
    d = x - y; return d.mean(), d.std(ddof=1), int((d > 0).sum())
for nm, a, b in (("HybTFd", "Hybrid(k=5)", "TFIDF-desc"), ("BMnBM", "BM25+name", "BM25"), ("TFnTFd", "TFIDF-desc+name", "TFIDF-desc")):
    for sub in ("seen", "unseen"):
        mu, sd, npos = paired(a, b, sub); put(f"P{nm}{sub}", f"{100*mu:+.1f}"); put(f"P{nm}{sub}Sd", f"{100*sd:.1f}"); put(f"P{nm}{sub}Pos", str(npos))
ek = pd.read_csv(R / "e2_exemplar_k_sweep.csv")
for (k, s), gg in ek.groupby(["k", "subset"]): put(f"K{KW[k]}{s}", pc(gg.top1.mean())); put(f"K{KW[k]}{s}Sd", pc(gg.top1.std(ddof=1)))
# E3
a3 = pd.read_csv(R / "e3_auroc.csv")
for m, gg in a3.groupby("method"):
    put(f"A{MC[m]}", f"{gg.auroc_test.mean():.3f}"); put(f"A{MC[m]}Sd", f"{gg.auroc_test.std(ddof=1):.3f}"); put(f"A{MC[m]}acc", pc(gg.acc_in_lib_test.mean()))
put("AnIn", f"{int(a3.n_in.mean()):,}"); put("AnOut", f"{int(a3.n_out.mean()):,}")
e3 = pd.read_csv(R / "e3_abstention.csv")
for (m, c, p), gg in e3.groupby(["method", "cost", "p_out"]):
    put(f"S{MC[m]}{CW[c]}{PW[p]}All", f"{gg.eas_answer_all.mean():.3f}"); put(f"S{MC[m]}{CW[c]}{PW[p]}Abs", f"{gg.eas_abstain.mean():.3f}")
    put(f"S{MC[m]}{CW[c]}{PW[p]}CovIn", pc(gg.coverage_in.mean())); put(f"S{MC[m]}{CW[c]}{PW[p]}CovOut", pc(gg.coverage_out.mean()))
    put(f"S{MC[m]}{CW[c]}{PW[p]}Sel", pc(gg.sel_acc_in.mean()))
# E4 + efficiency
ea = json.load(open(R / "error_analysis.json"))
for m in ("BM25", "Hybrid(k=5)", "TFIDF+LR"):
    x = ea[m]; put(f"X{MC[m]}Err", f"{x['n_err']:,}"); put(f"X{MC[m]}TopShare", pc(x["share_errors_in_top20_pairs"])); put(f"X{MC[m]}Spear", f"{x['spearman_acc_vs_trainsize']:+.2f}")
    put(f"X{MC[m]}MacroPt", pc(x["macro_vs_micro_top1"][0])); put(f"X{MC[m]}Micro", pc(x["macro_vs_micro_top1"][1]))
for key, nm in (("bootstrap_seed0_Hybrid(k=5)_minus_BM25", "HybBM"), ("bootstrap_seed0_TFIDF+LR_minus_Hybrid(k=5)", "LRHyb"), ("bootstrap_seed0_BM25_minus_TFIDF-desc", "BMTFd")):
    put(f"B{nm}", f"{100*ea[key]['mean']:+.1f}"); put(f"B{nm}Lo", f"{100*ea[key]['ci95'][0]:+.1f}"); put(f"B{nm}Hi", f"{100*ea[key]['ci95'][1]:+.1f}")
ef = pd.read_csv(R / "table_efficiency.csv")
for r in ef.itertuples(): put(f"F{MC[r.method]}Fit", f"{r.fit_s:.3f}"); put(f"F{MC[r.method]}Ms", f"{r.single_query_ms:.2f}")
with open(ROOT / "paper/numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by scripts/05_make_latex.py -- do not edit\n")
    for k, v in sorted(M.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------- tables
def ms(s, d=1): return f"{100*s.mean():.{d}f}\\,$\\pm$\\,{100*s.std(ddof=1):.{d}f}"
T = []
t = ["\\begin{table}[t]\\centering\\caption{Data cleaning (script \\texttt{src/data.py}). All counts are produced by the code.}\\label{tab:data}\\small\\begin{tabular}{lr}\\toprule Step & Rows\\\\\\midrule",
     f"Raw MetaTool single-tool queries & {dr['raw_rows']:,}\\\\", f"After dropping exact duplicate (query,tool) rows & {dr['after_exact_dup_removal']:,}\\\\",
     f"After dropping {dr['ambiguous_norm_queries_removed']} ambiguous texts (>1 gold tool) & {dr['after_ambiguity_removal']:,}\\\\", f"After one-row-per-normalised-text & {dr['after_norm_dedup']:,}\\\\",
     f"Queries/tool (min / median / max) & {dr['min_q_per_tool']} / {dr['median_q_per_tool']:.0f} / {dr['max_q_per_tool']:,}\\\\\\bottomrule\\end{{tabular}}\\end{{table}}"]
T.append("\n".join(t))
mm = ["BM25", "TFIDF-desc", "Hybrid(k=5)", "TFIDF+LR"]
t = ["\\begin{table*}[t]\\centering\\caption{E1: closed-set top-1 accuracy (\\%), mean\\,$\\pm$\\,s.d.\\ over 5 seeds, versus library size $N$ (chance $=100/N$).}\\label{tab:e1}\\small\\begin{tabular}{lccccc}\\toprule Method & $N{=}10$ & $N{=}25$ & $N{=}50$ & $N{=}100$ & $N{=}199$\\\\\\midrule"]
for m in ["Random"] + mm:
    t.append(m.replace("+", "+").replace("%", "\\%") + " & " + " & ".join(ms(e1[(e1.method == m) & (e1.N == N)].top1) for N in [10, 25, 50, 100, 199]) + "\\\\")
t.append("\\bottomrule\\end{tabular}\\end{table*}"); T.append("\n".join(t))
t = ["\\begin{table}[t]\\centering\\caption{E1 at $N{=}199$: additional metrics (mean over 5 seeds).}\\label{tab:e1b}\\small\\begin{tabular}{lccc}\\toprule Method & Top-3 (\\%) & MRR & Macro top-1 (\\%)\\\\\\midrule"]
for m in mm:
    x = e1[(e1.method == m) & (e1.N == 199)]; t.append(f"{m} & {ms(x.top3)} & {x.mrr.mean():.3f} & {ms(x.macro_top1)}\\\\")
t.append("\\bottomrule\\end{tabular}\\end{table}"); T.append("\n".join(t))
e2f = pd.concat([pd.read_csv(R / "e2_unseen_tools.csv"), pd.read_csv(R / "e2_name_ablation.csv")])
t = ["\\begin{table}[t]\\centering\\caption{E2: top-1 (\\%) over the full 199-tool library on test queries of tools whose training queries were seen / withheld ($\\approx$40 tools per seed). ``+name'' appends split tool-name tokens to the tool document.}\\label{tab:e2}\\small\\begin{tabular}{lcc}\\toprule Method & Seen tools & Unseen tools\\\\\\midrule"]
for m in ["BM25", "BM25+name", "TFIDF-desc", "TFIDF-desc+name", "Hybrid(k=5)", "Hybrid(k=5)+name", "TFIDF+LR"]:
    t.append(f"{m} & {ms(e2f[(e2f.method == m) & (e2f.subset == 'seen')].top1)} & {ms(e2f[(e2f.method == m) & (e2f.subset == 'unseen')].top1)}\\\\")
t.append("\\bottomrule\\end{tabular}\\end{table}"); T.append("\n".join(t))
t = ["\\begin{table}[t]\\centering\\caption{E3: EAS$_c$ at $p_{out}{=}0.5$ (mean of 5 seeds): always answering vs.\\ validation-tuned abstention. Abstention gain is positive by construction of the tuning (the threshold set includes ``answer all'') but is evaluated on held-out test data.}\\label{tab:e3}\\small\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l" + "cc" * 3 + "}\\toprule"]
t.append("& \\multicolumn{2}{c}{$c{=}0.5$} & \\multicolumn{2}{c}{$c{=}1$} & \\multicolumn{2}{c}{$c{=}2$}\\\\ Method & answer & abstain & answer & abstain & answer & abstain\\\\\\midrule")
for m in mm:
    row = [m]
    for c in (0.5, 1.0, 2.0):
        x = e3[(e3.method == m) & (e3.cost == c) & (e3.p_out == .5)]; row += [f"{x.eas_answer_all.mean():.3f}", f"{x.eas_abstain.mean():.3f}"]
    t.append(" & ".join(row) + "\\\\")
t.append("\\bottomrule\\end{tabular}}\\end{table}"); T.append("\n".join(t))
t = ["\\begin{table}[t]\\centering\\caption{Resource use at $N{=}199$ (1 vCPU Xeon 2.1\\,GHz): training/indexing time and single-query latency.}\\label{tab:eff}\\small\\begin{tabular}{lcc}\\toprule Method & Fit/index (s) & Latency (ms/query)\\\\\\midrule"]
for r in ef.itertuples(): t.append(f"{r.method} & {r.fit_s:.3f} & {r.single_query_ms:.2f}\\\\")
t.append("\\bottomrule\\end{tabular}\\end{table}"); T.append("\n".join(t))
cf = ea["TFIDF+LR"]["top10_confusions"][:6]
t = ["\\begin{table}[t]\\centering\\caption{Most frequent confusions of the supervised model (gold $\\rightarrow$ predicted), pooled over 5 seeds.}\\label{tab:conf}\\small\\begin{tabular}{llr}\\toprule Gold tool & Predicted tool & Errors\\\\\\midrule"]
for a_, b_, n in cf: t.append(f"\\texttt{{{a_.replace('_', chr(92) + '_')}}} & \\texttt{{{b_.replace('_', chr(92) + '_')}}} & {n}\\\\")
t.append("\\bottomrule\\end{tabular}\\end{table}"); T.append("\n".join(t))
open(ROOT / "paper/tables_generated.tex", "w").write("\n\n".join(T))
print(len(M), "macros;", len(T), "tables written")
