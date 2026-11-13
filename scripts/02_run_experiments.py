"""G3 experiments, run exactly as specified in configs/protocol.json.
E1 library-size scaling | E2 unseen tools (+ exemplar-k sensitivity, name ablation) | E3 open-world abstention with EAS."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.data import load_clean, split_queries
from src.methods import BM25, TfidfRetriever, SupervisedLR, RandomBaseline
from src.metrics import closed_set_metrics, top1_and_conf, eas, auroc_in_vs_out

cfg = json.load(open(ROOT / "configs/protocol.json")); C_LR = json.load(open(ROOT / "configs/selected_hparams.json"))["lr_C"]
df, desc, _ = load_clean(); ALL = sorted(df.Tool.unique()); R = ROOT / "results"
K = cfg["hybrid_exemplars_k"]

def make(name, seed):
    return {"Random": lambda: RandomBaseline(seed), "BM25": lambda: BM25(**cfg["bm25"]), "TFIDF-desc": lambda: TfidfRetriever(0),
            f"Hybrid(k={K})": lambda: TfidfRetriever(K, seed), "TFIDF+LR": lambda: SupervisedLR(C_LR, cfg["lr"]["max_iter"], seed)}[name]()
METHODS = ["Random", "BM25", "TFIDF-desc", f"Hybrid(k={K})", "TFIDF+LR"]
e1, e2, e2k, e2n, e3, e3a, preds = [], [], [], [], [], [], []

def timed_eval(m, lib, train, test):
    t0 = time.perf_counter(); m.fit(lib, desc, train); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); S = m.score(test.Query); sc_ms = (time.perf_counter() - t0) * 1000 / len(test)
    return S, fit_s, sc_ms

for seed in cfg["seeds"]:
    print(f"=== seed {seed}", flush=True)
    d = split_queries(df, seed); rng = np.random.default_rng(10_000 + seed)
    # ---------------- E1: library-size scaling (closed set)
    for N in cfg["library_sizes"]:
        lib = ALL if N == len(ALL) else sorted(rng.choice(ALL, N, replace=False).tolist()); ix = {t: i for i, t in enumerate(lib)}
        sub = d[d.Tool.isin(lib)]; train, test = sub[sub.split == "train"], sub[sub.split == "test"]; gold = test.Tool.map(ix).values
        for name in METHODS:
            S, fs, sm = timed_eval(make(name, seed), lib, train, test)
            r = closed_set_metrics(S, gold); r.update(seed=seed, N=N, method=name, fit_s=fs, score_ms_per_q=sm); e1.append(r)
            if N == len(ALL):
                p, cf = top1_and_conf(S)
                preds.append(pd.DataFrame({"seed": seed, "method": name, "qid": test.index.values, "gold": gold, "pred": p, "conf": cf}))
        print(f"  E1 N={N} done", flush=True)
    # ---------------- E2/E3 shared split of tools
    perm = rng.permutation(ALL); n_un = int(round(cfg["unseen_tool_fraction"] * len(ALL)))
    unseen, seen = sorted(perm[:n_un].tolist()), sorted(perm[n_un:].tolist())
    O_val, O_test = unseen[: n_un // 2], unseen[n_un // 2:]
    train_seen = d[(d.split == "train") & d.Tool.isin(seen)]
    test_all = d[d.split == "test"]; gold_all = test_all.Tool.map({t: i for i, t in enumerate(ALL)}).values
    is_unseen = test_all.Tool.isin(unseen).values
    fitted = {}
    for name in METHODS[1:]:
        m = make(name, seed); S, fs, sm = timed_eval(m, ALL, train_seen, test_all); fitted[name] = S
        for sub_name, mask in (("seen", ~is_unseen), ("unseen", is_unseen)):
            r = closed_set_metrics(S[mask], gold_all[mask]); r.update(seed=seed, method=name, subset=sub_name); e2.append(r)
    # name ablation (use tool-name tokens in the document)
    for nm, mk in (("BM25+name", lambda: BM25(**cfg["bm25"], use_name=True)), ("TFIDF-desc+name", lambda: TfidfRetriever(0, use_name=True)),
                   (f"Hybrid(k={K})+name", lambda: TfidfRetriever(K, seed, use_name=True))):
        S, _, _ = timed_eval(mk(), ALL, train_seen, test_all)
        for sub_name, mask in (("seen", ~is_unseen), ("unseen", is_unseen)):
            r = closed_set_metrics(S[mask], gold_all[mask]); r.update(seed=seed, method=nm, subset=sub_name); e2n.append(r)
    # exemplar-k sensitivity
    for k in cfg["hybrid_k_sweep"]:
        S, _, _ = timed_eval(TfidfRetriever(k, seed), ALL, train_seen, test_all)
        for sub_name, mask in (("seen", ~is_unseen), ("unseen", is_unseen)):
            r = closed_set_metrics(S[mask], gold_all[mask]); r.update(seed=seed, k=k, subset=sub_name); e2k.append(r)
    print("  E2 done", flush=True)
    # ---------------- E3: open-world abstention (library = seen tools only)
    ixs = {t: i for i, t in enumerate(seen)}; keep = [ALL.index(t) for t in seen]
    val_in = d[(d.split == "val") & d.Tool.isin(seen)]; te_in = d[(d.split == "test") & d.Tool.isin(seen)]
    out_val = d[d.Tool.isin(O_val)]; out_te = d[d.Tool.isin(O_test)]
    for name in METHODS[1:]:
        m = make(name, seed)
        if name == "TFIDF+LR":   # reuse the model fitted on the seen tools; unseen columns (-1) can never win
            m.fit(ALL, desc, train_seen); sc = lambda q: m.score(q)[:, keep]
        else:
            m.fit(seen, desc, train_seen); sc = m.score
        res = {}
        for key, part in (("val_in", val_in), ("test_in", te_in), ("val_out", out_val), ("test_out", out_te)):
            p, cf = top1_and_conf(sc(part.Query)); res[key] = (p, cf)
        cor = {k: res[k][0] == part.Tool.map(ixs).values for k, part in (("val_in", val_in), ("test_in", te_in))}
        auc = auroc_in_vs_out(res["test_in"][1], res["test_out"][1])
        e3a.append({"seed": seed, "method": name, "auroc_test": auc, "acc_in_lib_test": cor["test_in"].mean(), "n_in": len(te_in), "n_out": len(out_te)})
        cand = np.r_[-np.inf, np.unique(np.quantile(np.r_[res["val_in"][1], res["val_out"][1]], np.linspace(0, 1, 401)))]
        for c in cfg["abstention"]["costs"]:
            for p_out in cfg["abstention"]["p_out_sensitivity"]:
                def total(conf_in, corr_in, conf_out, tau):
                    a_in, a_out = conf_in >= tau, conf_out >= tau
                    return (1 - p_out) * eas(corr_in, a_in, c) + p_out * eas(np.zeros(len(conf_out), bool), a_out, c)
                v = [total(res["val_in"][1], cor["val_in"], res["val_out"][1], t) for t in cand]; tau = cand[int(np.argmax(v))]
                a_in, a_out = res["test_in"][1] >= tau, res["test_out"][1] >= tau
                e3.append({"seed": seed, "method": name, "cost": c, "p_out": p_out, "tau": tau,
                           "eas_abstain": total(res["test_in"][1], cor["test_in"], res["test_out"][1], tau),
                           "eas_answer_all": total(res["test_in"][1], cor["test_in"], res["test_out"][1], -np.inf),
                           "coverage_in": a_in.mean(), "coverage_out": a_out.mean(),
                           "sel_acc_in": (cor["test_in"][a_in].mean() if a_in.any() else np.nan)})
    print("  E3 done", flush=True)

for nm, rows in (("e1_closed_set", e1), ("e2_unseen_tools", e2), ("e2_exemplar_k_sweep", e2k), ("e2_name_ablation", e2n), ("e3_abstention", e3), ("e3_auroc", e3a)):
    pd.DataFrame(rows).to_csv(R / f"{nm}.csv", index=False)
pd.concat(preds).to_csv(R / "predictions_closed_199.csv", index=False)
print("ALL EXPERIMENTS DONE")
