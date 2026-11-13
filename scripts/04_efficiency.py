"""Resource efficiency: fit time, batch and single-query latency, N=199, seed 0, single CPU thread."""
import sys, json, time, platform, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from src.data import load_clean, split_queries
from src.methods import BM25, TfidfRetriever, SupervisedLR
cfg = json.load(open(ROOT / "configs/protocol.json")); C = json.load(open(ROOT / "configs/selected_hparams.json"))["lr_C"]
df, desc, _ = load_clean(); tools = sorted(df.Tool.unique()); d = split_queries(df, 0); tr, te = d[d.split == "train"], d[d.split == "test"].head(300)
rows = []
for m in (BM25(), TfidfRetriever(0), TfidfRetriever(5, 0), SupervisedLR(C, cfg["lr"]["max_iter"])):
    t0 = time.perf_counter(); m.fit(tools, desc, tr); fit = time.perf_counter() - t0
    m.score(te.Query.head(5))  # warm-up
    t0 = time.perf_counter(); [m.score([q]) for q in te.Query]; single = (time.perf_counter() - t0) * 1000 / len(te)
    rows.append({"method": m.name, "fit_s": round(fit, 3), "single_query_ms": round(single, 3)})
pd.DataFrame(rows).to_csv(ROOT / "results/table_efficiency.csv", index=False)
print(platform.processor() or platform.machine(), "| 1 vCPU"); print(pd.DataFrame(rows).to_string(index=False))
