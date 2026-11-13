"""Data loading, cleaning and splitting for the MetaTool ToolE single-tool subset."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def normalise(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()

def load_clean():
    """Load queries + descriptions and clean. Every step is counted in the returned report.
    1. drop exact duplicate (Query, Tool) rows
    2. drop queries whose normalised text maps to >1 distinct tool (ambiguous gold label)
    3. keep one row per normalised query text (prevents train/test text leakage)."""
    df = pd.read_csv(ROOT / "data/metatool_all_clean_data.csv")
    desc = json.load(open(ROOT / "data/metatool_plugin_des.json"))
    rep = {"raw_rows": len(df)}
    df["norm"] = df.Query.map(normalise)
    df = df.drop_duplicates(["Query", "Tool"]); rep["after_exact_dup_removal"] = len(df)
    ntool = df.groupby("norm").Tool.nunique()
    amb = set(ntool[ntool > 1].index)
    rep["ambiguous_norm_queries_removed"] = len(amb)
    df = df[~df.norm.isin(amb)]; rep["after_ambiguity_removal"] = len(df)
    df = df.drop_duplicates("norm"); rep["after_norm_dedup"] = len(df)
    df = df.reset_index(drop=True)
    rep["n_tools"] = int(df.Tool.nunique()); rep["all_tools_have_description"] = bool(df.Tool.isin(list(desc)).all())
    c = df.Tool.value_counts()
    rep["min_q_per_tool"] = int(c.min()); rep["max_q_per_tool"] = int(c.max()); rep["median_q_per_tool"] = float(c.median())
    return df, desc, rep

def split_queries(df, seed, fr=(0.6, 0.2, 0.2)):
    """Per-tool stratified random split -> column 'split' in {train,val,test}."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for t, idx in df.groupby("Tool").indices.items():
        idx = rng.permutation(idx); n = len(idx)
        ntr, nva = int(round(fr[0] * n)), int(round(fr[1] * n))
        split[idx[:ntr]] = "train"; split[idx[ntr:ntr + nva]] = "val"; split[idx[ntr + nva:]] = "test"
    out = df.copy(); out["split"] = split
    return out
