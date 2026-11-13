import numpy as np
from sklearn.metrics import roc_auc_score

def jitter(S, seed=12345):
    """Break exact score ties at random (fixed seed) so ordering never depends on tool order."""
    return S + np.random.default_rng(seed).random(S.shape) * 1e-9

def ranks_of_gold(S, gold_idx):
    S = jitter(S); g = S[np.arange(len(S)), gold_idx][:, None]
    return 1 + (S > g).sum(1)

def closed_set_metrics(S, gold_idx):
    r = ranks_of_gold(S, gold_idx)
    top1 = (r == 1)
    per_tool = [top1[gold_idx == c].mean() for c in np.unique(gold_idx)]
    return {"top1": top1.mean(), "top3": (r <= 3).mean(), "mrr": (1.0 / r).mean(), "macro_top1": float(np.mean(per_tool)), "n": len(r)}

def top1_and_conf(S):
    S = jitter(S); i = S.argmax(1)
    return i, S[np.arange(len(S)), i]

def eas(correct, answered, c):
    """EAS_c = mean( 1[answered & correct] - c * 1[answered & wrong] ); abstention scores 0.
    (Operational form of Eq. 2 of the review; for c=0 and no abstention it equals top-1 accuracy.)"""
    correct, answered = np.asarray(correct, bool), np.asarray(answered, bool)
    return float(np.mean(np.where(answered, np.where(correct, 1.0, -c), 0.0)))

def auroc_in_vs_out(conf_in, conf_out):
    y = np.r_[np.ones(len(conf_in)), np.zeros(len(conf_out))]
    return float(roc_auc_score(y, np.r_[conf_in, conf_out]))
