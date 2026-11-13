"""Tool-selection methods. Every method exposes:
    fit(lib_tools, desc, train_df)  and  score(queries) -> ndarray [n_queries, n_lib_tools]
Higher score = more likely tool. Column order == lib_tools order."""
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TOKEN = r"(?u)\b\w\w+\b"

def split_camel(name):
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name))

def tool_doc(tool, desc, use_name=False):
    return (split_camel(tool) + " " if use_name else "") + desc[tool]

class BM25:
    """Okapi BM25 (idf with rank_bm25-style epsilon floor so it can be validated against rank_bm25)."""
    name = "BM25"
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25, use_name=False):
        self.k1, self.b, self.eps, self.use_name = k1, b, epsilon, use_name
    def fit(self, lib_tools, desc, train_df=None):
        docs = [tool_doc(t, desc, self.use_name) for t in lib_tools]
        self.cv = CountVectorizer(stop_words="english", token_pattern=TOKEN)
        tf = self.cv.fit_transform(docs).toarray().astype(float)       # [N, V]
        N = tf.shape[0]; df_ = (tf > 0).sum(0)
        idf = np.log(N - df_ + 0.5) - np.log(df_ + 0.5)
        idf = np.where(idf < 0, self.eps * idf.mean(), idf)
        dl = tf.sum(1, keepdims=True); avg = dl.mean()
        self.W = idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avg))
        return self
    def score(self, queries):
        Q = self.cv.transform(queries).astype(float)                   # counts incl. repeats
        return np.asarray(Q @ self.W.T)

class TfidfRetriever:
    """Cosine similarity between TF-IDF(query) and TF-IDF(tool document).
    With exemplars_k > 0 this is the hybrid: tool doc = description + k sampled TRAIN queries of that tool."""
    def __init__(self, exemplars_k=0, seed=0, use_name=False):
        self.k, self.seed, self.use_name = exemplars_k, seed, use_name
        self.name = "TFIDF-desc" if exemplars_k == 0 else f"Hybrid(k={exemplars_k})"
    def fit(self, lib_tools, desc, train_df=None):
        rng = np.random.default_rng(self.seed); docs = []
        by_tool = {} if train_df is None or self.k == 0 else {t: g.Query.tolist() for t, g in train_df.groupby("Tool")}
        for t in lib_tools:
            d = tool_doc(t, desc, self.use_name)
            ex = by_tool.get(t, [])
            if self.k > 0 and ex:
                pick = rng.choice(len(ex), size=min(self.k, len(ex)), replace=False)
                d += " " + " ".join(ex[i] for i in pick)
            docs.append(d)
        self.vec = TfidfVectorizer(stop_words="english", token_pattern=TOKEN, sublinear_tf=True)
        self.D = self.vec.fit_transform(docs)
        return self
    def score(self, queries):
        return (self.vec.transform(queries) @ self.D.T).toarray()

class SupervisedLR:
    """Fine-tuned-analogue: TF-IDF + multinomial logistic regression trained on tool-labelled queries.
    Tools with no training queries can never be predicted (score = -1 < any probability)."""
    name = "TFIDF+LR"
    def __init__(self, C=10.0, max_iter=300, seed=0, shuffle_labels=False):
        self.C, self.max_iter, self.seed, self.shuffle = C, max_iter, seed, shuffle_labels
    def fit(self, lib_tools, desc, train_df):
        tr = train_df[train_df.Tool.isin(lib_tools)]
        y = tr.Tool.values.copy()
        if self.shuffle: y = np.random.default_rng(self.seed).permutation(y)
        self.vec = TfidfVectorizer(stop_words="english", token_pattern=TOKEN, sublinear_tf=True)
        X = self.vec.fit_transform(tr.Query)
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X, y)
        self.lib = list(lib_tools); pos = {t: i for i, t in enumerate(self.clf.classes_)}
        self.col = np.array([pos.get(t, -1) for t in self.lib])
        return self
    def score(self, queries):
        P = self.clf.predict_proba(self.vec.transform(queries))
        out = np.full((P.shape[0], len(self.lib)), -1.0)
        m = self.col >= 0
        out[:, m] = P[:, self.col[m]]
        return out

class RandomBaseline:
    name = "Random"
    def __init__(self, seed=0): self.seed = seed
    def fit(self, lib_tools, desc, train_df=None): self.n = len(lib_tools); return self
    def score(self, queries): return np.random.default_rng(self.seed).random((len(queries), self.n))
