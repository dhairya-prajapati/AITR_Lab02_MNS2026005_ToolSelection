"""Environment sanity test. Expected: versions printed + 'ENV OK'."""
import sys, platform, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy, pandas, sklearn, scipy, matplotlib, rank_bm25
from src.data import load_clean
print("python", sys.version.split()[0], "|", platform.platform())
for m in (numpy, pandas, sklearn, scipy, matplotlib):
    print(f"{m.__name__:12s}", m.__version__)
print("rank_bm25   ", getattr(rank_bm25, "__version__", "0.2.2 (no __version__ attr)"))
df, desc, rep = load_clean()
print("data report:", json.dumps(rep))
assert rep["n_tools"] == 199 and rep["all_tools_have_description"]
print("ENV OK")
