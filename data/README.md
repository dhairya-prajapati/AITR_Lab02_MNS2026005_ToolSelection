# Data

Source: MetaTool benchmark (Huang et al., ICLR 2024), official repository https://github.com/HowieHwong/MetaTool, commit `35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb`, MIT licence (copy in `METATOOL_LICENSE_MIT.txt`).

Files copied unmodified from the repository (small, MIT-licensed, so redistributed here for convenience):

| Local file | Repository path |
|---|---|
| `metatool_all_clean_data.csv` | `dataset/data/all_clean_data.csv` (20,614 single-tool queries, columns `Query`,`Tool`) |
| `metatool_plugin_des.json` | `dataset/plugin_des.json` (199 tool descriptions) |

To re-download: `git clone https://github.com/HowieHwong/MetaTool` and copy the two files above. Only these data files are used; MetaTool's LLM prompts/evaluation code are not.
Cleaning is done in `src/data.py` (see paper Table 1). No private or restricted data.
