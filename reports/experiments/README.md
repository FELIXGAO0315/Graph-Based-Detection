# Experiment tables (generated)

Every file here is written by `meridian-exp`. Do not edit by hand; regenerate instead:

```bash
meridian-exp all          # ladder, fixed, ablation, stability, perturb, loto, embedding
meridian-exp ladder       # or a single experiment
meridian-exp scale        # run time and detector output on 20 copies of the data (not part of all)
```

| file | experiment |
|---|---|
| `ladder` | complete-pipeline representation ladder A–D |
| `fixed` | fixed candidate set, same scorers, features A–D |
| `ablation` | one family of edges removed from the graph before detection |
| `stability` | detector thresholds and threshold quantiles, scorer threshold, entity resolution, case assembly and ranking choices, random seeds |
| `perturb` | address typos and dropped registry name matches, 3 seeds |
| `loto` | leave one type out, with and without open-set detectors |
| `embedding`, `embedding_linkpred` | explicit features vs graph embeddings |
| `scale` | the starter data and 20 independent copies of it: run time per stage, candidates per detector |

Apart from `scale`, all numbers are on the starter labels with the development scorer, so they describe the
development set only. Interpretation is in `../results.md`.
