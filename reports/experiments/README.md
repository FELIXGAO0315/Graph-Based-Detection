# Experiment tables (generated)

Every file here is written by `meridian-exp`. Do not edit by hand; regenerate instead:

```bash
meridian-exp all          # ladder, fixed, ablation, stability, perturb, loto, embedding
meridian-exp ladder       # or a single experiment
```

| file | experiment |
|---|---|
| `ladder` | complete-pipeline representation ladder A–D |
| `fixed` | fixed candidate set, same scorers, features A–D |
| `ablation` | one relation type removed from the input |
| `stability` | detector thresholds, scorer threshold, entity resolution, case assembly, random seeds |
| `perturb` | address typos and dropped registry name matches, 3 seeds |
| `loto` | leave one type out, with and without open-set detectors |
| `embedding`, `embedding_linkpred` | explicit features vs graph embeddings |

All numbers are on the starter labels with the development scorer, so they describe the
development set only. Interpretation is in `../results.md`.
