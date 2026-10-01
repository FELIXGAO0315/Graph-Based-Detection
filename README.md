# Graph-Based-Detection-of-Anomalous-Third-Party-Relationship-Patterns
GSK

Finds third parties and groups of third parties worth a compliance review in the Meridian
Biopharma extract, from the relationships between them: shared addresses and contacts, sponsors
and approvers, timing, institution links and beneficial ownership. The output is a ranked list of
review cases in the sponsor's submission format, each with traceable evidence.

## Quick start

Python 3.12.

```bash
pip install -r requirements.txt && pip install -e .
pip install -r requirements-gnn.txt        # optional: graph embeddings

meridian                                   # data -> checks -> detectors -> cases -> ranked submission -> local score
meridian --set features.ownership=false    # ownership layer off
meridian --set window='["2025-07-01","2026-06-30"]'   # keep only cases emerging in the test period
pytest                                     # time, resolution, scorer, pipeline (and embedding) checks
meridian-exp all                           # all experiments -> reports/experiments/
```

Each run writes to `artifacts/runs/<timestamp>/` (or `--out <dir>`):
- `submission.csv` in the template format, and `ranked_cases.csv` with extra columns;
- `cases/`: analyst case packages for the top 10 cases (`--packages N` to change);
- `cases.jsonl` (evidence per case), `checks.md` (data checks) and the entity-resolution lists;
- `run.json`: config, input hashes, versions and local scores.

Scores on the starter labels come from our own development scorer (`src/meridian/scorer.py`) and
describe the development set only.

See `reports/results.md` for current results and findings.

## Layout

```text
configs/default.yaml         run configuration: snapshot, window, detector parameters, ranking
src/meridian/
  data.py                    loading, registry parsing, as_of(t)
  checks.py                  data checks (contract, keys, dates, Dossier Appendix B tests)
  resolution.py              normalisation, registry matching, duplicate records
  context.py                 per-time-point views shared by detectors (entities, categories, ownership paths)
  detectors/                 one module per family of typologies, policy.py, open_set.py, record.py
  cases.py                   merging candidates into cases, emergence and active periods
  ranking.py                 calibration, type interleaving, tag selection, submission table
  calibrate.py               detector hit rates from labels (evaluation side only)
  scorer.py                  development scorer and candidate coverage (only module that reads labels)
  explain.py                 analyst case packages
  features.py                explicit record and case features for the ladder
  embeddings.py              optional: heterogeneous GraphSAGE embeddings
  experiments.py             ladder, ablation, stability, open-set and embedding experiments (meridian-exp)
  run.py                     command-line entry point (meridian)
tests/                       pytest checks
reports/results.md           current results, findings and design decisions
reports/experiments/         generated experiment tables
artifacts/runs/              run outputs (git-ignored)
student_bundle/              sponsor data, read-only
```
