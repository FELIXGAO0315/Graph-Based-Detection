# Graph-Based-Detection-of-Anomalous-Third-Party-Relationship-Patterns
GSK

Finds third parties and groups of third parties worth a compliance review in the Meridian
Biopharma extract, from the relationships between them: shared addresses and contacts, sponsors
and approvers, timing, institution links and beneficial ownership. The output is a ranked list of
review cases in the sponsor's submission format, each with traceable evidence.

The source tables are first turned into one time-aware heterogeneous graph (vendors, addresses,
contacts, engagements, employees, institutions, registry entities and persons, with typed and
dated edges). Detection then reads only that graph: each detector looks for a pattern of typed
paths (meta-paths) and cites the graph edges behind every case.

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
meridian-exp scale                         # run time on 20 copies of the data
```

Each run writes to `artifacts/runs/<timestamp>/` (or `--out <dir>`):
- `submission.csv` in the template format, and `ranked_cases.csv` with extra columns;
- `cases/`: analyst case packages for the top 10 cases (`--packages N` to change);
- `graph/`: the heterogeneous graph the detectors ran on (`nodes.parquet`, `edges.parquet`,
  `activity.parquet` with monthly payments per engagement) and `graph.json` with its content hash;
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
  graph.py                   builds the heterogeneous graph: typed node and edge tables, activity, content hash
  graph_views.py             the tables detectors read, rebuilt from the graph (round-trip tested)
  graph_ops.py               typed edges and meta-path instances over the graph
  context.py                 per-time-point views shared by detectors, built from the graph
  detectors/                 one module per family of typologies, policy.py, open_set.py, record.py;
                             evidence.py looks up the edges each finding cites
  cases.py                   merging candidates into cases, emergence and active periods
  ranking.py                 calibration, type interleaving, tag selection, submission table
  calibrate.py               detector hit rates from labels (evaluation side only)
  scorer.py                  development scorer and candidate coverage (only module that reads labels)
  explain.py                 analyst case packages
  features.py                explicit record and case features for the ladder
  embeddings.py              optional: heterogeneous GraphSAGE embeddings
  experiments.py             ladder, ablation, stability, open-set, embedding and scale experiments (meridian-exp)
  run.py                     command-line entry point (meridian)
tests/                       pytest checks
reports/results.md           current results, findings and design decisions
reports/experiments/         generated experiment tables
artifacts/runs/              run outputs (git-ignored)
student_bundle/              sponsor data, read-only
```
