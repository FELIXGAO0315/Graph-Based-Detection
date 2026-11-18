# Graph-Based Detection of Anomalous Third-Party Relationship Patterns

Duke MIDS capstone with GSK. The data is the synthetic Meridian Biopharma bundle: third parties, their
engagements, approvals, addresses, owners and monthly activity. The task is to submit a **ranked list of
review cases** for the TEST period (2025-07-01 – 2026-06-30). Each case is a group of vendor records, the
typologies it shows, a score and a reason. The sponsor scores the list blind against planted cases
(Dossier §8–9).

The work is in three layers. Layers 1 and 2 are the preparation: a vendor-level baseline and the
relationship logic. Layer 3 is the graph model that produces the submission.

| layer | notebook | role | status |
|---|---|---|---|
| 1 | `layer1_baseline.ipynb` | vendor-level baseline: each vendor's own records only | done, frozen in `results/layer1_frozen/` |
| 2 | `layer2_relationship_logic.ipynb` | relationship features per typology, new candidate typologies, case queue | done, frozen in `results/layer2_frozen/` |
| 3 | `layer3_graph_detection.ipynb` | **graph model**: heterogeneous graph, typology detectors, cases, ranking | v1.0–v1.2 submitted and scored; v1.3 is the current notebook |

**Layer 3 versions.** The notebook always holds the current version. Earlier versions are kept as a git commit
(the code), `results/graph/<version>/` (the results) and `output/<version>/` (the deliverable).

| version | code | what changed | starter AP (TRAINING / VALIDATION / TEST) | deliverable | status |
|---|---|---|---|---|---|
| v1.0 | commit `67553b4` | first version | 0.868 / 0.940 / 0.857 | `output/v1.0/submission_TEST.csv` | submitted; client score 2026-11 |
| v1.1 | commit `9fbf4fc` | oversized cases split, policy findings in their own list, ranking reworked | 0.864 / 0.929 / 0.869 | `output/v1.1/` | submitted; client score 2026-11 (P@10 0.80, tag accuracy 0.74) |
| v1.2 | commit `1b276d6` | public-institution expert clusters detected, tags that fit the case, weak and duplicate cases left out | 0.864 / 0.929 / 0.856 | `output/v1.2/` | submitted; client score 2026-11 (P@10 0.70, tag accuracy 0.79) |
| v1.3 | current notebook | developed on the main-data validation labels: period filter (covering findings, standing structures), uncoupled approver groups and chance spike groups ranked low, cross-border corridors, sponsor onboarding bursts, expert-owner benign note, expert cluster removed; main-data validation AP 0.308 → 0.419 | 0.893 / 0.934 / 0.881 | `output/v1.3/` | built 2026-11-18, not submitted |

---

## Repository layout

```
Graph-Based-Detection/
├── README.md
├── layer1_baseline.ipynb              Layer 1
├── layer2_relationship_logic.ipynb    Layer 2
├── layer3_graph_detection.ipynb       Layer 3, graph model, current version v1.3 (self-contained: all method code is in the notebook)
├── student_bundle/                    starter data: 1,002 vendor records + 33 labelled cases (labelled_cases_TRAINING_SET)
├── Full Dataset/                      main data: 21,260 vendor records; validation_student_safe.csv: the 43 labelled
│                                      VALIDATION cases (released 2026-11); TRAINING and TEST are scored blind by the sponsor
├── scripts/
│   ├── layer1/layer1_models.py        Layer 1 models
│   ├── layer2/                        graph_build.py, layer2_features.py, peer_groups.py
│   └── layer3/                        layer3_cases.py, layer3_detectors.py, layer3_stability.py,
│                                      compare_versions.py (two versions' submissions side by side, no labels needed)
├── features/                          Layer 1 feature tables as of each period end
├── results/
│   ├── L1_*.csv, L2_*.csv, *.png      Layer 1 and Layer 2 result tables and figures
│   ├── layer1_frozen/                 Layer 1 outputs reused by Layers 2 and 3 (periods, scores, config)
│   ├── layer2_frozen/                 Layer 2 features, relation pairs, risk-score weights
│   └── graph/
│       ├── v1.0/                      Layer 3 v1.0, complete archive (see "Layer 3 v1.0" below)
│       ├── v1.1/                      Layer 3 v1.1, everything the notebook writes, plus ablation/ (see "Layer 3 v1.1")
│       ├── v1.2/                      Layer 3 v1.2, everything the notebook writes (see "Layer 3 v1.2")
│       ├── v1.3/                      Layer 3 v1.3, everything the notebook writes, incl. the validation analysis (see "Layer 3 v1.3")
│       └── *.csv, graph/, figures/    an earlier v1.0 run, superseded (see the note under "Archive")
└── output/                            the deliverables, one folder per version, with the client's score once it is back
    ├── v1.0/submission_TEST.csv       first submission, scored 2026-11
    ├── v1.1/submission_TEST.csv       second submission, with policy_findings_TEST.csv and client_score_2026-11.csv
    ├── v1.2/submission_TEST.csv       third submission, with policy_findings_TEST.csv and client_score_2026-11.csv
    └── v1.3/submission_TEST.csv       v1.3 candidate, with policy_findings_TEST.csv
```

## Data and evaluation protocol

| | starter (`student_bundle/`) | main (`Full Dataset/`) |
|---|---|---|
| vendor records | 1,002 | 21,260 |
| labels | 33 planted cases (TRAINING 16, VALIDATION 7, TEST 10) | 43 VALIDATION cases, released 2026-11 (`validation_student_safe.csv`); TRAINING and TEST none (TEST is scored blind) |
| use | build the method, measure it, choose between variants | produce the submission; from v1.3, develop and choose on the VALIDATION labels |

Note that the main data reuses the starter's record ids for *different* records, so starter labels must never
be scored against the main data.

**Periods (Dossier §8.1).** A case belongs to the period in which it emerges:

| period | cases emerging | the graph is built as of |
|---|---|---|
| TRAINING | up to 2024-06-30 | 2024-06-30 |
| VALIDATION | 2024-07-01 – 2025-06-30 | 2025-06-30 |
| TEST | 2025-07-01 – 2026-06-30 | 2026-06-30 |

**How one period is run.** Take the TEST submission as an example:
1. Build the graph from **all data known by the period end**, 2026-06-30. A row counts once it is observed by
   that date, and end dates after it are blanked (`as_of(t)`). The full history is used, because many
   patterns (ownership chains, shared addresses) only show over time.
2. Run every detector on that graph and merge their findings into cases.
3. Keep only the cases that **emerged in the period**, plus older cases that **grew in it**, meaning one of
   their vendor entities was first onboarded in the period. From v1.3 also older cases with a covering finding that
   emerged in the period (ranked at half weight), and, behind all of them, older structures that still stand at the
   period start (see "Layer 3 v1.3").
4. Rank the remaining cases. That ranked list is the submission.

Labels are never read by detection, case assembly or ranking. They are read only by the scorer.

**Metrics (Dossier §9.1).** Matching goes down the ranking one-to-one: a submitted case matches the unmatched
planted case it overlaps most, if their IoU is ≥ 0.5.
- **AP** (headline): average precision over the ranked list.
- **P@k**: matched cases among the first k.
- **R@k**: planted cases matched within the first k.
- For the matched cases: **mean IoU**, **member recall**, **exact recovery** (the member sets are identical)
  and **tag accuracy**.
- **fp_top50**: false positives among the first 50 cases.

On the starter data a period has only 7–16 planted cases, so P@k is capped. With 7 VALIDATION cases, P@10 can
be at most 0.7 and P@25 at most 0.28.

---

## Layer 1: vendor-level baseline

Each vendor is scored from its own records only: its fields, its own engagements, payments, approvals,
services and addresses, plus reference tables. Six models are compared: linear, logistic, decision tree,
random forest, gradient boosting and Isolation Forest. Each is trained once on TRAINING, and the fitted model
then scores VALIDATION and TEST. Variables are selected by four penalised models voting.

| TEST (10 cases), mean of 5 seeds | AP_case | P@10 | R@100 | cases found @100 |
|---|---|---|---|---|
| random ranking | 0.073 | 0.07 | 0.11 | |
| Random forest (best on TEST) | 0.176 | 0.26 | 0.33 | 3 / 10 |
| Gradient boosting (main baseline) | 0.153 | 0.08 | 0.26 | 3 / 10 |

**What it shows.** A vendor's own records find few cases. The cases it misses (approver routes, ownership
chains, bridges) are defined by relationships *between* vendors. Full tables are in the notebook and in
`results/L1_*.csv`.

## Layer 2: relationship logic

Every one of the 14 typologies gets relationship variables and a rarity signal (normalised self-information).
Seven candidate new typologies are tested with an add/drop ablation and a five-criterion scorecard. Vendor
models (RF, GB and a risk score) and a case-level review queue are compared under the same periods.

| model (vendor queue) | VAL AP | TEST AP | TEST cases @100 | all 33 @100 |
|---|---|---|---|---|
| L1 RF | 0.119 | 0.153 | 3 / 10 | 10 / 33 |
| L2 GB | 0.574 | 0.525 | 6 / 10 | 20 / 33 |
| **L2 risk score (14+3)**, chosen on VALIDATION | 0.514 | **0.595** | **8 / 10** | **24 / 33** |

**What it shows.** Relationship information triples TEST AP over Layer 1. Its ceiling is the two
approver-route TEST cases (CASE-0006, CASE-0007), which no vendor-level model ranks. Layer 2 also proposes the
candidate typologies that Layer 3 tests as variants: `FMV_CAP_BREACH`, `REONBOARDED_AFTER_TERMINATION`,
`PUBLIC_INSTITUTION_EXPERT_CLUSTER` and `PAYMENT_AFTER_TERMINATION`. Results are in `results/L2_*.csv`.

---

## Layer 3 v1.0: graph model

Notebook: `layer3_graph_detection.ipynb` at commit `67553b4` (`git show 67553b4:layer3_graph_detection.ipynb`).
It is self-contained: every method module is a `%%module` cell of an in-notebook package, `meridian`.
Method hash: `0f4919409565`.

### Pipeline

```
source tables ─► as_of(t) view ─► normalisation + entity resolution ─► heterogeneous graph
   ─► typology detectors (meta-path patterns) ─► candidates ─► case assembly ─► window filter ─► ranking ─► submission
```

- **Graph** (starter, at 2026-06-30): 10,089 nodes of 12 types and 20,828 edges of 20 relation types, plus
  34,467 activity rows. The node types include VendorRecord, ResolvedEntity, Address, ContactPoint, Employee,
  Engagement, LegalEntity, NaturalPerson, HealthcareInstitution and ServiceCategory. Edges keep their valid
  time, observation time and source.
- **Entity resolution**: LEI, registry number, name and jurisdiction, fuzzy name. On the starter data,
  1,002 records resolve to 980 entities.
- **Peer groups**: every statistic is compared with the most specific peer group that has at least 5
  members, backing off to coarser groups. Documented benign explanations (registered agents, central roles,
  business events) lower a case's priority but never remove the signal.

### Typologies and detectors

The main model (variant **14**) uses the **14 typology tags** of the practice labels: the dossier's 13
typologies, with approver concentration and sponsor–approver coupling tagged separately. Two extra labels
are used: `OPEN_SET` for patterns outside the 14, and `POLICY_VIOLATION` for single-record policy findings.

| tag | detector(s) |
|---|---|
| SHARED_ADDRESS | `shared_address` |
| SPONSOR_CONCENTRATION | `sponsor_concentration` |
| APPROVER_CONCENTRATION | `approver_star`, `sponsor_approver_pair` (when outside authority) |
| SPONSOR_APPROVER_COUPLING | `sponsor_approver_pair` |
| DENSE_MULTI_RELATION_CLUSTER | `dense_cluster` |
| RAPID_ONBOARDING_LINKED | `rapid_onboarding` |
| BRIDGE_ENTITY | `bridge` |
| GEOGRAPHY_LINKED_CONCENTRATION | `geography` |
| ACTIVITY_TIMING_CONCENTRATION | `sync_spikes`, `post_termination` |
| COMMON_BENEFICIAL_OWNER | `common_owner` |
| LAYERED_COMPLEX_OWNERSHIP | `layered_ownership` |
| OWNERSHIP_OPERATIONAL_OVERLAP | `ownership_overlap` (emitted by `common_owner`) |
| OPAQUE_JURISDICTION_ROUTING | `opaque_routing` |
| OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | `ownership_change` |
| OPEN_SET | `rare_sharing_community`, `entity_change_point`, `duplicate_paid`, `entity_outlier` |
| POLICY_VIOLATION | `fmv_cap` (expert fee above the FMV cap), `compliance_review` (missing required review) |

**Variants (section 7).** These add the Layer 2 candidate typologies:

| variant | added typologies |
|---|---|
| **14** (chosen) | none |
| 14+1 | FMV_CAP_BREACH |
| 14+2 | FMV_CAP_BREACH, REONBOARDED_AFTER_TERMINATION |
| 14+3 | FMV_CAP_BREACH, PUBLIC_INSTITUTION_EXPERT_CLUSTER, PAYMENT_AFTER_TERMINATION |

The variant is chosen on VALIDATION (AP, then mean IoU). On the starter data VALIDATION does not separate
the four variants (AP 0.940 for all), so the first listed, **14**, is kept.

### Case assembly and ranking (v1.0 settings)

- **Merging.** Candidates from different detectors merge into one case when they describe the same group at
  about the same time. Two candidates with member IoU ≥ 0.8 merge outright. Groups with IoU ≥ 0.5 merge as a
  union, but only while the union stays within **1.5×** the lead candidate. Emergence dates must be within
  365 days.
- **Case score.** Each case takes its best candidate plus half the noisy-OR gain from corroborating
  detectors, counting each evidence edge once. Each candidate's value is: prior × within-detector percentile
  × benign weight. The benign weight is **0.7 per benign explanation, floor 0.3**.
- **Type interleaving.** A case's score decays by 0.85 for each case of the same primary tag already ranked,
  and is halved when the case nearly duplicates one already ranked.
- **Tags.** A case keeps the tags whose evidence is within 85% of its strongest.
- **What is submitted.** Every case assembled, with no truncation. Policy findings stay in the main queue.

### v1.0 results on the starter data (all cases submitted)

| period | planted | found | submitted | **AP** | P@10 | P@25 | P@50 | mean IoU | member recall | exact | tag exact | FP in top 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAINING | 16 | 16 | 131 | **0.868** | 0.80 | 0.56 | 0.32 | 0.984 | 0.984 | 0.938 | 0.750 | 34 |
| VALIDATION | 7 | 7 | 32 | **0.940** | 0.60 | 0.28 | 0.14 | 1.000 | 1.000 | 1.000 | 0.714 | 25 |
| TEST | 10 | 10 | 53 | **0.857** | 0.80 | 0.40 | 0.20 | 0.864 | 0.906 | 0.300 | 0.400 | 40 |
| all 33 at the snapshot | 33 | 33 | 179 | **0.928** | 1.00 | 0.88 | 0.66 | 0.951 | 0.964 | 0.758 | 0.667 | 17 |

Every labelled case is found in the top 50 of its own period (R@50 = 1.0), and the first hit is at rank 1 in
every period.

**AP by matching threshold** (the official threshold is not published):

| | TRAINING | VALIDATION | TEST | all 33 |
|---|---|---|---|---|
| IoU ≥ 0.3 | 0.932 | 0.940 | 0.857 | 0.946 |
| **IoU ≥ 0.5** | **0.868** | **0.940** | **0.857** | **0.928** |
| IoU > 0.5 (Layers 1–2 rule) | 0.868 | 0.940 | 0.779 | 0.858 |
| IoU ≥ 0.7 | 0.868 | 0.940 | 0.635 | 0.822 |

**Variants**, AP by period:

| variant | TRAINING | VALIDATION | TEST | all 33 |
|---|---|---|---|---|
| **14** | 0.868 | 0.940 | 0.857 | 0.928 |
| 14+1 | 0.868 | 0.940 | 0.857 | 0.928 |
| 14+2 | 0.817 | 0.940 | 0.804 | 0.883 |
| 14+3 | 0.868 | 0.940 | 0.865 | 0.904 |

### v1.0 on the main data and the client's blind score

The method was run unchanged on `Full Dataset/` with variant 14. One variant takes about 22 minutes.

| period | candidates | cases submitted | OPEN_SET cases |
|---|---|---|---|
| TRAINING | 3,364 | 2,706 | 1,130 |
| VALIDATION | 3,902 | 782 | 342 |
| **TEST** (deliverable) | 4,770 | **1,130** | 436 |

**Deliverable:** `output/v1.0/submission_TEST.csv`, the 1,130 TEST cases of variant 14 in the sponsor's template.

**Client blind score** (sponsor e-mail, 2026-11; recall withheld to avoid test leakage):

| mean IoU | member recall | exact recovery | tag accuracy | P@10 | P@25 | P@50 |
|---|---|---|---|---|---|---|
| ≈ 0.83 | ≈ 0.91 | just over half | ≈ 0.79 | 0.50 | 0.48 | 0.38 |

**Reading of the score.**
- **Case quality holds up on the main data**, but **precision is the weak point**: about 19 of the first 50
  cases are planted cases.
- **Oversized cases.** Nine of the first 50 cases have more than 15 members, up to 274. These are whole
  sponsor or approver portfolios, while the largest planted starter case has 13 members.
- **Policy findings in the main queue.** Seven of the first 50 cases carry `POLICY_VIOLATION`. Five of them
  are policy findings alone (four of one record, one of two records), which the dossier scores separately
  from the typologies.
- **Cases slightly too large.** Member recall above IoU means the matched cases carry extra records.
- **Missed patterns.** The sponsor reports that a meaningful number of planted cases were missed.

### Archive

`results/graph/v1.0/` holds the complete v1.0 output, produced by re-running the committed notebook on the
starter data:
- `metrics_by_period.csv`, `metrics_all.csv`, `ap_by_threshold.csv`, `case_landing.csv`
- `layer2_variants/metrics.csv`: every variant × period, for all cases and for the first 100
- `cases/starter/`: every ranked case list
- `graph/`: the graph export
- `full_dataset/client_score_2026-11.csv`: the client's score

The v1.0 deliverable itself is `output/v1.0/submission_TEST.csv`.

Note: the CSV files directly under `results/graph/` (`metrics_by_period.csv`, `layer2_variants/metrics.csv`,
…) come from an earlier run that kept only the first 100 cases, so their `n_submitted` and all-33 AP (0.933)
differ. Use `results/graph/v1.0/`.

---

## Layer 3 v1.1: after the client's score

Notebook: `layer3_graph_detection.ipynb` at commit `9fbf4fc` (`git show 9fbf4fc:layer3_graph_detection.ipynb`).
Method hash: `8b47cada7d5a`. Submitted as the second submission; the client's score is under "Layer 3 v1.2".
Variant: **14**, chosen on
VALIDATION (AP 0.929, against 0.924 for 14+1, 14+2 and 14+3). The detectors, typologies and case periods are
those of v1.0; only the settings in the table below change, each marked `v1.1` in section 0.1 of the notebook.

### What changed and why

| client finding on v1.0 | v1.1 change | setting (v1.0 → v1.1) |
|---|---|---|
| Oversized cases: 9 of the first 50 have more than 15 members, up to 274 | an approver case with more than 15 records is split into one case per sponsor (sponsor groups of at least 3 records) | `approver_star.max_members` none → 15, `min_split_records` 3 |
| Policy findings in the main queue: 7 of the first 50 | single-record `POLICY_VIOLATION` findings leave the main queue and are ranked as their own list, `policy_findings_TEST.csv` | `ranking.include_policy` true → false, `policy_list` true |
| Precision: about 19 of the first 50 are planted cases | a case with more than 15 members is down-weighted by 15 / members | `ranking.max_size` none → 15, `size_power` 1 |
| | a documented benign explanation counts for more | `benign_per_item` 0.7 → 0.5, `benign_floor` 0.3 → 0.15 |
| | lighter type interleaving: the strongest cases rank first even when they share a tag | `decay` 0.85 → 0.95 |
| Cases slightly too large (member recall above IoU) | tested and **not adopted**: a tighter merge limit costs a labelled case its corroboration (below) | `cases.max_growth` stays 1.5 |
| "Getting the reason right matters as much as getting the records right" | a sponsor group split from an approver case leads its reason with the sponsor and its own vendors and engagements, since the submission's `reason_codes` keeps each reason's first clause | code only (`detect_stars`); no setting |

### How the settings were chosen

Each change was first run on its own against v1.0 on the starter data (AP, IoU ≥ 0.5, all cases submitted).
The choice reads TRAINING and VALIDATION; TEST is reported only. Full table:
`results/graph/v1.1/ablation/ablation_starter.csv`, where each labelled case lands:
`ablation_case_landing.csv`.

| setting | TRAINING | VALIDATION | TEST |
|---|---|---|---|
| v1.0 | 0.868 | 0.940 | 0.857 |
| v1.0 + split oversized cases | 0.844 | 0.940 | 0.863 |
| v1.0 + policy list | 0.868 | 0.948 | 0.857 |
| v1.0 + tighter merge limit (`max_growth` 1.25) | **0.783** | **0.898** | 0.857 |
| v1.0 + size and benign weights | 0.875 | 0.924 | 0.863 |
| all four | 0.788 | 0.888 | 0.863 |
| all but the tighter merge limit | 0.853 | 0.929 | 0.863 |
| **all but the tighter merge limit, decay 0.95 (= v1.1)** | **0.864** | **0.929** | **0.869** |

- **The tighter merge limit was dropped.** It is the only change that loses a labelled case's rank by much:
  CASE-0021 (TRAINING, common beneficial owner) can no longer merge with the detector that corroborates it,
  and falls from rank 3 to 20. TEST mean IoU also drops (0.864 → 0.844).
- **The other changes are neutral on the starter data**, within one or two ranks of single cases. The starter
  data has little of what they fix: one or two cases above 15 members in each period's top 50, and at most one
  policy finding.

### v1.1 against v1.0 on the starter data

| period | version | submitted | **AP** | P@10 | P@25 | P@50 | mean IoU | member recall | exact | tag exact | FP in top 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAINING | v1.0 | 131 | **0.868** | 0.80 | 0.56 | 0.32 | 0.984 | 0.984 | 0.938 | 0.750 | 34 |
| | v1.1 | 135 | **0.864** | **0.90** | **0.60** | 0.32 | 0.984 | 0.984 | 0.938 | 0.750 | 34 |
| VALIDATION | v1.0 | 32 | **0.940** | 0.60 | 0.28 | 0.14 | 1.000 | 1.000 | 1.000 | 0.714 | 25 |
| | v1.1 | 32 | **0.929** | 0.60 | 0.28 | 0.14 | 1.000 | 1.000 | 1.000 | 0.714 | 25 |
| TEST | v1.0 | 53 | **0.857** | 0.80 | 0.40 | 0.20 | 0.864 | 0.906 | 0.300 | 0.400 | 40 |
| | v1.1 | 57 | **0.869** | 0.80 | 0.40 | 0.20 | 0.864 | 0.906 | 0.300 | 0.400 | 40 |
| all 33 at the snapshot | v1.0 | 179 | **0.928** | 1.00 | 0.88 | 0.66 | 0.951 | 0.964 | 0.758 | 0.667 | 17 |
| | v1.1 | 186 | **0.935** | 1.00 | 0.92 | 0.62 | 0.951 | 0.964 | 0.758 | 0.667 | 19 |

Every labelled case is still found in its own period's top 50 (16/16, 7/7, 10/10), first hit at rank 1. Case
boundaries do not change. At the snapshot two cases now fall just outside the top 50 (CASE-0011 at 51,
CASE-0003 at 63), so R@50 there is 0.939 (v1.0: 1.0).

**Where labelled cases moved** (rank in their own period; cases not listed did not move):

| period | case | typology | v1.0 | v1.1 |
|---|---|---|---|---|
| TRAINING | CASE-0004 | SPONSOR_CONCENTRATION | 9 | 6 |
| | CASE-0020 | COMMON_BENEFICIAL_OWNER | 29 | 19 |
| | CASE-0024 | OWNERSHIP_OPERATIONAL_OVERLAP | 31 | 29 |
| | CASE-0022 | LAYERED_COMPLEX_OWNERSHIP | 11 | 10 |
| | CASE-0029 | OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | 6 | 7 |
| | CASE-0027 | OPAQUE_JURISDICTION_ROUTING | 7 | 9 |
| | CASE-0030 | OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | 14 | 16 |
| | CASE-0018 | ACTIVITY_TIMING_CONCENTRATION | 19 | 21 |
| | CASE-0015 | GEOGRAPHY_LINKED_CONCENTRATION | 13 | 18 |
| VALIDATION | CASE-0025 | OWNERSHIP_OPERATIONAL_OVERLAP | 6 | 3 |
| | CASE-0013 | BRIDGE_ENTITY | 3 | 6 |
| | CASE-0011 | RAPID_ONBOARDING_LINKED | 12 | 14 |
| TEST | CASE-0006 | APPROVER_CONCENTRATION | 10 | 8 |
| | CASE-0007 | APPROVER_CONCENTRATION | 14 | 11 |
| | CASE-0014 | GEOGRAPHY_LINKED_CONCENTRATION | 8 | 10 |
| | CASE-0012 | BRIDGE_ENTITY | 11 | 12 |

**AP by matching threshold:**

| | TRAINING | VALIDATION | TEST | all 33 |
|---|---|---|---|---|
| IoU ≥ 0.3 | 0.892 | 0.929 | 0.869 | 0.943 |
| **IoU ≥ 0.5** | **0.864** | **0.929** | **0.869** | **0.935** |
| IoU > 0.5 | 0.864 | 0.929 | 0.791 | 0.872 |
| IoU ≥ 0.7 | 0.864 | 0.929 | 0.646 | 0.829 |

**Variants**, AP by period (all cases submitted):

| variant | TRAINING | VALIDATION | TEST | all 33 |
|---|---|---|---|---|
| **14** | 0.864 | **0.929** | 0.869 | 0.935 |
| 14+1 | 0.864 | 0.924 | 0.869 | 0.934 |
| 14+2 | 0.814 | 0.924 | 0.793 | 0.888 |
| 14+3 | 0.864 | 0.924 | 0.881 | 0.916 |

**Reading.** On the starter data v1.1 neither improves nor worsens v1.0 beyond what 7–16 cases a period can
resolve: TRAINING −0.004, VALIDATION −0.011 (three cases reordered), TEST +0.012. The starter data cannot
decide between the versions. The changes are aimed at the main data.

### v1.1 on the main data

Variant 14 on `Full Dataset/`, about 16 minutes. The main data has no labels, so the comparison with v1.0 is
of what each version puts at the top of its TEST queue (`scripts/layer3/compare_versions.py`, saved in
`results/graph/v1.1/full_dataset/compare_with_v1.0.csv`).

| period | candidates | cases submitted | OPEN_SET cases | policy findings (own list) |
|---|---|---|---|---|
| TRAINING | 3,760 | 3,078 | 1,195 | 21 |
| VALIDATION | 4,429 | 956 | 368 | 7 |
| **TEST** (deliverable) | 5,477 | **1,392** | 460 | **15** |

| TEST queue | version | mean members | largest case | cases > 15 members | cases with POLICY_VIOLATION | shared with the other version (IoU ≥ 0.5) |
|---|---|---|---|---|---|---|
| first 10 | v1.0 | 8.6 | 53 | 1 | 2 | 6 |
| | v1.1 | 3.7 | 9 | **0** | **0** | 6 |
| first 25 | v1.0 | 11.8 | 83 | 4 | 3 | 15 |
| | v1.1 | 6.3 | 14 | **0** | **0** | 15 |
| first 50 | v1.0 | 19.4 | 274 | 9 | 7 | 31 |
| | v1.1 | 5.9 | 15 | **0** | **0** | 31 |
| first 100 | v1.0 | 14.7 | 274 | 22 | 12 | 57 |
| | v1.1 | 6.1 | 16 | 1 | 0 | 57 |
| all | v1.0 | 9.6 | 368 | 88 | 17 | |
| | v1.1 | 6.0 | 77 | 57 | 0 | |

**Deliverable:** `output/v1.1/submission_TEST.csv` (1,392 TEST cases, variant 14) and
`output/v1.1/policy_findings_TEST.csv` (15 single-record findings: 5 expert fees above the FMV cap, 10
engagements without a required compliance review).

**Reading.**
- **Both structural problems the client named are gone from the top of the queue.** The first 50 cases hold
  no case above 15 members (v1.0: 9, up to 274) and no policy finding (v1.0: 7).
- **19 of the first 50 cases are new.** 31 match a v1.0 case. Whether the new ones are planted cases cannot be
  told without labels; the client's next score, or the validation-period labels, will tell.
- **The queue is longer.** Splitting approver cases by sponsor gives 1,392 TEST cases (v1.0: 1,130): 353 are
  led by approver concentration (v1.0: 66). Most sit low in the queue (median rank 865), and 6 are in the first
  50 (v1.0: 4).

### Reason text of split cases (fixed 2026-11-03)

The submission's `reason_codes` keeps the first clause of each detector's reason. In the first v1.1 run, a
sponsor group split from an approver case kept the approver's reason ("approver … handled 38 of 105 financial
approvals…"), and the sponsor the case is about appeared only in the full reason. The reason now leads with
the group:

> sponsor EMP-1C0BEF79 (medical_lead) brought 8 vendors (9 engagements, RND in US) to approver EMP-EC77EF37
> (finance_approver_l3), who handled 38 of 105 such financial approvals where ~2 approvers were authorised
> (expected share 19%, surprise 10^-4.4)

Its out-of-scope and below-level counts are those of the group's own engagements. Members, tags, scores and
ranks do not change (checked on every starter period and on the main-data TEST list). On the main data 349 of
the 1,392 TEST cases get the new reason, 7 of them in the first 50 (ranks 6, 14, 26, 33, 40, 41, 47). The
method hash covers the settings only, so it stays `8b47cada7d5a`.

### Archive

`results/graph/v1.1/` has the same layout as `results/graph/v1.0/`, plus:
- `ablation/`: the ablation above
- `cases/starter/policy_<PERIOD>.csv` and `cases/*/policy_<VARIANT>_<PERIOD>.csv`: the policy lists
- `cases/full_dataset/`: every variant and period on the main data
- `full_dataset/submitted.csv`, `full_dataset/compare_with_v1.0.csv`: main-data counts and the comparison above

The v1.1 deliverable and the client's score of it (`client_score_2026-11.csv`) are in `output/v1.1/`.

---

## Layer 3 v1.2: after the client's second score

Notebook: `layer3_graph_detection.ipynb` at commit `1b276d6` (`git show 1b276d6:layer3_graph_detection.ipynb`). Method
hash: `1c7d686e5f30`. Submitted as the third submission; the client's score is under "Layer 3 v1.3". Variant: **14**, chosen on
VALIDATION as before. Case assembly and every detector of v1.1 are unchanged; v1.2 adds one detector and
changes how tags are chosen and which cases are submitted. Each change is marked `v1.2` in the code or in
section 0.1 of the notebook.

### The client's score of v1.1

| metric (TEST, main data) | v1.0 | v1.1 |
|---|---|---|
| P@10 | 0.50 | **0.80** |
| P@25 | 0.48 | **0.60** |
| P@50 | 0.38 | **0.48** |
| mean case IoU of matched cases | 0.83 | above 0.80 |
| mean member recall | 0.91 | about 0.90 |
| exact recovery | just over half | about half |
| typology accuracy on matched cases | 0.79 | **0.74** |
| typologies with zero recall | at least one (not reported then) | at least one, the same blind spot |

Source: the client's e-mail (`output/v1.1/client_score_2026-11.csv`). The client named the
next priorities: keep the ranking gains, broaden detection coverage, improve typology assignment, and reduce
the volume of weak or overlapping cases. The typology with zero recall is not named, to keep the scoring blind.

### What changed and why

| client finding on v1.1 | v1.2 change | setting or code |
|---|---|---|
| A typology has zero recall, in v1.0 too: "a structural blind spot" in candidate generation | a new detector: at least 3 individual experts affiliated with one public-sector institution and engaged by one sponsor, at least 3 of them joining within 90 days; tag `PUBLIC_INSTITUTION_EXPERT_CLUSTER` (see "The blind spot" below) | `params.public_institution` {min_experts 3, window_days 90}; detector `public_institution_cluster` |
| Typology accuracy 0.79 → 0.74; "the right records for the wrong reason" | a tag comes only from a detector whose own members cover the case (member IoU ≥ 0.75); a typology that fits replaces the general typologies it is built from (table below); tags within 0.8 of the strongest | `ranking.tag_fit` 0.75 (new), `tag_ratio` 0.85 → 0.8; `SUBSUMES` in `meridian.ranking` |
| | an approver group split by sponsor (v1.1) is also tagged `SPONSOR_APPROVER_COUPLING` when the sponsor sends at least half of its approved engagements to that approver, the pair detector's rule; in the starter labels every approver case is such a pair | `detect_stars`; `approval.min_sponsor_share` 0.5 |
| | the reason starts with the detector behind the submitted primary tag, and the type interleaving of the ranking uses the submitted primary tag | `short_reason`, `Case.primary_tag` |
| "Reduce the overall volume of weak or overlapping candidate cases" | a near-duplicate of a higher case (member IoU ≥ 0.5) is left out instead of halved; a case whose only evidence is one open-set detector is left out (two open-set detectors that agree are kept); both are listed in `results/graph/v1.2/full_dataset/dropped_TEST.csv` | `ranking.drop_duplicates` true, `open_set_min_detectors` 2 |

**Tag precedence.** When the typology on the left fits the case, the tags on the right are not submitted: they
describe the relations the left one is made of, not a separate reason.

| typology that fits | general tags it replaces |
|---|---|
| DENSE_MULTI_RELATION_CLUSTER | SHARED_ADDRESS, SPONSOR_CONCENTRATION, APPROVER_CONCENTRATION, SPONSOR_APPROVER_COUPLING, COMMON_BENEFICIAL_OWNER, OWNERSHIP_OPERATIONAL_OVERLAP, RAPID_ONBOARDING_LINKED |
| OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | COMMON_BENEFICIAL_OWNER, OWNERSHIP_OPERATIONAL_OVERLAP (the related vendors share the new owner) |
| OWNERSHIP_OPERATIONAL_OVERLAP | SHARED_ADDRESS, SPONSOR_CONCENTRATION (the operational overlap) |
| RAPID_ONBOARDING_LINKED, ACTIVITY_TIMING_CONCENTRATION, GEOGRAPHY_LINKED_CONCENTRATION, SPONSOR_APPROVER_COUPLING, PUBLIC_INSTITUTION_EXPERT_CLUSTER | SPONSOR_CONCENTRATION (each is defined on one sponsor's vendors) |

OWNERSHIP_OPERATIONAL_OVERLAP is submitted with COMMON_BENEFICIAL_OWNER, and SPONSOR_APPROVER_COUPLING with
APPROVER_CONCENTRATION, as in the starter labels; the more specific tag of each pair comes first.

### The blind spot

The main data has no labels, so which typology has zero recall cannot be confirmed. What was checked:
- **Every candidate the method generates can be matched.** In v1.1, nearly every typed TEST-period candidate
  has a submitted case at IoU ≥ 0.5 (all but 7 of 608, one `shared_address` and six `ownership_overlap`). A
  typology with zero recall is therefore one the method generates no candidates for, as the client says.
- **Ownership changes.** Registry R holds 14 vendors whose direct owner changed after onboarding (4 in TEST);
  `ownership_change` finds all 14. A change further up the chain occurs for 12 more vendors, none in TEST.
  Registry L has no parent change after onboarding.
- **Payments after termination.** 4 vendors are paid after their termination date; `post_termination` finds all 4.
- **Employees as owners or vendors** (a conflict of interest): employee names match registry persons and
  individual vendors only by chance; no employee sponsors or approves a vendor owned by a person of the same name.
- **Payment fields** (round amounts, manual payments, invoices above the contract band, paying market): no
  tail beyond what a data set 21 times larger explains. 33 approvals dated more than 30 days after the
  engagement start all sit in known sponsor–approver pairs.
- **Experts of one public institution engaged by one sponsor.** The starter data has exactly one such group
  (6 experts in VN, inside CASE-0032, a case whose five tags have no name for that part). The main data has 16
  (at least 3 experts), 5 of them with at least 4 experts joining within 90 days. Dossier §3.1 names the
  pattern ("one internal team connecting a dense group of public-institution HCPs, synchronized participation
  around a product and period"). Layer 2 proposed it (variant 14+3), but variant 14, submitted in v1.0 and v1.1,
  did not run it.

The expert cluster is the strongest candidate found, and v1.2 adds it. If the next score still reports a
typology with zero recall, the blind spot is elsewhere.

### v1.2 against v1.1 on the starter data

| period | version | submitted | **AP** | P@10 | P@25 | P@50 | mean IoU | exact | **tag exact** | FP in top 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAINING | v1.1 | 135 | **0.864** | 0.90 | 0.60 | 0.32 | 0.984 | 0.938 | 0.750 | 34 |
| | v1.2 | 74 | **0.864** | 0.90 | 0.60 | 0.32 | 0.984 | 0.938 | **1.000** | 34 |
| VALIDATION | v1.1 | 32 | **0.929** | 0.60 | 0.28 | 0.14 | 1.000 | 1.000 | 0.714 | 25 |
| | v1.2 | 19 | **0.929** | 0.60 | 0.28 | 0.14 | 1.000 | 1.000 | **1.000** | 12 |
| TEST | v1.1 | 57 | **0.869** | 0.80 | 0.40 | 0.20 | 0.864 | 0.300 | 0.400 | 40 |
| | v1.2 | 31 | **0.856** | 0.80 | 0.40 | 0.20 | 0.864 | 0.300 | **0.700** | 21 |
| all 33 at the snapshot | v1.1 | 186 | **0.935** | 1.00 | 0.92 | 0.62 | 0.951 | 0.758 | 0.667 | 19 |
| | v1.2 | 112 | **0.924** | 1.00 | 0.96 | 0.64 | 0.951 | 0.758 | **0.909** | 18 |

Every labelled case is still found in its own period's top 50 (16/16, 7/7, 10/10). Case boundaries do not
change.

**Tags.** All 16 TRAINING and all 7 VALIDATION cases now carry exactly their labelled tags (v1.1: 12 and 5).
What was fixed:
- SPONSOR_CONCENTRATION cases (CASE-0004, CASE-0005) no longer carry RAPID_ONBOARDING_LINKED: the rapid
  onboarding finding covered half of their members.
- RAPID_ONBOARDING_LINKED cases no longer carry SPONSOR_CONCENTRATION (CASE-0011), or only it (CASE-0010).
- The dense clusters (CASE-0008, CASE-0009) are tagged DENSE_MULTI_RELATION_CLUSTER instead of four or five
  of the relations they are made of.
- The ownership change with two related vendors (CASE-0029) is tagged OWNERSHIP_CHANGE_STRUCTURAL_SHIFT, not
  COMMON_BENEFICIAL_OWNER.
- CASE-0026 gets OWNERSHIP_OPERATIONAL_OVERLAP back, and CASE-0017 loses SPONSOR_CONCENTRATION.

The three TEST cases still wrong are the composite cases with three to six labelled tags (CASE-0031,
CASE-0032, CASE-0033).

**Ranks.** In TRAINING and VALIDATION no labelled case moves. In TEST, CASE-0007 and CASE-0012 move down one
place each (11 → 12, 12 → 13), so AP is 0.856 (−0.013): a SPONSOR_CONCENTRATION case rises from 32 to 11,
because several sponsor cases now carry a more specific primary tag and the interleaving counts fewer sponsor
cases before it.

**Volume.** 40–45% fewer cases are submitted, and the false positives in the top 50 halve in VALIDATION and
TEST (the lists are shorter than 50). No labelled case is among the cases left out.

Caveat: the tag rules were checked on these labelled cases, the development data; the main-data score is the
real test.

### v1.2 on the main data

Variant 14 on `Full Dataset/`; the whole notebook, with the four variants of section 8, runs in about 80 minutes.

| period | candidates | cases submitted (v1.1 → v1.2) | cases tagged OPEN_SET (v1.1 → v1.2) | policy findings (own list) |
|---|---|---|---|---|
| TRAINING | 3,764 | 3,078 → 1,967 | 1,195 → 16 | 21 |
| VALIDATION | 4,433 | 956 → 622 | 368 → 14 | 7 |
| **TEST** (deliverable) | 5,484 | 1,392 → **969** | 460 → 17 | **15** |

**Deliverable:** `output/v1.2/submission_TEST.csv` (969 TEST cases, variant 14) and
`output/v1.2/policy_findings_TEST.csv` (the same 15 findings as v1.1).

| TEST queue | version | mean members | largest case | cases > 15 members | OPEN_SET as primary tag | shared with the other version (IoU ≥ 0.5) |
|---|---|---|---|---|---|---|
| first 10 | v1.1 | 3.7 | 9 | 0 | 0 | 8 |
| | v1.2 | 4.1 | 9 | 0 | 0 | 8 |
| first 25 | v1.1 | 6.3 | 14 | 0 | 1 | 23 |
| | v1.2 | 5.9 | 14 | 0 | 0 | 23 |
| first 50 | v1.1 | 5.9 | 15 | 0 | 2 | 44 |
| | v1.2 | 5.9 | 15 | 0 | 1 | 44 |
| first 100 | v1.1 | 6.1 | 16 | 1 | 8 | 85 |
| | v1.2 | 5.7 | 16 | 1 | 3 | 85 |
| all | v1.1 | 6.0 | 77 | 57 | 460 | |
| | v1.2 | 6.2 | 77 | 48 | 17 | |

(`results/graph/v1.2/full_dataset/compare_with_v1.1.csv`.)

**Volume.** 969 TEST cases (v1.1: 1,392, −30%). Left out: 402 cases whose only evidence was one open-set
detector (327 relation-graph communities, 29 change points, 29 entity outliers, 17 paid duplicates) and 24
near-duplicates of a higher case. In v1.1 the highest of them was at rank 78 and half were below rank 1,023, so
the first 50 are not affected by the cut.

**Tags.** Of the 966 cases with the same members in both versions, 86 change tags:
- 24 approver groups split by sponsor become `SPONSOR_APPROVER_COUPLING|APPROVER_CONCENTRATION`;
- 40 cases tagged only OPEN_SET in v1.1 get a typology whose detector covers them (13 ACTIVITY_TIMING_CONCENTRATION,
  7 OWNERSHIP_OPERATIONAL_OVERLAP, 7 SHARED_ADDRESS, 7 APPROVER_CONCENTRATION, …);
- the rest lose a tag whose detector covered only part of the case, or replace general tags by the specific one.

**The first 50.** 44 of the first 50 are v1.1 cases. 12 of them change tags, for example:

| rank | v1.1 tags | v1.2 tags |
|---|---|---|
| 1, 2 | LAYERED_COMPLEX_OWNERSHIP (the layered structure is one of the two vendors) | OWNERSHIP_OPERATIONAL_OVERLAP\|COMMON_BENEFICIAL_OWNER |
| 4, 15 | COMMON_BENEFICIAL_OWNER\|OWNERSHIP_OPERATIONAL_OVERLAP\|OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | OWNERSHIP_CHANGE_STRUCTURAL_SHIFT |
| 6 | COMMON_BENEFICIAL_OWNER\|OWNERSHIP_OPERATIONAL_OVERLAP\|DENSE_MULTI_RELATION_CLUSTER | DENSE_MULTI_RELATION_CLUSTER |
| 17 | SPONSOR_CONCENTRATION | GEOGRAPHY_LINKED_CONCENTRATION |
| 26 | OPEN_SET | ACTIVITY_TIMING_CONCENTRATION |
| 42, 50 | APPROVER_CONCENTRATION | SPONSOR_APPROVER_COUPLING\|APPROVER_CONCENTRATION |

6 cases enter the first 50: three common-owner cases (v1.1 ranks 53, 77, 139), two approver groups split by
sponsor (v1.1 ranks 54 and 1,159) and one public-institution expert cluster. The 6 they replace move to ranks
54–132.

**Public-institution expert clusters.** 3 in TEST: 3 experts at a hospital in SK (rank 48), 3 at a clinic in
KW (rank 58), and 5 at a clinic in NG (rank 64). The NG experts are engaged by the medical lead whose four
congress vendors are v1.0's case 3, and both groups are paid most in May 2026
(`results/graph/v1.2/figures/ng_expert_cluster_TEST.png`): the shape of the starter's composite case CASE-0032
(congress vendors plus public-institution experts under one sponsor). Here they are two separate cases.

### Archive

`results/graph/v1.2/` has the layout of `results/graph/v1.1/` without `ablation/`, plus
`full_dataset/dropped_TEST.csv` (the TEST cases left out of the deliverable, and why),
`full_dataset/compare_with_v1.1.csv` and `figures/ng_expert_cluster_TEST.png` (the NG expert cluster and v1.0's
case 3 under one sponsor).

---

## Layer 3 v1.3: the main-data validation labels

Notebook: `layer3_graph_detection.ipynb` (current). Method hash: `80e0a80a439a`. Variant: **14**, chosen on the main-data
VALIDATION labels (section 8). v1.3 is the first version developed on labelled main data. Each change is marked `v1.3`
in the code or in section 0.1 of the notebook; section 9 of the notebook holds the validation analysis. Built
2026-11-18, not yet submitted.

### v1.3 in short: what changed and the effect

The client's score of v1.2 (P@10 0.70, at least one typology with zero recall) came with 43 labelled VALIDATION cases
of the main data. v1.3 is built on them. It changes which cases count for the period, ranks three kinds of noisy
findings lower, adds two detectors and removes one. The last column measures each change by removing it from v1.3
on the 43 labelled cases; the change log below gives each change's overfitting risk.

| # | change | why: what the validation labels showed | effect (without → with the change) |
|---|---|---|---|
| 1 | **Period filter.** A case also counts for the period when a finding that covers it emerged in the period. A structure that already stood at the period start is listed after the period's own cases | 8 of the 43 cases were found by a detector but dropped as older than the period | found 25 → 33; AP 0.391 → 0.419 |
| 2 | **Approver groups without a sponsor coupling rank low** (`approver_hub`, open-set prior 0.3) | 239 such cases in the period, none of them labelled; every labelled approver case is a sponsor–approver pair | AP 0.391 → 0.419; P@10 0.8 → 0.9 |
| 3 | **Cross-border corridor detector** (`corridor`, GEOGRAPHY_LINKED_CONCENTRATION) | geography had the lowest recall (1 of 3); VAL-24 was found, but under the wrong tags | VAL-24 gets the right tag; tag exact 0.727 → 0.758 |
| 4 | **Public-institution expert cluster removed** | the labels use the 14 tags only | neutral; 3 TEST cases with a tag the labels never use leave the list |
| 5 | **Variant chosen on the main-data labels** (43 cases instead of 7 starter cases) | the main-data labels are now available | variant 14, which also has the best AP (0.419; 14+1 and 14+3 0.413) |
| 6 | **Chance spike groups rank low** (`portfolio_spikes`): a spike group that chance explains among the sponsor's vendors (binomial chance ≥ 0.001) | 117 spike groups in the period, none labelled, five in the first 50 | AP 0.382 → 0.419; P@10 0.8 → 0.9; P@25 0.56 → 0.64 |
| 7 | **Sponsor-burst detector** (`sponsor_burst`, RAPID_ONBOARDING_LINKED): at least 4 new vendors of one country under one sponsor within 14 days, far above the sponsor's own rate | VAL-18 was missed: the relation detectors skip sponsors with more than 30 vendors | found 32 → 33; AP 0.413 → 0.419 |
| 8 | **Benign note on expert owners**: the owner is also engaged as an individual expert, and no sponsor engages both | 310 such findings in the period, none labelled | AP 0.417 → 0.419; starter VALIDATION 0.929 → 0.934 |

The effects overlap, so they do not add up. All eight together:

| | v1.2 | **v1.3** |
|---|---|---|
| main-data VALIDATION (43 labelled cases): **AP** | 0.308 | **0.419** |
| P@10 / P@25 / P@50 | 0.8 / 0.44 / 0.38 | **0.9 / 0.64 / 0.40** |
| cases found (IoU ≥ 0.5) | 24 | **33** |
| found cases with the exact members / exactly the right tags | 0.583 / 0.625 | **0.636 / 0.758** |
| recall of easy / medium / hard cases | 0.75 / 0.59 / 0.22 | **1.00 / 0.86** / 0.22 |
| starter labels, AP: TRAINING / VALIDATION / TEST | 0.864 / 0.929 / 0.856 | **0.893 / 0.934 / 0.881** |
| TEST deliverable, cases | 969 | 1,897 |

- **Where it gains:** at the top of the list and in coverage. The first 25 hold 16 labelled cases instead of 11, and
  every typology now has recall of at least 0.40 (v1.2: three typologies below it, geography 1 of 3). Over 2,000
  bootstrap resamples the AP change is +0.113 (90% interval +0.045 to +0.195). The settings were chosen on these
  same cases, so the interval is optimistic.
- **Where it does not gain:** the first 50 hold one more labelled case (20 instead of 19), the hard cases stay at 2 of
  9, and activity timing at 2 of 5. 10 cases are still missed (see Limitations).
- **On TEST:** the first 10 share 9 cases with v1.2's list and the first 50 share 38. The list is longer because of
  change 1: 930 of its cases enter through a covering finding or as standing structures, all of them after rank 100.
  How v1.3 scores on TEST will be known only from the client's score.

### The client's score of v1.2, and the validation labels

| metric (TEST, main data) | v1.0 | v1.1 | v1.2 |
|---|---|---|---|
| P@10 | 0.50 | **0.80** | 0.70 |
| P@25 | 0.48 | **0.60** | 0.56 |
| P@50 | 0.38 | **0.48** | 0.42 |
| mean case IoU of matched cases | 0.83 | 0.81 | 0.81 |
| mean member recall | 0.91 | 0.90 | 0.89 |
| exact recovery | 0.52 | 0.50 | 0.52 |
| typology accuracy on matched cases | **0.79** | 0.74 | **0.79** |
| typologies with zero recall | at least one | at least one | at least one: "a structural blind spot" |

Source: the client's e-mail (`output/v1.2/client_score_2026-11.csv`). With it the sponsor released the labelled
VALIDATION cases of the main data, `Full Dataset/validation_student_safe.csv`: 43 cases with their member records,
typology tags, difficulty tier, market and emergence date. The TEST labels stay blind. The sponsor's boundaries, and
how v1.3 keeps them:

| boundary | in v1.3 |
|---|---|
| validation labels for development, error analysis and calibration only | read by the scorer and by notebook section 9; detection, case assembly and ranking never read them |
| `difficulty_tier` is not a model feature | used only to stratify recall in section 9 |
| market prevalence is not a risk signal | no rate of labelled cases by market enters the method |
| use the emergence date for temporal validity | each case is scored on the VALIDATION run (graph as of 2025-06-30); 16 of its 199 member records were onboarded after that date and are left out of the case |
| document the changes made after reading the labels | the change log below, and notebook section 9 |

The labels use the 14 tags only (no PUBLIC_INSTITUTION_EXPERT_CLUSTER), 3 to 7 cases each; 6 cases carry several tags
(three BRIDGE_ENTITY|LAYERED_COMPLEX_OWNERSHIP|OWNERSHIP_OPERATIONAL_OVERLAP, two with five or six tags). Tiers: 12 easy,
22 medium, 9 hard.

### What the validation labels showed

All numbers in this section: the main-data VALIDATION run against the 43 labelled cases.

| VALIDATION list | cases | **AP** | P@10 | P@25 | P@50 | found | mean IoU | exact | tag exact |
|---|---|---|---|---|---|---|---|---|---|
| v1.1 (archived main-data run) | 956 | 0.298 | 0.7 | 0.44 | 0.36 | 27 | 0.847 | 0.519 | 0.444 |
| v1.2 (archived main-data run) | 622 | 0.308 | **0.8** | 0.44 | 0.38 | 24 | 0.877 | 0.583 | 0.625 |
| **v1.3** | 1,457 | **0.419** | **0.9** | **0.64** | **0.40** | **33** | **0.900** | **0.636** | **0.758** |

Each list is the VALIDATION period on the main data (graph as of 2025-06-30, the period's cases), scored at IoU ≥ 0.5
against the 43 cases. Note the direction of v1.1 → v1.2: on these labels v1.2 is ahead at the top (P@10 0.7 → 0.8),
while the client's TEST score fell (0.8 → 0.7); point 5 below.

1. **The structural blind spot is the period filter, not a missing detector.** v1.2 misses 19 of the 43 cases. For 8
   of them (VAL-01, 02, 09, 24, 31, 33, 35, 36) a detector finds the case (candidate IoU 0.75–1.0), but the case is
   dated before the period and dropped. A case took the earliest date of everything merged into it: VAL-01's shared
   address formed in the period (a finding dated 2024-11-22) but merged with an older relation of the same vendors
   (2023). And some planted structures carry no date inside the period at all: VAL-24's RU vendors have shared one SG
   correspondence address since 2017–2022, VAL-35's owners were registered in 2009, and nothing in the ownership,
   address, engagement or payment data changes near their emergence dates.
2. **GEOGRAPHY_LINKED_CONCENTRATION has the lowest recall**: 1 of 3 in v1.2 (0 of 3 counting the members onboarded
   after the period), and the one found ranks 159th. It is the most likely zero-recall typology of the TEST score.
   VAL-24, the cross-border corridor of point 1, is found as a shared address and an ownership overlap, never as
   geography.
3. **Approver hubs without a sponsor coupling are noise.** `approver_star` leads 239 cases in the VALIDATION period and
   none matches a labelled case. Every labelled approver case is a coupled sponsor–approver pair (4 of 4 here, and
   every starter case), found by `sponsor_approver_pair`.
4. **v1.2's drop rules cost three composite cases.** VAL-40, VAL-42 and VAL-43 (3–6 tags each) were reached only by
   open-set detectors; v1.2 leaves out such lone open-set cases. They sat at ranks 267–571 in v1.1.
5. **The ranking change of v1.2 is not confirmed.** The client's P@k fell from v1.1 to v1.2 (one, one and three cases
   at 10, 25 and 50). The cause is the type interleaving on the submitted primary tag (v1.2): plain common-owner and
   approver cases move up. On the validation labels the same change is neutral to positive (P@10 0.7 → 0.8), and
   reverting it costs AP (0.308 → 0.294). Both differences are within one to three cases; v1.3 keeps v1.2's
   interleaving.

Points 6 to 8 come from an error analysis of the list with changes 1 to 5 only (AP 0.374): what fills its first 50,
and why the 11 cases still missed are missed.

6. **Synchronised spend spikes are chance in large portfolios.** `sync_spikes` leads 117 cases in the VALIDATION
   period and none matches a labelled case, yet five sit in the first 50. Each is three to six vendors out of 20 to 30
   under one sponsor, which a two-month spike rate of 2.4% per vendor produces by chance. The labelled starter spike
   groups are three of a sponsor's four vendors (chance 5·10⁻⁵).
7. **Onboarding bursts under busy sponsors go unlinked.** VAL-18 is four CO vendors onboarded under one sponsor within
   11 days. The relation detectors skip a sponsor with more than 30 vendors as a hub, so nothing tied the four
   together. Counted against each sponsor's own rate, such bursts are rare: 6 in the VALIDATION period, and 4 of them
   belong to labelled cases (VAL-05, VAL-14, VAL-18, VAL-20).
8. **An expert who owns vendors is not, by itself, a planted case.** 310 owner findings in the VALIDATION period have
   an owner who is also engaged as an individual expert, with no sponsor engaging both; none matches a labelled case.
   The one labelled starter case with an expert owner (CASE-0024) is a group that the expert's own sponsor engages.

Points 6 to 8 lead to changes 6 to 8. What v1.3 does not change: the first 50 hold 20 of the 43
cases, ranks 51 to 100 hold one, and four of the next hits (VAL-05, VAL-06, VAL-18, VAL-23, ranks 94–186) are
onboarding cases that carry the acquisition-integration benign note. Acquisition windows cover about half of the
VALIDATION period and none of the TEST period, so the note was left as it is.

### Change log: material changes made after reading the validation labels

The sponsor asked that every material change made after reviewing the validation labels state the issue observed, the change, the expected improvement and the overfitting risk. Each effect below is measured on the main-data VALIDATION run by removing that one change from the final v1.3 (AP over the 43 cases; "found" = cases matched at IoU ≥ 0.5). Rows 1–5 answer the client's score of v1.2; rows 6–8 followed an error analysis of the list with rows 1–5 only (AP 0.374): its false positives in the first 50 and the cases still missed.

| # | validation issue observed | change made (setting) | expected improvement | overfitting risk |
|---|---|---|---|---|
| 1 | 8 of the 43 cases are found by a detector (candidate IoU 0.75–1.0) but dated before the period and dropped: a case took the earliest date of everything merged into it, and some planted structures (an ownership chain, a foreign correspondence address) carry no date inside the period at all | **period filter** (`meridian.cases.in_window`): a case also belongs to the period when a typed finding that covers it (member IoU ≥ 0.5) emerged in it, ranked with weight 0.5; a case whose covering finding still holds at the period start is submitted behind the period's own cases, with weight 0.1 (`window_findings`, `window_finding_weight`, `window_standing`) | found 25 → 33; every typology's recall ≥ 0.40; AP 0.391 → 0.419. The standing cases themselves rank 254th–557th; most of the AP comes from their entering each detector's percentile distribution, which reorders the period's own cases | **medium.** The rule follows the labels' emergence dates, which the data cannot reproduce for static structures. The weights are set by rule, not tuned: standing weight 0.5 scores 0.431 on these labels but costs the starter VALIDATION period (0.934 → 0.896); finding weight 1.0 scores 0.427 but puts two older owner groups above labelled starter TEST cases (0.881 → 0.793). The cost is volume: older structures are appended to every list, most of them not cases of the period |
| 2 | `approver_star` leads 239 cases in the VALIDATION period and none matches a labelled case; every labelled approver case is a coupled sponsor–approver pair (4 of 4 here, all starter cases) | an approver group without sponsor coupling is reported by `approver_hub` and ranks with the open-set prior 0.3 (`detector_priors`) | AP 0.391 → 0.419, P@10 0.8 → 0.9, P@25 0.60 → 0.64; starter TRAINING 0.864 → 0.893 | **low–medium.** The rule rests on every labelled approver case of both label sets; the prior is the method's existing open-set prior, not fitted |
| 3 | GEOGRAPHY_LINKED_CONCENTRATION has the lowest recall (v1.2: 1 of 3; 0 of 3 counting members onboarded after the period). VAL-24, three RU vendors with one SG correspondence address that remit to SG, is found only as a shared address and an ownership overlap | a **cross-border corridor** detector (`corridor`): at least 3 vendor entities of one market share an address abroad that is also their remit-to country; a fitting geography tag replaces the shared-address and ownership tags it is built from (`SUBSUMES`) | VAL-24 carries the right tag: tag exact 0.727 → 0.758; geography 1 → 2 of 3 found (with the period filter) | **high per case, small in reach.** One labelled example; the rule fires on 2 groups in the whole data (RU→SG, which is VAL-24, and MA→PA, unlabelled) |
| 4 | the labels use the 14 tags only; the v1.2 expert-cluster cases match none | `public_institution_cluster` leaves the method (variant 14+3 still runs it) | neutral on validation; 3 TEST cases that cannot carry a right tag leave the list | low |
| 5 | the variant is now chosen on these labels | **variant choice** (notebook section 8): AP, then mean IoU, among the variants whose added typologies occur in the labels (a typology the sponsor does not label cannot be the right reason for a case); this leaves **14**, which also has the highest AP (0.419; 14+1 and 14+3 0.413) | the deliverable is chosen on 43 main-data cases instead of 7 starter cases | low: the rule uses the labels' vocabulary, not their scores |
| 6 | `sync_spikes` leads 117 cases in the VALIDATION period and none matches a labelled case, yet five of them sit in the first 50 of the list. They are three to six vendors out of 20 to 30 under one sponsor; the labelled starter spike groups are three of a sponsor's four vendors | the chance of a spike group is judged against the sponsor's portfolio: P(Binomial(N, q) ≥ k) for k of the sponsor's N vendors, q the data's two-month spike rate. A group with chance ≥ 0.001 is reported as `portfolio_spikes` and ranks with the open-set prior 0.3 (`activity.max_p`, `detector_priors`) | AP 0.382 → 0.419, P@10 0.8 → 0.9, P@25 0.56 → 0.64; the starter periods do not change (their labelled groups have chance 5·10⁻⁵) | **low–medium.** A significance rule, not fitted: 0.001 and 0.0001 give the same lists; 0.01 keeps more of the groups (AP 0.391 against 0.411 when tested). It rests on 117 unmatched groups and two labelled starter groups; it does not find the activity-timing cases still missed |
| 7 | VAL-18, four CO vendors onboarded under one sponsor within 11 days, is missed: the relation detectors skip a sponsor with more than 30 vendors as a hub, so nothing linked the four | a **sponsor-burst** detector (`sponsor_burst`, RAPID_ONBOARDING_LINKED): within 14 days a sponsor brings at least 4 new vendor entities of one country, with a chance below 0.001 under a Poisson rate from the sponsor's own previous two years there (`sponsor_burst` parameters) | found 32 → 33 (VAL-18, IoU 0.80); AP 0.413 → 0.419. Of the 6 bursts in the VALIDATION period, 4 belong to labelled cases (VAL-05, VAL-14, VAL-18, VAL-20); in the starter data it matches CASE-0008 to CASE-0011 (IoU 0.88–1.00) | **medium.** Designed after one missed case, but it is a generic rate test and rare: 85 bursts in the main data's whole history at the VALIDATION snapshot, 3 in the TEST period. Two of the VALIDATION bursts (VAL-05, VAL-18) carry an acquisition-integration benign note; their cases rank 94th and 102nd. It widens one starter case by a vendor (CASE-0009, IoU 1.00 → 0.88) |
| 8 | 310 common-owner and ownership-overlap findings in the VALIDATION period have an owner who is also engaged as an individual expert, with no sponsor engaging both; none matches a labelled case | such a finding carries a benign note, "possibly the expert's own practice or consultancy" (weight 0.5). When one sponsor engages the expert and the group it stays a conflict of interest, as before (`detect_common_owner`) | AP 0.417 → 0.419; starter VALIDATION 0.929 → 0.934 | **medium.** The note on every expert owner scored higher here (AP 0.427, P@25 0.68), but the labelled starter CASE-0024 is a same-sponsor expert group (TRAINING 0.893 → 0.872); the narrower rule costs neither label set |

All eight together: AP 0.308 (v1.2) → 0.419, found 24 → 33 (the table above). Adding rows 6–8 to rows 1–5: AP 0.374 → 0.419, P@10 0.8 → 0.9, P@25 0.56 → 0.64; the starter periods 0.893 / 0.929 / 0.881 → 0.893 / 0.934 / 0.881.

**Tested and not adopted** (each against the same validation run):

| tested | effect on validation | decision |
|---|---|---|
| type interleaving of v1.1 (the client's P@k fell v1.1 → v1.2) | AP 0.374 → 0.344, P@25 0.56 → 0.48 (with rows 1–5 only) | not adopted: v1.2's interleaving is better on the labels; the TEST hint is one to three cases |
| detector hit-rate calibration from the labels (in-sample, cross-fitted, shrunk) | AP 0.28–0.34, against 0.34 with equal priors | not adopted: a detector-level rate overrides the within-detector order |
| a case-level ranker on label-free case features, cross-validated by labelled case | AP 0.22–0.26 out of fold | not adopted: 32 positives are too few |
| fixed lower priors for detectors with few hits (ownership overlap, layered ownership, bridge 0.3; common owner 0.7) | AP 0.346–0.365 against 0.374 (with rows 1–5 only) | not adopted |
| domestic bridges with the open-set prior (every labelled bridge links institutions abroad) | AP 0.411 → 0.407; starter VALIDATION 0.929 → 0.908 (with fewer than 5 cross-border bridges their percentile falls to 0.5) | not adopted |
| keep the lone open-set cases (undo v1.2's rule) | found 33 → 36 (three composite cases), AP +0.002, tag exact 0.758 → 0.694 | not adopted: the cases carry the OPEN_SET tag, a wrong reason |
| a looser spend-spike threshold (activity timing) | VAL-28 found only at the 0.65 quantile, among 320 spike groups, with chance 0.05 | not adopted |
| one global setting at a time (corroboration 0.75 / 1.0, finding weight 0.75, type decay 0.9, benign weight 0.6, hub prior 0.2) | AP 0.422–0.435 against 0.427 (tested with the broader expert-owner note of row 8); all but one cost a starter period (TRAINING by 0.008–0.024, VALIDATION or TEST by 0.020); the approver-hub prior 0.2 costs nothing and gains 0.004 | not adopted: the two label sets disagree, and +0.004 is within one case |
| the acquisition-integration benign note (VAL-05, VAL-06, VAL-18 and VAL-23 carry it and rank 94th–186th) | not changed: acquisition windows cover about half of the VALIDATION period, but the TEST period has none | — |
| within-detector percentiles against every case at the snapshot | AP 0.374 → 0.372 (with rows 1–5 only) | not adopted (kept as the option `percentile_pool`) |

### v1.3 against v1.2 on the validation labels

Recall by typology (a case counts for each of its tags):

| typology | cases | v1.1 | v1.2 | v1.3 |
|---|---|---|---|---|
| ACTIVITY_TIMING_CONCENTRATION | 5 | 0.40 | 0.40 | 0.40 |
| APPROVER_CONCENTRATION | 4 | 0.50 | 0.50 | **0.75** |
| BRIDGE_ENTITY | 4 | 1.00 | 0.50 | 0.50 |
| COMMON_BENEFICIAL_OWNER | 7 | 0.71 | 0.57 | **0.86** |
| DENSE_MULTI_RELATION_CLUSTER | 7 | 0.86 | 0.71 | 0.71 |
| GEOGRAPHY_LINKED_CONCENTRATION | 3 | 0.33 | 0.33 | **0.67** |
| LAYERED_COMPLEX_OWNERSHIP | 4 | 0.75 | 0.25 | 0.50 |
| OPAQUE_JURISDICTION_ROUTING | 3 | 0.67 | 0.33 | 0.67 |
| OWNERSHIP_CHANGE_STRUCTURAL_SHIFT | 4 | 1.00 | 0.75 | 0.75 |
| OWNERSHIP_OPERATIONAL_OVERLAP | 5 | 0.80 | 0.40 | 0.60 |
| RAPID_ONBOARDING_LINKED | 5 | 0.80 | 0.60 | **0.80** |
| SHARED_ADDRESS | 5 | 0.40 | 0.40 | **0.80** |
| SPONSOR_APPROVER_COUPLING | 6 | 0.67 | 0.50 | 0.67 |
| SPONSOR_CONCENTRATION | 3 | 1.00 | 1.00 | 1.00 |

By difficulty tier (sponsor metadata, evaluation only): easy 0.75 / 0.75 / **1.00** (12 cases), medium 0.59 / 0.59 /
**0.86** (22), hard 0.56 / 0.22 / 0.22 (9), for v1.1 / v1.2 / v1.3. The hard cases are where v1.3 gains nothing: 2 of 9
(VAL-04, VAL-10, VAL-15, VAL-26 and the three composite cases are missed).

**Uncertainty.** Over 2,000 bootstrap resamples of the 43 cases, the AP change v1.2 → v1.3 is +0.113 (90% interval
+0.045 to +0.195), positive in every resample. The settings were
chosen on these same cases, so the interval is optimistic.

**What P@k can reach.** P@k cannot exceed the number of planted cases of the period divided by k. The VALIDATION period
has 43, so P@100 is at most 0.43 there (v1.3: 0.21). The TEST period's number is not disclosed; the client's P@50 of
0.48 for v1.1 means at least 24, and the ratio of TEST to VALIDATION hits for v1.1 suggests about 50–60. P@50 or P@100
near 0.9 would need about 45 or 90 TEST cases in the list's first 50 or 100.

### v1.3 on the starter data

The starter data checks that nothing breaks; v1.3 was not tuned on it, beyond the finding weight and the narrower
expert-owner rule above.

| period | version | submitted | **AP** | P@10 | P@25 | mean IoU | exact | tag exact | FP in top 50 |
|---|---|---|---|---|---|---|---|---|---|
| TRAINING | v1.2 | 74 | 0.864 | 0.9 | 0.60 | 0.984 | 0.938 | 1.000 | 34 |
| | **v1.3** | 74 | **0.893** | 0.9 | 0.60 | 0.977 | 0.875 | 1.000 | 34 |
| VALIDATION | v1.2 | 19 | 0.929 | 0.6 | 0.28 | 1.000 | 1.000 | 1.000 | 12 |
| | **v1.3** | 65 | **0.934** | 0.6 | 0.28 | 1.000 | 1.000 | 1.000 | 43 |
| TEST | v1.2 | 31 | 0.856 | 0.8 | 0.40 | 0.864 | 0.300 | 0.700 | 21 |
| | **v1.3** | 85 | **0.881** | 0.8 | 0.40 | 0.864 | 0.300 | 0.700 | 40 |
| all 33 at the snapshot | v1.2 | 112 | 0.924 | 1.0 | 0.96 | 0.951 | 0.758 | 0.909 | 18 |
| | **v1.3** | 112 | **0.946** | 1.0 | 0.96 | 0.947 | 0.727 | 0.909 | 18 |

All 33 labelled cases are found, with the same tags as v1.2 and the same boundaries but one: a sponsor burst
corroborates CASE-0009 (TRAINING rank 5 → 1) and adds one vendor to it (IoU 1.00 → 0.88). The lists are longer (the
period filter adds older structures behind the period's cases), so with fewer than 50 cases in v1.2's lists the false
positives in the top 50 rise.

### v1.3 on the main data (TEST)

Variant 14 on `Full Dataset/`; the whole notebook runs in about 70 minutes.

**Deliverable:** `output/v1.3/submission_TEST.csv` (1,897 TEST cases) and `output/v1.3/policy_findings_TEST.csv` (the
same 15 findings as v1.1 and v1.2).

| TEST list | cases | shared with v1.2's list (same first k, IoU ≥ 0.5) |
|---|---|---|
| first 10 | 10 | 9 |
| first 25 | 25 | 22 |
| first 50 | 50 | 38 |
| first 100 | 100 | 77 |
| all | 1,897 (v1.2: 969) | 966 of v1.2's 969 |

(`results/graph/v1.3/full_dataset/compare_with_v1.2.csv`.)

- **Why each case is in the period:** 727 emerged in it, 240 grew in it, 7 have a covering finding that emerged in it
  (first at rank 113), and 923 are standing structures (first at rank 299, median 1,225). The first 100 hold none of the
  last two kinds.
- **Left the first 50:** four approver groups without sponsor coupling (v1.2 ranks 28, 35, 43, 50), two spike groups
  (31, 38; now 107 and 114), the expert cluster (48), two common-owner cases (30, 41), a sponsor–approver pair (42), an
  ownership change (44) and a rapid onboarding (47). **Entered:** five common-owner cases, an ownership overlap, two
  bridges, two sponsor–approver pairs, a sponsor concentration (v1.2 ranks 53–94) and a sponsor burst (rank 39; v1.2:
  125).
- **Changes 6 to 8 on TEST:** with or without them the first 10 are the same cases, and 48 of the first 50. The
  three sponsor bursts of the TEST period lead a case at rank 39 and one at 112, and corroborate the rapid onboarding
  at rank 11. No `portfolio_spikes` case is in the first 100; three `sync_spikes` cases are.
- **Corridors:** both are listed as standing structures with the tag GEOGRAPHY_LINKED_CONCENTRATION: MA→PA (rank 356,
  4 entities, unlabelled) and RU→SG (rank 367; this is VAL-24, a VALIDATION case, so a false positive in TEST).
- **Grown cases at the top:** 3 of the first 10 grew in the period (ranks 3, 5, 9). All three were in v1.1's and v1.2's
  first 10 as well, where the client found 8 and 7 of 10, so they are likely planted cases; one more reason not to
  down-weight grown cases.

### Limitations

- **Activity timing** is the weakest typology on the labels (2 of 5, both payments after termination). v1.3 stops the
  spike groups that chance explains from crowding the top, but finds no new activity-timing case: the planted spikes
  (2–3.5× the vendor's own median, some on vendors with a few months of history) sit below the per-vendor threshold,
  and VAL-28's group is ordinary for its sponsor's 15 vendors (chance 0.05).
- **10 validation cases are still missed**: 5 only partly found (boundaries: one of three addresses in another form,
  approver groups across sponsors, a two-sponsor event cluster), 3 composite cases reached only by open-set detectors
  (VAL-42 and VAL-43 pair a regulatory consultant with a clinical-site company that has no registry record, so the
  ownership link is not in the data), 2 not found (VAL-22 has 2 of its 5 members by the period end; VAL-29 is a
  single vendor with two months of payments).
- **P@50 barely moves** (0.38 → 0.40 on the labels): v1.3 moves labelled cases up into the first 25 (11 → 16) more
  than it brings new ones into the first 50, and ranks 51–100 hold one labelled case.
- **43 cases is a small sample.** The bootstrap interval in section 9.3 of the notebook gives the uncertainty of the AP
  change.

### Archive

`results/graph/v1.3/` has the layout of `results/graph/v1.2/`, plus in `full_dataset/`: `validation_metrics.csv` (every
variant on the validation labels), `validation_compare.csv`, `validation_recall_by_tag.csv`,
`validation_recall_by_difficulty.csv` and `validation_case_landing.csv` (section 9), and `compare_with_v1.2.csv`.

---

## Running

The Layer 3 notebook needs only the data folders and pandas, numpy, scipy, scikit-learn, networkx,
pyarrow, pyyaml and unidecode. Run it top to bottom from the repository root.
- Sections 1–7 use the starter data, in about 5 minutes. Results go to `results/graph/<VERSION>/`.
- Section 8 runs the four variants on `Full Dataset/` (about 60 minutes), chooses the variant on
  `Full Dataset/validation_student_safe.csv` when it is there, and writes the deliverable to `output/<VERSION>/`:
  `submission_TEST.csv` and `policy_findings_TEST.csv`.
- Section 9 analyses the main-data validation labels (about 5 minutes).

**A new version:** set `VERSION` and the settings in section 0.1, mark each change with the version, run the
notebook, and add a section to this README with its scores against the previous version. Commit the notebook so
that the version can be recovered.
