# Results

All scores come from our development scorer (one-to-one matching down the ranking, member IoU ≥ 0.5) on the 33 labelled starter cases. The detectors were designed while looking at these cases, so these numbers describe the development set and are best read as an upper bound. The first honest estimate will come from the validation labels, scored once with the frozen version before any tuning.

Full tables are in `experiments/` and can be regenerated with `meridian-exp all`. Unless stated otherwise, runs use the default configuration with seed 0.

## Headline

Production configuration: typed detectors plus open-set detectors, label-free priors.

| P@10 | P@25 | R@50 | R@100 | AP | Mean IoU | Macro / worst tag recall | Tags exact / Jaccard | False positives in top 50 |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.88 | 1.00 | 1.00 | 0.902 | 0.91 | 1.00 / 1.00 | 0.67 / 0.81 | 17 |

- All 33 planted cases rank within the top 50.
- With 33 planted cases, 17 false positives in the top 50 is the minimum possible.
- Over five seeds AP is 0.898 ± 0.008 (range 0.890–0.908).
- Baselines: a random submission scores about 0. The single exact-address rule scores AP 0.09. Record-level information alone scores AP 0.04.

## Detection on a heterogeneous graph

The source tables are turned into one time-aware heterogeneous graph, and detection reads only that graph. Normalisation and entity resolution read the source tables to build it. After that, every detector works on the graph plus a few reference tables: the service taxonomy, approval policy, FMV rate cards, country context and business events.

**What the graph holds.** At the snapshot the starter graph has:
- 10,089 nodes of 12 types: vendor records, resolved entities, addresses, contact points, declared groups, engagements, employees, service categories, institutions, registry legal entities, natural persons and accounting parents;
- 20,438 edges of 19 relation types;
- monthly payments as an activity table on engagement nodes, because they are a time series rather than a relation.

Every edge keeps its valid time, observation time, source table and record. Edges that depend on entity resolution also keep the match confidence. Uncertain registry matches stay in the graph as candidates rather than being dropped or silently accepted. Each run exports the graph (`graph/` in the run folder) with a content hash; the same inputs and configuration give the same hash.

**Nothing is lost.** Every table the detectors read can be rebuilt from the graph, and the rebuilt tables equal the source tables value for value (tested). Detection on the graph gives exactly the same cases as detection on the tables did.

**Detectors are meta-path patterns.** A meta-path is a chain of typed relations. Each relational detector looks for a pattern over one or more of them:

| Relation | Meta-path | Used by |
|---|---|---|
| Shared address | vendor –LOCATED_AT→ address ←LOCATED_AT– vendor | shared address, dense cluster, open-set communities |
| Shared contact | vendor –HAS_CONTACT→ phone / domain ← vendor | dense cluster, open-set communities |
| Shared sponsor | vendor –CONTRACTED_UNDER→ engagement –SPONSORED_BY→ employee | sponsor concentration, clusters, timing |
| Shared approver | vendor → engagement –APPROVED_BY→ employee | sponsor–approver pairs, clusters |
| Sponsor–approver pair | employee ←SPONSORED_BY– engagement –APPROVED_BY→ employee | approver concentration and coupling |
| Institution | vendor –SERVES / SITE_OF / AFFILIATED_WITH→ institution | bridge entities, clusters |
| Ownership | vendor –REGISTERED_AS→ entity ←OWNS / CONTROLS– … ← person | the five ownership typologies |

**Every case cites its edges.** Each finding records the graph edges behind it. Case packages list those edges with their dates and source records, and draw them as a subgraph. For CASE-0023, for example, the subgraph shows the full trust and holdco chain from the vendor to the person at the top.

## What the experiments show

**1. Relational information adds a large gain** (`ladder`, `fixed`)

| Level | Information | AP (complete pipeline) | Candidate coverage | AP, fixed candidates (logistic) |
|---|---|---|---|---|
| A | Single records | 0.04 | 0.09 | 0.39 ± 0.03 |
| B | + relation rules | 0.58 | 0.76 | 0.48 ± 0.01 |
| C | + graph structure | 0.56 | 0.82 | 0.54 ± 0.03 |
| D | + ownership | 0.87 | 1.00 | 0.53 ± 0.02 |

- In the complete pipeline the gain comes from finding cases: coverage rises from 0.09 to 1.00.
- On the fixed candidate set, relation rules and graph structure improve how the same candidates are ordered; ownership adds no further ordering gain there.
- The graph-structure level (C) finds two more planted cases but does not raise AP (0.58 → 0.56): on this data, the relation rules already find most of what it finds.

**2. Ownership, sponsor and address edges carry most of the value** (`ablation`)

Each row removes one family of edges from the graph before detection. Entity resolution, which built the graph, is unchanged.

| Edges removed | AP drop |
|---|---|
| Ownership (REGISTERED_AS, IS_PERSON, OWNS, CONTROLS, CONSOLIDATED_BY) | 0.29 |
| Sponsor (SPONSORED_BY) | 0.26 |
| Address (LOCATED_AT) | 0.12 |
| Approver (APPROVED_BY) | 0.07 |
| Payments (activity on engagements) | 0.04 |
| Institution links (SERVES, SITE_OF, AFFILIATED_WITH) | 0.03 |
| Contacts (HAS_CONTACT) | none |

- Transliterated name matching to the register is worth about 0.14 AP. Matching on identifiers only gives AP 0.76 and R@100 0.85.

**3. Results are stable** (`stability`, `perturb`)

| Change | AP |
|---|---|
| Random seed (5 seeds) | 0.890–0.908 |
| Any single detector threshold ±25% | 0.867–0.913 |
| All thresholds −25% at once | 0.889 |
| All thresholds +25% at once | 0.857 |
| 10–20% of name-based registry matches dropped (3 seeds) | 0.86 ± 0.03 |
| Typos in 10–20% of addresses (3 seeds) | 0.90–0.92 |
| Scorer threshold IoU ≥ 0.3 or 0.5 | 0.902–0.912 |
| Scorer threshold IoU ≥ 0.7 | 0.791 |

- At IoU ≥ 0.7 a few case boundaries are only just good enough.

**4. Learned models do not beat the transparent detectors** (`fixed`, `embedding`)

On the same candidates, learned scorers reach at most about AP 0.56, against 0.902 for the detector ranking.

| Case representation | Logistic regression (cross-validated) | Isolation forest |
|---|---|---|
| Explicit features (level D) | 0.53 ± 0.02 | 0.30 |
| GraphSAGE embedding only | 0.31 ± 0.04 | 0.08 |
| Explicit + embedding | 0.53 ± 0.01 | 0.21 |
| Explicit + embedding, encoder fitted to 2024-06-30 and frozen | 0.56 ± 0.04 | 0.22 |

- Adding embeddings changes the logistic score by 0.00–0.04, within the seed variation, and the label-free scorer gets worse.
- The planted signals are counts in time windows relative to peers, and deep ownership paths. Explicit features encode these directly; a two-layer mean-aggregating GNN does not.
- With 33 positives there is too little to learn a ranker.
- Embeddings are therefore kept as a comparison (`meridian-exp embedding`), not used in production.
- Link-prediction AUC is 0.82–0.98, so the encoder itself trains properly.
- The frozen-encoder protocol for the main data already runs: fit on the training period, then apply to later snapshots.

**5. Coverage of unseen patterns is limited** (`loto`)

We removed each type's dedicated detector in turn and measured recall for that type.

- The open-set detectors recover part of SHARED_ADDRESS (0.25 → 0.50) and ACTIVITY_TIMING_CONCENTRATION (0.50 → 0.75), and none of the other types.
- Five types are fully recovered by other typed detectors anyway.
- Expect weak recall on patterns outside the published list in the blind test.

## Findings in the data

- **CASE-0023 and CASE-0030 are one vendor (TP-98B17648) at two times.** It was layered through a trust from 2007 to 2019, then sold into a BVI holdco on 2019-07-08. Ownership has to be evaluated across each vendor's history, and two patterns on the same members are separate cases when their periods do not meet.
- **Business events only explain matching activity.** COVID-19 (2020-03 to 2021-12) applied as a blanket benign explanation suppressed every onboarding burst in those years and pushed CASE-0009 out of the top 100. It only explains logistics and cold-chain activity.
- **FMV breaches are booked as `clinical_services`.** Both experts above their annual cap are investigators paid this way. Counting only consulting fees and honoraria finds no breach.
- **Same-name vendors are usually group siblings, not duplicates.** Without a shared identifier, address or phone, they belong to one declared group, sometimes with different registry numbers.
- **Unlabelled "expert owns a vendor" pairs are all personal service companies** named after the expert. The planted ones use unrelated company names.
- **One approver sits in two planted sponsor–approver pairs** (EMP-AEE53F8B in CASE-0006 and CASE-0007), so this pattern must be detected per pair.
- **Likely generation shortcuts.** All 9 approvals dated outside the approver's role period, and all 38 records with approver level below the band requirement, belong to labelled cases. Both are used only as corroboration, because the main data may not keep them. The graph shows the same thing from another side: at earlier dates, some engagements in those cases name a sponsor or approver whose role has not started yet.
- **Results depended on the row order of the source files.** Community detection and isolation forests are sensitive to the order of their inputs. Putting every table in a fixed, content-defined order moved AP from 0.895 to 0.902, a shift about the size of the seed variation.

## Design decisions and why

| Decision | Reason |
|---|---|
| Detection reads only the graph | Makes the graph the single, inspectable input; ablation, evidence and the embedding comparison all use the same object |
| Every table is put in a fixed order defined by its content | Results must not depend on how the source files happen to be sorted |
| Label-free ranking: equal detector priors, 0.3 for open-set detectors | Cross-fitted calibration on the starter labels scored lower (AP 0.83 vs 0.902); revisit with the validation labels |
| Benign explanations down-weight only the signal they explain | A coworking address does not explain an ownership convergence; case-level down-weighting pushed planted multi-relation clusters such as CASE-0009 out of the top 100 |
| Merge candidates only if their active periods meet | Keeps CASE-0023 and CASE-0030 apart; keeps an owner group and its later address overlap together |
| Overlapping groups from different detectors merge into their union (capped at 1.5× the lead) | Multi-typology planted cases are unions across relations (AP 0.873 without unions); the cap stops groups chaining through a shared hub |
| Down-weight a case when 60% of its members are already in a higher-ranked case | Fragments of one scheme were the largest group of false positives and cannot match a second case |
| Groups sharing only process relations (sponsor, approver, category, institution) are down-weighted | Shared identity items (address, phone, domain, owner) separate planted clusters from standard local processes |
| Same-day batch entry and the first three months of spend are not bursts | Acquisition integration loads vendors in batches; new vendors ramp up together |
| Submit tags within 85% of the strongest evidence | Exact tag match 0.67 vs 0.39 when all tags are kept; keeping only the strongest tag gives 0.79 but misses correct tags; the official tag-accuracy definition is unknown |
| Offshore lists: IMF OFC list and Garcia-Bernardo et al. (2017) sink/conduit OFCs; conduit-only routes without a PEP are not reported | FSI and FATF are not in the bundle; conduit holdcos are common and legitimate |
| FMV country bands: Europe, JP, CA, AU, NZ → EU_JP; US → US; rest → EMERGING | The bundle gives no mapping; to be confirmed with the sponsor |
| Policy findings (FMV breaches) are ranked with the cases under `POLICY_VIOLATION` | The sponsor scores them separately; they count as false positives against the starter labels |

## Limitations and open items

- Every number above is on the development set. The main data is about 20 times larger, with a lower share of planted cases and more benign look-alikes, so expect more false positives near the top.
- Emergence-date rules cannot be checked until the validation labels arrive.
- Still to do: interactive case browsing and a small review exercise (Dossier §13.3).
- Questions for the sponsor:
  - the official matching threshold and scoring script;
  - how tag accuracy is computed for multi-tag cases;
  - how single-record policy findings are scored;
  - the FMV country-band mapping;
  - the release dates for the main data and the validation labels.
