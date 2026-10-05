# Results

All scores come from our development scorer (one-to-one matching down the ranking, member IoU ≥ 0.5) on the 33 labelled starter cases. The detectors were designed while looking at these cases, so these numbers describe the development set and are best read as an upper bound. The first honest estimate will come from the validation labels, scored once with the frozen version before any tuning (see "Validation and test runs" in the README).

Full tables are in `experiments/` and can be regenerated with `meridian-exp all` (and `meridian-exp scale`). Unless stated otherwise, runs use the default configuration with seed 0.

## Headline

Production configuration: typed detectors plus open-set detectors, label-free priors.

| P@10 | P@25 | R@50 | R@100 | AP | Mean IoU | Macro / worst tag recall | Tags exact / Jaccard | False positives in top 50 |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.88 | 1.00 | 1.00 | 0.933 | 0.93 | 1.00 / 1.00 | 0.67 / 0.80 | 17 |

- All 33 planted cases rank within the top 50.
- With 33 planted cases, 17 false positives in the top 50 is the minimum possible.
- Over five seeds AP is 0.928 ± 0.007 (range 0.921–0.934).
- 24 of the 33 planted cases are recovered exactly (IoU 1.0).
- Baselines: a random submission scores about 0. The single exact-address rule scores AP 0.09. Record-level information alone scores AP 0.02.

## Detection on a heterogeneous graph

The source tables are turned into one time-aware heterogeneous graph, and detection reads only that graph. Normalisation and entity resolution read the source tables to build it. After that, every detector works on the graph plus a few reference tables: the service taxonomy, approval policy, FMV rate cards, country context and business events.

**What the graph holds.** At the snapshot the starter graph has:
- 10,089 nodes of 12 types: vendor records, resolved entities, addresses, contact points, declared groups, engagements, employees, service categories, institutions, registry legal entities, natural persons and accounting parents;
- 20,438 edges of 19 relation types;
- monthly payments as an activity table on engagement nodes, because they are a time series rather than a relation.

Every edge keeps its valid time, observation time, source table and record. Edges that depend on entity resolution also keep the match confidence. Uncertain registry matches stay in the graph as candidates rather than being dropped or silently accepted. Each run exports the graph (`graph/` in the run folder) with a content hash; the same inputs and configuration give the same hash, and the same cases and case packages.

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

**Each piece of evidence counts once.** A case found by several detectors scores higher than one found by a single detector, but only for evidence that is new. When two detectors cite the same edges (a common-owner finding and the ownership-overlap finding built on the same ownership paths share 97% of their edges; a layered chain and the opaque routing through it share all of them), the second adds nothing.

## What the experiments show

**1. Relational information adds a large gain** (`ladder`, `fixed`)

| Level | Information | AP (complete pipeline) | Candidate coverage | AP, fixed candidates (logistic) |
|---|---|---|---|---|
| A | Single records | 0.02 | 0.09 | 0.39 ± 0.03 |
| B | + relation rules | 0.55 | 0.76 | 0.46 ± 0.01 |
| C | + graph structure | 0.54 | 0.82 | 0.52 ± 0.03 |
| D | + ownership | 0.86 | 1.00 | 0.51 ± 0.02 |

- In the complete pipeline the gain comes from finding cases: coverage rises from 0.09 to 1.00.
- On the fixed candidate set, relation rules and graph structure improve how the same candidates are ordered; ownership adds no further ordering gain there.
- The graph-structure level (C) finds two more planted cases but does not raise AP (0.55 → 0.54): on this data, the relation rules already find most of what it finds.

**2. Ownership, sponsor, address and approver edges carry most of the value** (`ablation`)

Each row removes one family of edges from the graph before detection. Entity resolution, which built the graph, is unchanged.

| Edges removed | AP drop |
|---|---|
| Ownership (REGISTERED_AS, IS_PERSON, OWNS, CONTROLS, CONSOLIDATED_BY) | 0.31 |
| Sponsor (SPONSORED_BY) | 0.27 |
| Address (LOCATED_AT) | 0.13 |
| Approver (APPROVED_BY) | 0.12 |
| Institution links (SERVES, SITE_OF, AFFILIATED_WITH) | 0.05 |
| Payments (activity on engagements) | 0.05 |
| Contacts (HAS_CONTACT) | none |

- Transliterated name matching to the register is worth about 0.16 AP. Matching on identifiers only gives AP 0.77 and R@100 0.85.

**3. Results are stable, except at a stricter matching threshold** (`stability`, `perturb`)

| Change | AP |
|---|---|
| Random seed (5 seeds) | 0.921–0.934 |
| Any single detector threshold 25% looser or stricter | 0.896–0.936 |
| All thresholds 25% looser at once | 0.920 |
| All thresholds 25% stricter at once | 0.876 |
| Any ratio-threshold quantile from 0.80 to 0.99 (one at a time) | 0.895–0.935 |
| 10–20% of name-based registry matches dropped (3 seeds) | 0.88 ± 0.04 |
| Typos in 10–20% of addresses (3 seeds) | 0.92–0.94 |
| Scorer threshold IoU ≥ 0.3 | 0.960 |
| Scorer threshold IoU ≥ 0.5 | 0.933 |
| Scorer threshold IoU > 0.5 (same as ≥ 0.6) | 0.826 |
| Scorer threshold IoU ≥ 0.7 | 0.787 |

- For quantile thresholds, "25% looser" means a 25% larger tail share (0.95 → 0.933).
- Most of the drop from "≥ 0.5" to "> 0.5" comes from one case. The case ranked 2nd holds 5 of the 10 members of CASE-0032, so its IoU is exactly 0.5. Under a strict threshold it becomes a false positive at rank 2, and a case further down matches CASE-0032 instead. The official threshold, and whether exactly 0.5 matches, is not published.
- The spike threshold is the most sensitive: at quantile 0.95 or above, one or two activity cases drop out of the top 50.

**4. Learned models do not beat the transparent detectors** (`fixed`, `embedding`)

On the same candidates, learned scorers reach at most about AP 0.53, against 0.933 for the detector ranking.

| Case representation | Logistic regression (cross-validated) | Isolation forest |
|---|---|---|
| Explicit features (level D) | 0.51 ± 0.02 | 0.27 |
| GraphSAGE embedding only | 0.30 ± 0.03 | 0.08 |
| Explicit + embedding | 0.52 ± 0.02 | 0.18 |
| Explicit + embedding, encoder fitted to 2024-06-30 and frozen | 0.53 ± 0.04 | 0.21 |

- Adding embeddings changes the logistic score by 0.01–0.03, within the seed variation, and the label-free scorer gets worse.
- The planted signals are counts in time windows relative to peers, and deep ownership paths. Explicit features encode these directly; a two-layer mean-aggregating GNN does not.
- With 33 positives there is too little to learn a ranker.
- Embeddings are therefore kept as a comparison (`meridian-exp embedding`), not used in production.
- Link-prediction AUC is 0.82–0.98, so the encoder itself trains properly.
- The frozen-encoder protocol for the main data already runs: fit on the training period, then apply to later snapshots.

**5. Coverage of unseen patterns is limited** (`loto`)

We removed each type's dedicated detector in turn and measured recall for that type.

- The open-set detectors recover part of SHARED_ADDRESS (0.25 → 0.75) and ACTIVITY_TIMING_CONCENTRATION (0.50 → 0.75), and none of the other types.
- Six types are fully recovered by other typed detectors anyway.
- Expect weak recall on patterns outside the published list in the blind test.

**6. How cases are formed and scored** (`stability`, assembly and ranking choices)

Each row switches one choice back on its own; everything else stays at the default.

| Choice | AP, IoU ≥ 0.5 | AP, IoU ≥ 0.7 | Mean IoU |
|---|---|---|---|
| Default | 0.933 | 0.787 | 0.93 |
| Corroboration counts shared evidence again | 0.908 | 0.792 | 0.93 |
| Percentile ranks also for detectors with 2–4 candidates | 0.909 | 0.778 | 0.93 |
| Fragment rule at 60% of members instead of 50% | 0.924 | 0.779 | 0.93 |
| Open-set candidates may widen a case | 0.933 | 0.787 | 0.91 |
| A candidate joins the first qualifying case, not the best-matching one | 0.933 | 0.787 | 0.93 |
| Single-vendor findings join a group that cites the same edges | 0.901 | 0.796 | 0.94 |

- Open-set communities usually reach past the group they overlap. When they may widen a case, CASE-0002 and CASE-0015 are matched at IoU 0.75 and 0.77; when they only corroborate it, both are matched exactly.
- Best-match assignment changes nothing on the starter data. It is kept because the result no longer depends on the order in which cases were formed.
- Joining single-vendor findings to a group on shared edges recovers CASE-0033 exactly (IoU 0.5 → 1.0). The joined case is then led by a common-owner finding, a frequent type, and type interleaving moves it from rank 14 to 39. It helps only under a stricter matching threshold (R@50 0.97 vs 0.94 at IoU ≥ 0.7), so it is available (`cases.evidence_union`) but off.

**7. The pipeline runs at the size of the main data** (`scale`)

The main data is about 20 times larger than the starter set. We replicated the starter data 20 times as independent copies: every identifier, name, address and contact is distinct per copy, and only the reference tables are shared.

| | 1 copy | 20 copies |
|---|---|---|
| Vendor records | 1,002 | 20,040 |
| Graph nodes / edges | 10,089 / 20,438 | 200,982 / 408,513 |
| Candidates / cases | 272 / 178 | 5,684 / 3,776 |
| Run time, graph to ranked list | 8 s | 195 s |

- Run time grows about linearly (10 s per copy at 20 copies), with peak memory under 3 GB. Per-copy output matches the single copy for every detector but three.
- Geography: the copies' onboarding bursts fall on the same dates in the same (shared) markets, so they merge into one window per market. This comes from the replication, not the method.
- Sponsor–approver pairs: a market's pool of eligible approvers grows with the copies, so more pairs pass the minimum of three eligible approvers. On the main data the pool will be the real number of approvers per market.
- Duplicate payments: the copy suffix lands after the legal form ("… S.R.L. Copy01"), so one pair per copy no longer matches by name.
- Community detection needs its resolution scaled with the graph. At a fixed resolution, modularity merges ever larger communities as the graph grows; on 20 copies it found 3 groups per copy instead of 70. The resolution is therefore stated per 1,000 vendor entities, and the per-copy output now matches.

## Thresholds

Thresholds that compare a vendor, sponsor or market with its peers are quantiles of each dataset's own distribution, so they are recomputed on the main data without labels. The quantile was set where the original threshold sat on the starter distribution, so starter results did not change. The other thresholds are fixed rules.

| Threshold | Setting | Kind | Peer group or source |
|---|---|---|---|
| Sponsor: new vendors in 60 days, relative to the role-peer median | top 5% of sponsors (3.0× on the starter data) | quantile | sponsors with the same role; all sponsors when the role has fewer than 5 others |
| Geography: new intermediaries in 365 days, relative to the market's earlier yearly rate | top 10% of market windows (3.0×) | quantile | all windows of all markets |
| Spend spike: a month's spend relative to the vendor's own local median | top 15% of spend peaks (2.5×) | quantile | every vendor's local spend maxima |
| Entity outlier: how far a vendor sits in the tail of its one or two most extreme features (record attributes, relation degrees, ownership structure) | top 2% of vendor entities (at least 10) | quantile | all vendor entities |
| Shared address: vendor entities at one address at the same time | ≥ 3 | fixed | Dossier worked example ("at least three active vendors at a rare address") |
| Common owner | ≥ 2 entities | fixed | definition of a common owner |
| Payments after termination | ≥ 1 month | fixed | definition |
| FMV cap | annual cap of the rate card | fixed | reference table `fmv_rate_cards` |
| Sponsor–approver pair: shared engagements, share of the sponsor's engagements, eligible approvers | ≥ 5, ≥ 50%, ≥ 3 | fixed | minimum evidence; the pair's score is a binomial tail probability against the approver's share of the market pool |
| Windows: sponsor 60 days, geography 365 days, rapid onboarding 14 days, ownership change 90 days, spend spikes 2 months | — | fixed | typology descriptions of the sponsor's cases |
| Group sizes: new vendors ≥ 4 (sponsor, geography), linked vendors ≥ 4 (rapid onboarding), spiking vendors ≥ 3, cluster size ≥ 4 | — | fixed | minimum group size for a reviewable pattern |
| Dense cluster: vendor pairs linked on ≥ 4 relation types; ≥ 4 types each held by ≥ 80% of the members | — | fixed | typology description ("overlap on several relationship types") |
| Bridge: links to institutions in another country | ≥ 2 | fixed | definition of a bridge between two communities |
| Layered ownership | ≥ 4 layers, or ≥ 3 layers across ≥ 3 jurisdictions | fixed | typology description |
| Ownership change: further changes within the window | ≥ 2 | fixed | typology description |
| Open-set communities: size 3–15, ≥ 2 relation types each held by ≥ 60% | — | fixed | reviewable group size |

The sensitivity of every threshold is in finding 3 and in `experiments/stability.md`.

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
| Label-free ranking: equal detector priors, 0.3 for open-set detectors | Cross-fitted calibration on the starter labels scored lower (AP 0.835 vs 0.933); revisit with the validation labels |
| Other detectors corroborate a case only with evidence edges not already counted | Detectors built on the same finding (common owner and ownership overlap; a layered chain and the opaque routing through it) otherwise vote twice; AP 0.908 when shared evidence counts again |
| Detectors with fewer than 5 candidates rank them at the 50th percentile | With one or two candidates a percentile reflects pool size, not strength: a lone candidate would get the top rank automatically |
| Benign explanations down-weight only the signal they explain | A coworking address does not explain an ownership convergence; case-level down-weighting pushed planted multi-relation clusters such as CASE-0009 out of the top 100 |
| Merge candidates only if their active periods meet | Keeps CASE-0023 and CASE-0030 apart; keeps an owner group and its later address overlap together |
| Overlapping groups from different detectors merge into their union (capped at 1.5× the lead) | Multi-typology planted cases are unions across relations (AP 0.858 without unions); the cap stops groups chaining through a shared hub |
| Open-set candidates corroborate a case but never widen it | Communities reach past the typed group they overlap (mean IoU 0.91 when they widen cases, 0.93 when not) |
| A candidate joins the case it overlaps most | The result must not depend on the order in which cases were formed |
| One shared-address candidate per co-location episode | A second group at the same address in a later period is a separate pattern; on the starter data every address has one episode |
| Down-weight a case when half of its members are already in a higher-ranked case | Fragments of one scheme were the largest group of false positives and cannot match a second case; at 60%, a two-vendor case sharing one vendor slipped through (AP 0.924) |
| Groups sharing only process relations (sponsor, approver, category, institution) are down-weighted | Shared identity items (address, phone, domain, owner) separate planted clusters from standard local processes |
| Same-day batch entry and the first three months of spend are not bursts | Acquisition integration loads vendors in batches; new vendors ramp up together |
| Peer-relative thresholds are quantiles of each dataset's distribution | Fixed ratios were read off the starter case descriptions; quantiles carry over to the main data without its labels |
| Community resolution is stated per 1,000 vendor entities | At a fixed resolution, modularity finds ever larger communities as the graph grows |
| Submit tags within 85% of the strongest evidence | Exact tag match 0.67 vs 0.39 when all tags are kept; keeping only the strongest tag gives 0.76 but misses correct tags; the official tag-accuracy definition is unknown |
| Offshore lists: IMF OFC list and Garcia-Bernardo et al. (2017) sink/conduit OFCs; conduit-only routes without a PEP are not reported | FSI and FATF are not in the bundle; conduit holdcos are common and legitimate |
| FMV country bands: Europe, JP, CA, AU, NZ → EU_JP; US → US; rest → EMERGING | The bundle gives no mapping; to be confirmed with the sponsor |
| Policy findings (FMV breaches) are ranked with the cases under `POLICY_VIOLATION` | The sponsor scores them separately; they count as false positives against the starter labels |

## Limitations and open items

- Every number above is on the development set. The main data is about 20 times larger, with a lower share of planted cases and more benign look-alikes, so expect more false positives near the top.
- Two matched cases sit at or near the matching threshold (CASE-0032 at IoU 0.5, CASE-0033 at 0.5). A stricter official threshold would cost about 0.1 AP.
- Emergence-date rules cannot be checked until the validation labels arrive.
- Still to do: interactive case browsing and a small review exercise (Dossier §13.3).
- Questions for the sponsor:
  - the official matching threshold and scoring script, including whether IoU exactly 0.5 matches;
  - how tag accuracy is computed for multi-tag cases, and whether tags outside the 14 (`OPEN_SET`, `POLICY_VIOLATION`) are accepted;
  - how single-record policy findings are scored;
  - the FMV country-band mapping;
  - for the validation run, whether detection should see only data up to the end of the validation period;
  - the release dates for the main data and the validation labels.
