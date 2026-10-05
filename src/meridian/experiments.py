"""Experiments: representation ladder, relation ablation, stability, leave one type out, embeddings, scale.

    meridian-exp ladder | fixed | ablation | stability | perturb | loto | embedding | scale | all
                 [--out reports/experiments] [--copies 20]

Two ladder experiments are reported separately:
  ladder  complete pipeline: each level runs only the detectors it is allowed to use
  fixed   fixed candidate set: the same label-free candidates, scored by the same
          downstream model on case features of increasing relational content

On the starter data every number is a development-set description.
"""
from __future__ import annotations

import argparse
import copy
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .cases import assemble
from .data import as_of, canonical, load_bundle
from .run import ROOT, load_config, pipeline
from .scorer import ScorerConfig, load_labels, overlap, prepare_submission, score

LEVEL_A = ["fmv_cap", "post_termination", "record_outlier"]
LEVEL_B = LEVEL_A + ["shared_address", "sponsor_concentration", "sponsor_approver_pair", "rapid_onboarding",
                     "bridge", "geography", "sync_spikes", "entity_change_point"]
LEVEL_C = LEVEL_B + ["dense_cluster", "rare_sharing_community", "duplicate_paid", "entity_outlier"]
LEVEL_D = LEVEL_C + ["common_owner", "layered_ownership", "opaque_routing", "ownership_change"]
LADDER = {"A": (LEVEL_A, False), "B": (LEVEL_B, False), "C": (LEVEL_C, False), "D": (LEVEL_D, True)}

METRICS = ["P@10", "P@25", "R@25", "R@50", "R@100", "AP", "mean_iou", "macro_tag_recall", "worst_tag_recall",
           "fp_top50", "vendor_AP", "coverage"]


def _cfg(base: dict, **overrides) -> dict:
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def vendor_ap(sub: pd.DataFrame, truth: pd.DataFrame, all_records) -> float:
    """Vendor-level AP: a record's score is 1/rank of the best case containing it."""
    best = {}
    for r in sub.itertuples():
        for m in r.member_record_ids.split("|"):
            best.setdefault(m, 1.0 / r.case_rank)
    pos = set().union(*truth["members"])
    y = np.array([r in pos for r in all_records])
    s = np.array([best.get(r, 0.0) for r in all_records])
    return float(average_precision_score(y, s))


def evaluate(sub, cands, truth, bundle, scfg: ScorerConfig) -> dict:
    res = score(prepare_submission(sub[["case_rank", "member_record_ids", "typology_tags", "score",
                                        "reason_codes"]]), truth, scfg)
    cov = np.mean([max((overlap(c.members, t.members, scfg.measure) for c in cands), default=0) >= scfg.threshold
                   for t in truth.itertuples()])
    out = {k: res[k] for k in METRICS if k in res}
    out["vendor_AP"] = vendor_ap(sub, truth, bundle.third_parties["record_id"])
    out["coverage"] = float(cov)
    out["n_cases"] = len(sub)
    return out


def run_eval(bundle, cfg, truth, scfg) -> dict:
    sub, cases, cands, ctx, _ = pipeline(cfg, bundle)
    return evaluate(sub, cands, truth, bundle, scfg)


# ---------------------------------------------------------------------------
def ladder_full(bundle, base, truth, scfg) -> pd.DataFrame:
    rows = []
    for level, (dets, own) in LADDER.items():
        cfg = _cfg(base, detectors=dets, **{"features.ownership": own})
        rows.append({"level": level, **run_eval(bundle, cfg, truth, scfg)})
    rows.append({"level": "production (default: typed + open-set, no record_outlier)", **run_eval(bundle, base, truth, scfg)})
    return pd.DataFrame(rows)


def _oof_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, k: int = 5) -> np.ndarray:
    """Out-of-fold logistic scores; folds are grouped so a planted case never spans train and test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    folds = np.array([fold_of[g] for g in groups])
    out = np.zeros(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        if y[tr].sum() == 0 or (~y[tr].astype(bool)).sum() == 0:
            out[te] = y[tr].mean()
            continue
        sc = StandardScaler().fit(x[tr])
        m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(sc.transform(x[tr]), y[tr])
        out[te] = m.predict_proba(sc.transform(x[te]))[:, 1]
    return out


def _fixed_candidates(bundle, base, truth, scfg):
    """Label-free candidates from the full detector set, with their match status (for scoring only)."""
    cfg = _cfg(base, detectors=LEVEL_D)
    _, cases, cands, ctx, _ = pipeline(cfg, bundle)
    cases = assemble(cands, **cfg.get("cases", {}))  # full set, no top-k cut
    member_sets = [c.members for c in cases]
    matched = []
    for m in member_sets:
        best = max(truth.itertuples(), key=lambda t: overlap(m, t.members, scfg.measure))
        matched.append(best.case_id if overlap(m, best.members, scfg.measure) >= scfg.threshold else None)
    y = np.array([m is not None for m in matched], dtype=int)
    groups = np.array([m if m is not None else f"neg{i}" for i, m in enumerate(matched)])
    return cases, cands, ctx, member_sets, y, groups


def _score_both(x, y, groups, seed, cases, cands, truth, bundle, scfg) -> list[dict]:
    """Rank the fixed candidates with the two downstream scorers and evaluate."""
    xs = StandardScaler().fit_transform(x)
    scorers = {
        "isolation_forest": -IsolationForest(n_estimators=300, random_state=seed).fit(xs).score_samples(xs),
        "logistic_oof": _oof_logistic(x, y, groups, seed),
    }
    rows = []
    for name, s in scorers.items():
        order = np.argsort(-s, kind="stable")[:100]
        sub = pd.DataFrame({
            "case_rank": range(1, len(order) + 1),
            "member_record_ids": ["|".join(sorted(cases[i].members)) for i in order],
            "typology_tags": ["|".join(cases[i].tags) for i in order],
            "score": s[order], "reason_codes": "",
        })
        rows.append({"scorer": name, "n_features": x.shape[1], **evaluate(sub, cands, truth, bundle, scfg)})
    return rows


def _summarise(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.groupby(["scorer", "level"], sort=False).agg(
        n_features=("n_features", "first"), AP=("AP", "mean"), AP_sd=("AP", "std"), R50=("R@50", "mean"),
        P25=("P@25", "mean"), vendor_AP=("vendor_AP", "mean"), fp_top50=("fp_top50", "mean")).reset_index()


def ladder_fixed(bundle, base, truth, scfg, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Same candidates, same downstream scorer, features of increasing relational content."""
    from .features import CaseFeaturizer

    cases, cands, ctx, member_sets, y, groups = _fixed_candidates(bundle, base, truth, scfg)
    rows = []
    for seed in seeds:
        fz = CaseFeaturizer(ctx, seed=seed)
        for level in ("A", "B", "C", "D"):
            x = fz.matrix(member_sets, level).to_numpy()
            for r in _score_both(x, y, groups, seed, cases, cands, truth, bundle, scfg):
                rows.append({"seed": seed, "level": level, **r})
    return _summarise(rows)


def embedding_comparison(bundle, base, truth, scfg, seeds=(0, 1, 2)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit features vs graph embeddings vs both, same candidates and scorers."""
    from .embeddings import EmbeddingCaseFeatures, vendor_embeddings
    from .features import CaseFeaturizer

    from .context import Context
    from .data import as_of

    cases, cands, ctx, member_sets, y, groups = _fixed_candidates(bundle, base, truth, scfg)
    # Main-data protocol: fit on the training period only, apply frozen.
    fit_ctx = Context(as_of(bundle, "2024-06-30"), ctx.cfg)
    rows, lp = [], []
    for seed in seeds:
        emb_frozen, _ = vendor_embeddings(ctx, seed=seed, fit_ctx=fit_ctx)
        emb, report = vendor_embeddings(ctx, seed=seed)
        for part in ("val", "test"):
            for rel, auc in report[part].items():
                lp.append({"seed": seed, "split": part, "relation": rel, "AUC": auc})
        explicit = CaseFeaturizer(ctx, seed=seed).matrix(member_sets, "D")
        embedded = EmbeddingCaseFeatures(emb, seed=seed).matrix(member_sets)
        frozen = EmbeddingCaseFeatures(emb_frozen, seed=seed).matrix(member_sets)
        for level, x in (("D explicit", explicit), ("E embedding only", embedded),
                         ("D + E", pd.concat([explicit, embedded], axis=1)),
                         ("E only, encoder fitted to 2024-06-30", frozen),
                         ("D + E, encoder fitted to 2024-06-30", pd.concat([explicit, frozen], axis=1))):
            for r in _score_both(x.to_numpy(), y, groups, seed, cases, cands, truth, bundle, scfg):
                rows.append({"seed": seed, "level": level, **r})
    lp = pd.DataFrame(lp).groupby(["split", "relation"])["AUC"].agg(["mean", "std"]).round(3).reset_index()
    return _summarise(rows), lp


# ---------------------------------------------------------------------------
# Relation ablation removes edge types from the canonical graph before detection.
# Entity resolution, which built the graph, is left as it is, so each row measures
# what detection gains from that relation.
ABLATION_GROUPS = {
    "address (LOCATED_AT)": ["LOCATED_AT"],
    "contact (HAS_CONTACT)": ["HAS_CONTACT"],
    "sponsor (SPONSORED_BY)": ["SPONSORED_BY"],
    "approver (APPROVED_BY)": ["APPROVED_BY"],
    "institution (SERVES, SITE_OF, AFFILIATED_WITH)": ["SERVES", "SITE_OF", "AFFILIATED_WITH"],
    "ownership (REGISTERED_AS, IS_PERSON, OWNS, CONTROLS, CONSOLIDATED_BY)":
        ["REGISTERED_AS", "IS_PERSON", "OWNS", "CONTROLS", "CONSOLIDATED_BY"],
}


def ablation(bundle, base, truth, scfg) -> pd.DataFrame:
    full = run_eval(bundle, base, truth, scfg)
    rows = [{"removed": "(none)", **full}]
    for name, rels in ABLATION_GROUPS.items():
        rows.append({"removed": name, **run_eval(bundle, _cfg(base, **{"graph.drop_relations": rels}), truth, scfg)})
    rows.append({"removed": "payments (activity on engagements)",
                 **run_eval(bundle, _cfg(base, **{"graph.drop_activity": True}), truth, scfg)})
    df = pd.DataFrame(rows)
    for k in ["AP", "R@50", "macro_tag_recall"]:
        df[f"d_{k}"] = df[k] - full[k]
    return df


# ---------------------------------------------------------------------------
THRESHOLD_PARAMS = {
    "shared_address.min_entities": 3, "sponsor.min_new": 4, "sponsor.min_ratio": 3.0, "approval.min_pair": 5,
    "approval.min_sponsor_share": 0.5, "dense_cluster.min_covered_types": 4, "rapid_onboarding.min_linked": 4,
    "geography.min_new": 4, "geography.min_ratio": 3.0, "activity.min_spike_ratio": 2.5,
    "ownership_change.min_signals": 2, "layered.min_layers": 4,
}


def stability(bundle, base, truth) -> pd.DataFrame:
    rows = []
    for meas in ("iou", "overlap_coef"):
        for th in (0.3, 0.5, 0.7):
            s = ScorerConfig(meas, th)
            rows.append({"experiment": "scorer threshold", "setting": f"{meas}>={th}", **run_eval(bundle, base, truth, s)})
    scfg = ScorerConfig(**base.get("scorer", {}))
    for factor in (0.75, 1.25):
        over = {}
        for key, v in THRESHOLD_PARAMS.items():
            nv = v * factor
            over[f"params.{key}"] = int(round(nv)) if isinstance(v, int) else nv
        rows.append({"experiment": "all detector thresholds", "setting": f"x{factor}",
                     **run_eval(bundle, _cfg(base, **over), truth, scfg)})
    for key, v in THRESHOLD_PARAMS.items():
        for factor in (0.75, 1.25):
            nv = v * factor
            nv = int(round(nv)) if isinstance(v, int) else nv
            if nv == v:
                continue
            rows.append({"experiment": "one threshold", "setting": f"{key}={nv}",
                         **run_eval(bundle, _cfg(base, **{f"params.{key}": nv}), truth, scfg)})
    for name, over in {"strict: identifiers only": {"resolution.name_merge": "none", "resolution.registry_name_match": False},
                       "default": {},
                       "loose: names alone, accept ambiguous": {"resolution.name_merge": "any",
                                                                "resolution.accept_candidates": True}}.items():
        rows.append({"experiment": "entity resolution", "setting": name, **run_eval(bundle, _cfg(base, **over), truth, scfg)})
    for merge in (0.6, 0.8, 1.01):
        rows.append({"experiment": "case merge IoU", "setting": f"merge_iou={merge}",
                     **run_eval(bundle, _cfg(base, **{"cases.merge_iou": merge}), truth, scfg)})
    for union in (0.4, 0.5, 1.01):
        rows.append({"experiment": "case union IoU", "setting": f"union_iou={union}",
                     **run_eval(bundle, _cfg(base, **{"cases.union_iou": union}), truth, scfg)})
    for inter in (False, True):
        rows.append({"experiment": "type interleaving", "setting": str(inter),
                     **run_eval(bundle, _cfg(base, **{"ranking.interleave": inter}), truth, scfg)})
    # Random seeds (Louvain communities, isolation forests); the base config uses seed 0.
    for seed in (0, 1, 2, 3, 4):
        rows.append({"experiment": "random seed", "setting": f"seed={seed}",
                     **run_eval(bundle, _cfg(base, seed=seed), truth, scfg)})
    return pd.DataFrame(rows)


def _typo(s: str, rng) -> str:
    if len(s) < 4:
        return s
    i = int(rng.integers(0, len(s)))
    op = rng.integers(0, 3)
    c = chr(int(rng.integers(97, 123)))
    return s[:i] + s[i + 1:] if op == 0 else (s[:i] + c + s[i + 1:] if op == 1 else s[:i] + c + s[i:])


# ---------------------------------------------------------------------------
# Scale: the starter data replicated k times as independent copies
_ID_COLUMNS = {"record_id", "tax_id", "lei", "registry_no", "engagement_id", "sponsor_employee_id", "approval_id",
               "approver_employee_id", "link_id", "institution_id", "employee_id", "delegation_id",
               "from_employee_id", "to_employee_id", "child_lei", "parent_lei", "statement_id", "reg_record_id",
               "subject_sid", "party_sid"}
_NAME_COLUMNS = {"legal_name", "declared_parent_group", "name", "full_name", "child_legal_name", "parent_legal_name"}
_REFERENCE_TABLES = {"markets", "paying_entities", "service_categories", "approval_policy_matrix", "fmv_rate_cards",
                     "country_risk_context", "business_events"}
_EXTENSION = re.compile(r"^(.*?)(\s*(?:x|ext\.?)\s*\d+)?$", re.I)


def _copy_table(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """Copy i of a business table, with every identifier, name, address and contact made distinct."""
    if i == 0:
        return df
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        m = df[c].notna()
        v = df.loc[m, c].astype(str)
        if c in _ID_COLUMNS:
            df.loc[m, c] = v + f"R{i:02d}"
        elif c in _NAME_COLUMNS:
            df.loc[m, c] = v + f" Copy{i:02d}"
        elif c == "raw_address":
            df.loc[m, c] = v + f", Building {i:02d}"
        elif c == "phone":  # before any extension, which phone normalisation drops
            df.loc[m, c] = [_EXTENSION.sub(lambda g: g.group(1) + f"{i:02d}" + (g.group(2) or ""), p) for p in v]
        elif c == "email_domain":
            df.loc[m, c] = f"r{i:02d}-" + v
    return df


def replicate(bundle, copies: int):
    """The bundle replicated `copies` times as independent copies sharing only the reference tables."""
    return replace(bundle, **{name: canonical(name, pd.concat([_copy_table(df, i) for i in range(copies)],
                                                              ignore_index=True))
                              for name, df in bundle.tables().items() if name not in _REFERENCE_TABLES})


def _timed_run(bundle, cfg: dict) -> tuple[dict, dict, dict]:
    """Seconds per stage, candidates per detector and graph size for one pipeline run."""
    from .context import Context
    from .detectors import DETECTORS, OPEN_SET_DETECTORS
    from .graph import build_graph
    from .ranking import rank
    secs, n = {}, {}
    t = time.perf_counter()
    view = as_of(bundle, cfg["snapshot"])
    graph = build_graph(Context(view, cfg))
    secs["entity resolution and graph"] = time.perf_counter() - t
    t = time.perf_counter()
    ctx = Context.from_graph(graph, view, cfg)
    secs["graph views"] = time.perf_counter() - t
    cands = []
    for name, fn in {**DETECTORS, **OPEN_SET_DETECTORS}.items():
        t = time.perf_counter()
        out = fn(ctx)
        secs[f"detector {name}"] = time.perf_counter() - t
        cands += out
    for c in cands:
        n[c.detector] = n.get(c.detector, 0) + 1
    t = time.perf_counter()
    cases = assemble(cands, **cfg.get("cases", {}))
    secs["case assembly"] = time.perf_counter() - t
    t = time.perf_counter()
    rank(cases, cfg.get("ranking", {}))
    secs["ranking"] = time.perf_counter() - t
    size = {"vendor records": len(bundle.third_parties), "graph nodes": len(graph.nodes),
            "graph edges": len(graph.edges), "candidates": len(cands), "cases": len(cases)}
    return secs, n, size


def scale(bundle, base, copies: int = 20) -> pd.DataFrame:
    """Run time and detector output on the starter data and on `copies` independent copies of it.

    If the pipeline scales, every count per copy matches the single copy and run time
    grows about linearly. Per-copy counts that differ show where a detector depends on
    the size of the data (for example a fixed community resolution or a peer pool that
    grows with the data)."""
    rows = []
    s1, n1, z1 = _timed_run(bundle, base)
    sk, nk, zk = _timed_run(replicate(bundle, copies), base)
    k = f"{copies} copies"
    for m in z1:
        rows.append({"measure": m, "1 copy": f"{z1[m]:,}", k: f"{zk[m]:,}", "per copy": f"{zk[m] / copies:,.1f}"})
    for d in sorted(set(n1) | set(nk)):
        rows.append({"measure": f"candidates: {d}", "1 copy": str(n1.get(d, 0)), k: str(nk.get(d, 0)),
                     "per copy": f"{nk.get(d, 0) / copies:.2f}"})
    s1["total"], sk["total"] = sum(s1.values()), sum(sk.values())
    for m in sorted(s1, key=lambda m: (m == "total", -sk[m])):
        rows.append({"measure": f"seconds: {m}", "1 copy": f"{s1[m]:.1f}", k: f"{sk[m]:.1f}",
                     "per copy": f"{sk[m] / copies:.2f}"})
    return pd.DataFrame(rows)


def perturb(bundle, base, truth, scfg, seeds=(0, 1, 2)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for p in (0.1, 0.2):
            rng = np.random.default_rng(seed)
            a = bundle.tp_address_history.copy()
            hit = rng.random(len(a)) < p
            a.loc[hit, "raw_address"] = [_typo(s, rng) for s in a.loc[hit, "raw_address"]]
            rows.append({"perturbation": f"address typo {p:.0%}", "seed": seed,
                         **run_eval(replace(bundle, tp_address_history=a), base, truth, scfg)})
            cfg = _cfg(base, **{"resolution.drop_name_matches": p, "resolution.perturb_seed": seed})
            rows.append({"perturbation": f"drop {p:.0%} of name-based registry matches", "seed": seed,
                         **run_eval(bundle, cfg, truth, scfg)})
    df = pd.DataFrame(rows)
    return df.groupby("perturbation")[["AP", "R@50", "R@100", "macro_tag_recall", "mean_iou", "coverage"]].agg(
        ["mean", "std"]).round(3)


# ---------------------------------------------------------------------------
def leave_one_type_out(bundle, base: dict, truth: pd.DataFrame, scfg) -> pd.DataFrame:
    """For each tag, drop every detector whose primary tag it is, with and without open-set detectors.

    Recall is measured on the planted cases carrying that tag (at the scorer
    threshold, top 100). Multi-tag cases can still be found by the other
    typed detectors; the open-set gain is the difference between the last two columns.
    """
    from .detectors import DETECTORS, OPEN_SET_DETECTORS
    from .detectors.base import TAGS

    primary = {
        "SHARED_ADDRESS": ["shared_address"], "SPONSOR_CONCENTRATION": ["sponsor_concentration"],
        "APPROVER_CONCENTRATION": ["sponsor_approver_pair"], "SPONSOR_APPROVER_COUPLING": ["sponsor_approver_pair"],
        "DENSE_MULTI_RELATION_CLUSTER": ["dense_cluster"], "RAPID_ONBOARDING_LINKED": ["rapid_onboarding"],
        "BRIDGE_ENTITY": ["bridge"], "GEOGRAPHY_LINKED_CONCENTRATION": ["geography"],
        "ACTIVITY_TIMING_CONCENTRATION": ["sync_spikes", "post_termination"],
        "COMMON_BENEFICIAL_OWNER": ["common_owner", "ownership_overlap"],
        "LAYERED_COMPLEX_OWNERSHIP": ["layered_ownership"], "OPAQUE_JURISDICTION_ROUTING": ["opaque_routing"],
        "OWNERSHIP_OPERATIONAL_OVERLAP": ["ownership_overlap"],
        "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT": ["ownership_change"],
    }
    typed = list(DETECTORS)
    open_names = list(OPEN_SET_DETECTORS)  # priors for these come from base ranking.detector_priors

    def recall(cfg, tag):
        sub = pipeline(cfg, bundle)[0]
        res = score(prepare_submission(sub[["case_rank", "member_record_ids", "typology_tags", "score",
                                            "reason_codes"]]), truth, scfg)
        return res["recall_by_tag"].get(tag, np.nan), res["AP"]

    rows = []
    for tag in TAGS:
        drop = primary[tag]
        n = int(truth["tags"].map(lambda ts: tag in ts).sum())
        cfg_all = copy.deepcopy(base)
        cfg_all["detectors"] = typed + open_names
        cfg_out = copy.deepcopy(base)
        cfg_out["detectors"] = typed
        cfg_out["exclude_detectors"] = drop
        cfg_open = copy.deepcopy(cfg_all)
        cfg_open["exclude_detectors"] = drop
        r_all, _ = recall(cfg_all, tag)
        r_out, ap_out = recall(cfg_out, tag)
        r_open, ap_open = recall(cfg_open, tag)
        rows.append({"held_out": tag, "n_cases": n, "removed": ",".join(drop), "recall_all": r_all,
                     "recall_without_type": r_out, "recall_without_type_plus_open_set": r_open,
                     "open_set_gain": r_open - r_out, "AP_without": ap_out, "AP_without_plus_open": ap_open})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def to_md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" ".join(c).strip() for c in d.columns]
        d = d.reset_index()
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in d.itertuples(index=False):
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["ladder", "fixed", "ablation", "stability", "perturb", "loto", "embedding",
                                           "scale", "all"])
    ap.add_argument("--copies", type=int, default=20, help="copies of the data for the scale experiment")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "experiments"))
    a = ap.parse_args(argv)
    base = load_config(a.config)
    bundle = load_bundle(ROOT / base["bundle"])
    truth = load_labels(ROOT / base["labels"])
    scfg = ScorerConfig(**base.get("scorer", {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = ["ladder", "fixed", "ablation", "stability", "perturb", "loto"] if a.experiment == "all" else [a.experiment]
    if a.experiment == "all":
        from .embeddings import HAVE_TORCH
        if HAVE_TORCH:
            todo.append("embedding")
        else:
            print("torch/torch_geometric not installed: skipping the embedding comparison")
    for name in todo:
        if name == "ladder":
            df = ladder_full(bundle, base, truth, scfg)
        elif name == "fixed":
            df = ladder_fixed(bundle, base, truth, scfg)
        elif name == "ablation":
            df = ablation(bundle, base, truth, scfg)
        elif name == "stability":
            df = stability(bundle, base, truth)
        elif name == "perturb":
            df = perturb(bundle, base, truth, scfg)
        elif name == "embedding":
            df, lp = embedding_comparison(bundle, base, truth, scfg)
            (out / "embedding_linkpred.md").write_text(to_md(lp))
            print(f"== link prediction (training check only)\n{to_md(lp)}\n")
        elif name == "loto":
            df = leave_one_type_out(bundle, base, truth, scfg)
        elif name == "scale":
            df = scale(bundle, base, a.copies)
        (out / f"{name}.csv").write_text(df.to_csv(index=not isinstance(df.index, pd.RangeIndex)))
        (out / f"{name}.md").write_text(to_md(df))
        print(f"== {name}\n{to_md(df)}\n")


if __name__ == "__main__":
    main()
