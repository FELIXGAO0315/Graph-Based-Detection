"""Optional: heterogeneous graph embeddings.

Design:
  - PyG HeteroData; one attribute encoder (Linear) per node type;
  - two-layer, relation-specific GraphSAGE (to_hetero), 64 dimensions;
  - typed reverse edges, so employee and owner information reaches vendors;
  - no learnable per-node ID vectors, so the encoder applies to new nodes;
  - no case IDs, row numbers or label text anywhere.

Engagements are folded into vendor-employee (sponsored_by, approved_by) and
vendor-category edges: with two layers a vendor then sees the other vendors
of its sponsor, which is where most planted signals live.

Training: relation-typed link prediction with a DistMult decoder,
negatives drawn only among legal node types; supervision edges are held out
of message passing together with their reverse edges (RandomLinkSplit with
disjoint_train_ratio). Link-prediction AUC is only a training check. On the
main data the encoder should be fitted on the training period (`fit_as_of`)
and applied frozen to later snapshots; the starter has no periods, so it is
fitted at the snapshot without labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .context import Context

try:  # optional dependency (pip install torch from the CPU index, then torch_geometric)
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    import torch_geometric.transforms as T
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

ROLES = ["brand_manager", "medical_lead", "market_access_lead", "clinical_ops", "supply_buyer", "other_sponsor",
         "market_gm", "category_manager", "procurement_ss", "compliance_reviewer", "finance_approver_l1",
         "finance_approver_l2", "finance_approver_l3", "regional_finance_l3", "system_default"]
ADDR_TYPES = ["commercial", "residential", "coworking", "corporate_hq", "hospital_campus", "registered_agent",
              "virtual_office", "po_box", "industrial"]
PARENTS = ["COM", "HCP", "EVT", "MED", "RND", "SCM", "PAT", "GNA"]
FLAGS = ["hcp", "gov", "int", "flow"]
INST_TYPES = ["hospital", "academic_medical_center", "clinic", "health_authority", "patient_organization"]
SUBTYPES = ["operating", "holdco", "operating_parent", "group_parent", "fund", "trust"]

# Vendor-centric relations predicted during training; reverse edges are added by ToUndirected.
TARGETS = [("vendor", "located_at", "address"), ("vendor", "has_contact", "contact"),
           ("vendor", "sponsored_by", "employee"), ("vendor", "approved_by", "employee"),
           ("vendor", "provides", "category"), ("vendor", "linked_to", "institution"),
           ("vendor", "registered_as", "reg_entity"), ("reg_entity", "owned_by", "reg_entity"),
           ("reg_entity", "owned_by_person", "person")]


def _onehot(values, vocab) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab) + 1), dtype=np.float32)
    for r, v in enumerate(values):
        out[r, idx.get(v, len(vocab))] = 1.0
    return out


@dataclass
class GraphTables:
    """Node ids and raw features per type, edges as id pairs. Built from one Context."""
    nodes: dict
    feats: dict
    edges: dict


def graph_tables(ctx: Context, use_ownership: bool = True) -> GraphTables:
    from .detectors.ownership import OFFSHORE_CONDUIT, OFFSHORE_SINK
    from .features import record_features

    nodes, feats, edges = {}, {}, {}
    rf = record_features(ctx)
    nodes["vendor"] = list(rf.index)
    feats["vendor"] = rf.to_numpy(np.float32)

    a = ctx.addresses
    at = a.groupby("addr_key")["address_type"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
    nodes["address"] = list(at.index)
    feats["address"] = _onehot(list(at), ADDR_TYPES)
    edges[("vendor", "located_at", "address")] = a[["record_id", "addr_key"]].drop_duplicates().to_numpy()

    c = ctx.contacts
    ph = c.dropna(subset=["phone_key"])
    dm = c.dropna(subset=["domain_key"])
    contact_ids = [f"phone:{x}" for x in ph["phone_key"].unique()] + [f"domain:{x}" for x in dm["domain_key"].unique()]
    nodes["contact"] = contact_ids
    feats["contact"] = np.array([[1.0, 0.0] if i.startswith("phone:") else [0.0, 1.0] for i in contact_ids], np.float32)
    edges[("vendor", "has_contact", "contact")] = np.concatenate([
        np.c_[ph["record_id"], "phone:" + ph["phone_key"]], np.c_[dm["record_id"], "domain:" + dm["domain_key"]]])

    roles = ctx.roles.set_index("employee_id")
    nodes["employee"] = list(roles.index)
    scope = roles["scope_markets"].fillna("")
    feats["employee"] = np.c_[_onehot(list(roles["role"]), ROLES), roles["approver_level"].to_numpy(np.float32) / 3,
                              (scope == "*").to_numpy(np.float32), scope.str.contains("R:|S:").to_numpy(np.float32)]
    e = ctx.engagements
    edges[("vendor", "sponsored_by", "employee")] = e.dropna(subset=["sponsor_employee_id"])[
        ["record_id", "sponsor_employee_id"]].drop_duplicates().to_numpy()
    ap = ctx.approvals
    fin = ap[(ap["approval_type"] == "financial") & ap["approver_employee_id"].notna()]
    edges[("vendor", "approved_by", "employee")] = fin[["record_id", "approver_employee_id"]].drop_duplicates().to_numpy()

    cats = ctx.categories
    nodes["category"] = list(cats.index)
    feats["category"] = np.c_[_onehot(list(cats["parent"]), PARENTS),
                              np.array([[f in fs for f in FLAGS] for fs in cats["flag_set"]], np.float32)]
    prov = e.dropna(subset=["cat_primary"])[["record_id", "cat_primary"]]
    edges[("vendor", "provides", "category")] = prov.drop_duplicates().to_numpy()

    inst = ctx.b.healthcare_institutions.set_index("institution_id")
    nodes["institution"] = list(inst.index)
    feats["institution"] = np.c_[_onehot(list(inst["inst_type"]), INST_TYPES), inst["public_sector"].to_numpy(np.float32)]
    edges[("vendor", "linked_to", "institution")] = ctx.links[["record_id", "institution_id"]].drop_duplicates().to_numpy()

    if use_ownership:
        ents = ctx.b.reg_entities.set_index("statement_id")
        nodes["reg_entity"] = list(ents.index)
        j = ents["jurisdiction"]
        feats["reg_entity"] = np.c_[_onehot(list(ents["entity_subtype"]), SUBTYPES), j.isin(OFFSHORE_SINK).to_numpy(np.float32),
                                    j.isin(OFFSHORE_CONDUIT).to_numpy(np.float32), ents["listed"].fillna(False).to_numpy(np.float32)]
        pers = ctx.b.reg_persons.set_index("statement_id")
        nodes["person"] = list(pers.index)
        feats["person"] = np.c_[(pers["pep_status"] == "isPep").to_numpy(np.float32),
                                (pers["pep_status"] == "unknown").to_numpy(np.float32)]
        edges[("vendor", "registered_as", "reg_entity")] = np.array(list(ctx.reg_entity_of.items()), dtype=object).reshape(-1, 2)
        ri = ctx.b.reg_interests.dropna(subset=["party_sid"])  # all interests known by t, current and ended
        edges[("reg_entity", "owned_by", "reg_entity")] = ri[ri["party_kind"] == "entity"][["subject_sid", "party_sid"]].drop_duplicates().to_numpy()
        edges[("reg_entity", "owned_by_person", "person")] = ri[ri["party_kind"] == "person"][["subject_sid", "party_sid"]].drop_duplicates().to_numpy()
        edges[("vendor", "is_person", "person")] = np.array(list(ctx.reg_person_of.items()), dtype=object).reshape(-1, 2)
    return GraphTables(nodes, feats, edges)


class Scaler:
    """Per node type robust scaling, fitted on the fit graph and reused on later graphs."""

    def fit(self, g: GraphTables):
        self.med = {t: np.median(x, axis=0) for t, x in g.feats.items()}
        self.iqr = {t: np.maximum(np.quantile(x, 0.9, axis=0) - np.quantile(x, 0.1, axis=0), 1.0)
                    for t, x in g.feats.items()}
        return self

    def transform(self, t: str, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.med[t]) / self.iqr[t], -5, 5).astype(np.float32)


def to_heterodata(g: GraphTables, scaler: Scaler) -> "HeteroData":
    data = HeteroData()
    index = {t: {n: i for i, n in enumerate(ids)} for t, ids in g.nodes.items()}
    for t, x in g.feats.items():
        data[t].x = torch.tensor(scaler.transform(t, x))
    for (s, r, d), pairs in g.edges.items():
        if s not in index or d not in index:
            continue
        pairs = [(index[s][a], index[d][b]) for a, b in pairs if a in index[s] and b in index[d]]
        ei = torch.tensor(pairs, dtype=torch.long).t() if pairs else torch.zeros((2, 0), dtype=torch.long)
        data[s, r, d].edge_index = ei
    data = T.ToUndirected()(data)
    return data, index


if HAVE_TORCH:
    class _GNN(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden)
            self.conv2 = SAGEConv((-1, -1), hidden)

        def forward(self, x, edge_index):
            return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)

    class HeteroEncoder(nn.Module):
        def __init__(self, data: "HeteroData", hidden: int = 64):
            super().__init__()
            self.lin = nn.ModuleDict({t: nn.Linear(data[t].num_features, hidden) for t in data.node_types})
            self.gnn = to_hetero(_GNN(hidden), data.metadata(), aggr="sum")

        def forward(self, x_dict, edge_index_dict):
            h = {t: F.relu(self.lin[t](x)) for t, x in x_dict.items()}
            return self.gnn(h, edge_index_dict)

    class DistMult(nn.Module):
        def __init__(self, edge_types, hidden: int = 64):
            super().__init__()
            self.rel = nn.ParameterDict({"__".join(et): nn.Parameter(torch.ones(hidden)) for et in edge_types})

        def forward(self, z_src, z_dst, et):
            return (z_src * self.rel["__".join(et)] * z_dst).sum(-1)


def _typed_negatives(data, et, n: int, gen) -> "torch.Tensor":
    s, _, d = et
    return torch.stack([torch.randint(data[s].num_nodes, (n,), generator=gen),
                        torch.randint(data[d].num_nodes, (n,), generator=gen)])


def train_encoder(g: GraphTables, seed: int = 0, hidden: int = 64, epochs: int = 150, lr: float = 0.01):
    """Fit the encoder by relation-typed link prediction. Returns (model, scaler, report)."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    scaler = Scaler().fit(g)
    data, _ = to_heterodata(g, scaler)
    targets = [et for et in TARGETS if et in data.edge_types and data[et].edge_index.size(1) >= 10]
    split = T.RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=targets,
                              rev_edge_types=[(d, f"rev_{r}", s) for s, r, d in targets],
                              disjoint_train_ratio=0.3, add_negative_train_samples=False, is_undirected=False)
    train, val, test = split(data)
    model = HeteroEncoder(data, hidden)
    dec = DistMult(targets, hidden)
    with torch.no_grad():  # initialise lazy layers
        model(train.x_dict, train.edge_index_dict)
    opt = torch.optim.Adam(list(model.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(train.x_dict, train.edge_index_dict)
        loss = 0.0
        for et in targets:
            pos = train[et].edge_label_index
            neg = _typed_negatives(train, et, pos.size(1), gen)
            s, _, d = et
            lp = dec(z[s][pos[0]], z[d][pos[1]], et)
            ln = dec(z[s][neg[0]], z[d][neg[1]], et)
            loss = loss + F.binary_cross_entropy_with_logits(torch.cat([lp, ln]),
                                                             torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))
        loss.backward()
        opt.step()
    model.eval()
    report = {}
    with torch.no_grad():
        for name, part in (("val", val), ("test", test)):
            z = model(part.x_dict, part.edge_index_dict)
            aucs = {}
            for et in targets:
                pos = part[et].edge_label_index[:, part[et].edge_label == 1]
                neg = _typed_negatives(part, et, pos.size(1), gen)
                s, _, d = et
                y = np.r_[np.ones(pos.size(1)), np.zeros(neg.size(1))]
                sc = torch.cat([dec(z[s][pos[0]], z[d][pos[1]], et), dec(z[s][neg[0]], z[d][neg[1]], et)]).numpy()
                if 0 < y.sum() < len(y):
                    aucs[et[1]] = float(roc_auc_score(y, sc))
            report[name] = aucs
    report["final_loss"] = float(loss.detach())
    return model, scaler, report


def embed(model, scaler: Scaler, g: GraphTables) -> pd.DataFrame:
    """Apply a frozen encoder to a (possibly later) graph; returns vendor embeddings."""
    data, _ = to_heterodata(g, scaler)
    with torch.no_grad():
        z = model(data.x_dict, data.edge_index_dict)["vendor"].numpy()
    return pd.DataFrame(z, index=g.nodes["vendor"])


def vendor_embeddings(ctx: Context, seed: int = 0, fit_ctx: Context | None = None, **kw):
    """Fit on `fit_ctx` (default: ctx itself) and embed the vendors of ctx."""
    if not HAVE_TORCH:
        raise ImportError("graph embeddings need torch and torch_geometric (see pyproject extra 'gnn')")
    g_fit = graph_tables(fit_ctx or ctx, ctx.use_ownership)
    model, scaler, report = train_encoder(g_fit, seed=seed, **kw)
    g = g_fit if fit_ctx is None else graph_tables(ctx, ctx.use_ownership)
    return embed(model, scaler, g), report


class EmbeddingCaseFeatures:
    """Case representation from vendor embeddings: a low-dimensional summary of the
    mean embedding (PCA fitted on all vendors, no labels) plus group-structure
    features, which mean pooling alone would lose."""

    def __init__(self, emb: pd.DataFrame, n_components: int = 8, seed: int = 0):
        from sklearn.decomposition import PCA
        self.emb = emb
        x = emb.to_numpy()
        self.norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        self.row = {r: i for i, r in enumerate(emb.index)}
        self.pca = PCA(n_components=n_components, random_state=seed).fit(x)

    def features(self, members) -> dict:
        idx = [self.row[r] for r in members if r in self.row]
        x = self.emb.to_numpy()[idx]
        n = self.norm[idx]
        pcs = self.pca.transform(x.mean(axis=0, keepdims=True))[0]
        out = {f"E_pc{i}": float(v) for i, v in enumerate(pcs)}
        if len(idx) > 1:
            sim = n @ n.T
            k = len(idx)
            out["E_mean_cos"] = float((sim.sum() - k) / (k * (k - 1)))
            out["E_spread"] = float(np.linalg.norm(x - x.mean(axis=0), axis=1).mean())
        else:
            out["E_mean_cos"] = 1.0
            out["E_spread"] = 0.0
        return out

    def matrix(self, member_sets) -> pd.DataFrame:
        return pd.DataFrame([self.features(m) for m in member_sets]).astype(float).fillna(0.0)
