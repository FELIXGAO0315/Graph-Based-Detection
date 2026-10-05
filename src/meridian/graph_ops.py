"""Graph operations used by the detectors: typed edges and meta-paths.

A meta-path is a sequence of typed relations, for example
    VendorRecord -CONTRACTED_UNDER-> Engagement -SPONSORED_BY-> Employee.
`metapath` returns every instance of one, with the node at each step, the
edge id of each hop (the evidence a case can cite) and the edge attributes.
Two vendors are related through a meta-path when their instances end on
the same node: that is how shared addresses, contacts, sponsors, approvers
and institutions are found.
"""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from .graph import Graph

DATE_FIELDS = ("valid_from", "valid_to", "observed_at")


class HeteroGraph:
    """Indexed view of a canonical Graph for traversal."""

    def __init__(self, g: Graph):
        self.g = g

    @cached_property
    def node_type(self) -> dict:
        return dict(zip(self.g.nodes["node_id"], self.g.nodes["node_type"]))

    @cached_property
    def _by_relation(self) -> dict[str, pd.DataFrame]:
        e = self.g.edges.copy()
        for c in DATE_FIELDS:
            e[c] = pd.to_datetime(e[c]).astype("datetime64[ns]")
        e["props"] = [json.loads(p) for p in e["relation_properties"]]
        return {rel: df.reset_index(drop=True) for rel, df in e.groupby("relation_type", sort=True)}

    def edges(self, *relations: str) -> pd.DataFrame:
        """Edges of the given relation types (all columns, dates parsed, props as dicts)."""
        parts = [self._by_relation[r] for r in relations if r in self._by_relation]
        if not parts:
            return self.g.edges.iloc[0:0].assign(props=[])
        return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    @cached_property
    def _path_cache(self) -> dict:
        return {}

    def metapath(self, *steps: str | tuple, where=None) -> pd.DataFrame:
        if where is None:
            if steps not in self._path_cache:
                self._path_cache[steps] = self._metapath(*steps)
            return self._path_cache[steps]
        return self._metapath(*steps, where=where)

    def _metapath(self, *steps: str | tuple, where=None) -> pd.DataFrame:
        """Instances of a meta-path.

        Each step is a relation name (followed source -> target) or a tuple of
        relation names (any of them). Returns columns n0..nk (node ids),
        e1..ek (edge ids), r1..rk (relation), and per hop the edge's
        valid_from/valid_to/observed_at/source_record_id/props as `<field><i>`.
        `where` is an optional dict {hop index: predicate(edge_row_frame) -> mask}.
        """
        out = None
        for i, step in enumerate(steps, start=1):
            rels = (step,) if isinstance(step, str) else tuple(step)
            e = self.edges(*rels)
            if where and i in where:
                # an explicit boolean row mask: a plain list would select columns on an empty frame
                e = e.loc[np.asarray(where[i](e), dtype=bool)]
            hop = pd.DataFrame({
                f"n{i - 1}": e["source_node_id"].to_numpy(), f"n{i}": e["target_node_id"].to_numpy(),
                f"e{i}": e["edge_id"].to_numpy(), f"r{i}": e["relation_type"].to_numpy(),
                f"valid_from{i}": e["valid_from"].to_numpy(), f"valid_to{i}": e["valid_to"].to_numpy(),
                f"observed_at{i}": e["observed_at"].to_numpy(), f"source_record_id{i}": e["source_record_id"].to_numpy(),
                f"props{i}": e["props"].to_numpy(),
            })
            out = hop if out is None else out.merge(hop, on=f"n{i - 1}", how="inner")
        return out

    def shared(self, *steps, where=None) -> dict[str, list]:
        """End node -> start nodes of the meta-path instances that reach it (in instance order)."""
        mp = self.metapath(*steps, where=where)
        k = len(steps)
        out: dict[str, list] = {}
        for start, end in zip(mp["n0"], mp[f"n{k}"]):
            out.setdefault(end, []).append(start)
        return out


def strip(node_id: str) -> str:
    """'vendor:TP-123' -> 'TP-123'."""
    return node_id.split(":", 1)[1]


def hetero(ctx) -> HeteroGraph:
    """The HeteroGraph of a graph-backed Context (cached on the context)."""
    if ctx.graph is None:
        raise ValueError("this context was not built from a graph (use Context.from_graph)")
    if "_hetero" not in ctx.__dict__:
        ctx.__dict__["_hetero"] = HeteroGraph(ctx.graph)
    return ctx.__dict__["_hetero"]
