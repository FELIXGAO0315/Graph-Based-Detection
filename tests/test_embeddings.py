import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from meridian.context import Context
from meridian.data import as_of
from meridian.embeddings import EmbeddingCaseFeatures, embed, graph_tables, train_encoder


def test_encoder_fitted_early_applies_frozen_to_later_snapshot(bundle):
    fit_ctx = Context(as_of(bundle, "2024-06-30"))
    ctx = Context(as_of(bundle, "2026-06-30"))
    model, scaler, report = train_encoder(graph_tables(fit_ctx), seed=0, epochs=20)
    assert report["val"] and all(0.0 <= v <= 1.0 for v in report["val"].values())
    emb = embed(model, scaler, graph_tables(ctx))
    assert emb.shape == (len(ctx.vendors), 64)
    assert np.isfinite(emb.to_numpy()).all()
    # Vendors onboarded after the fit date still get embeddings (no per-node ID vectors).
    new = ctx.vendors.index[ctx.vendors["onboarding_date"] > pd.Timestamp("2024-06-30")]
    assert len(new) > 0 and emb.loc[new].abs().sum(axis=1).gt(0).all()


def test_case_features_keep_group_structure(bundle):
    ctx = Context(as_of(bundle, "2026-06-30"))
    model, scaler, _ = train_encoder(graph_tables(ctx), seed=0, epochs=5)
    emb = embed(model, scaler, graph_tables(ctx))
    f = EmbeddingCaseFeatures(emb).matrix([frozenset(emb.index[:3]), frozenset(emb.index[:1])])
    assert {"E_mean_cos", "E_spread", "E_pc0"} <= set(f.columns)
    assert f.loc[1, "E_spread"] == 0.0
