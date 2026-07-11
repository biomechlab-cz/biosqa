"""Multi-head ONNX inference (Plan 1 §12.1) -- runs a tiny real multi-output
graph through ``OnnxRunner`` so the head->activation->probability mapping and the
backward-compatible single-head path are both exercised end-to-end.

Skipped if ``onnx``/``onnxruntime`` are unavailable (both are app deps, and the
maintainer's repo-root venv used to run this suite carries them -- see
app/README.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from biosqa.inference.onnx_runner import OnnxRunner, _softmax  # noqa: E402
from biosqa.model.model_card import ModelCardError  # noqa: E402

L = 8  # tiny window length for the test graph == card L_m

VALID_CARD = {
    "modality": "ecg",
    "L_m": L,
    "fs_hz": 250,
    "class_order": ["Q0", "Q1", "Q2", "Q3"],
    "normalization": {"method": "none"},
    "training_data_hash": "sha256:test",
    "model_version": "test",
}

V2_HEADS = [
    {"name": "grade", "output_name": "q_logits", "kind": "ordinal",
     "activation": "softmax", "class_order": ["Q0", "Q1", "Q2", "Q3"]},
    {"name": "usable", "output_name": "bin_logits", "kind": "binary",
     "activation": "softmax", "class_order": ["BAD", "OK"]},
    {"name": "artifact", "output_name": "type_logits", "kind": "multilabel",
     "activation": "sigmoid",
     "class_order": ["clean", "baseline_wander", "motion", "muscle", "electrode", "powerline"],
     "threshold": 0.5},
]


def _build_onnx(path, outputs):
    """A linear model: Flatten([B,1,L])->[B,L] then one MatMul per output head.

    ``outputs`` is a list of (tensor_name, dim); the batch dim is exported as the
    symbolic axis "batch" (matching the real dynamic-batch export contract).
    """
    rng = np.random.default_rng(0)
    inp = helper.make_tensor_value_info("window", TensorProto.FLOAT, ["batch", 1, L])
    nodes = [helper.make_node("Flatten", ["window"], ["flat"], axis=1)]
    inits, value_infos = [], []
    for name, dim in outputs:
        w_name = f"W_{name}"
        inits.append(numpy_helper.from_array(rng.standard_normal((L, dim)).astype(np.float32), w_name))
        nodes.append(helper.make_node("MatMul", ["flat", w_name], [name]))
        value_infos.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", dim]))
    graph = helper.make_graph(nodes, "multihead", [inp], value_infos, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _runner(tmp_path, card, outputs):
    (tmp_path / "ecg.model_card.json").write_text(json.dumps(card))
    _build_onnx(tmp_path / "ecg.onnx", outputs)
    runner = OnnxRunner("ecg", tmp_path)
    runner.load()
    return runner


def _v2_outputs():
    return [("q_logits", 4), ("bin_logits", 2), ("type_logits", 6)]


def test_multihead_shapes_and_activations(tmp_path):
    card = {**VALID_CARD, "heads": V2_HEADS}
    runner = _runner(tmp_path, card, _v2_outputs())
    windows = np.random.default_rng(1).standard_normal((3, L)).astype(np.float32)

    pred = runner.predict_windows_multihead(windows)
    assert set(pred.per_head) == {"grade", "usable", "artifact"}
    assert pred.per_head["grade"].shape == (3, 4)
    assert pred.per_head["usable"].shape == (3, 2)
    assert pred.per_head["artifact"].shape == (3, 6)
    # softmax heads: rows sum to 1; sigmoid head: all in [0, 1], rows need NOT sum to 1.
    assert np.allclose(pred.per_head["grade"].sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(pred.per_head["usable"].sum(axis=1), 1.0, atol=1e-5)
    art = pred.per_head["artifact"]
    assert np.all((art >= 0.0) & (art <= 1.0))
    assert pred.primary.shape == (3, 4)


def test_predict_windows_returns_primary_raw_logits(tmp_path):
    card = {**VALID_CARD, "heads": V2_HEADS}
    runner = _runner(tmp_path, card, _v2_outputs())
    windows = np.random.default_rng(2).standard_normal((5, L)).astype(np.float32)

    raw = runner.predict_windows(windows)
    assert raw.shape == (5, 4) and raw.dtype == np.float32
    # predict_windows is the primary head's RAW logits; softmaxing them must
    # reproduce the multihead grade probabilities.
    pred = runner.predict_windows_multihead(windows)
    assert np.allclose(_softmax(raw), pred.primary, atol=1e-5)


def test_empty_windows(tmp_path):
    card = {**VALID_CARD, "heads": V2_HEADS}
    runner = _runner(tmp_path, card, _v2_outputs())
    empty = np.empty((0, L), dtype=np.float32)
    assert runner.predict_windows(empty).shape == (0, 4)
    pred = runner.predict_windows_multihead(empty)
    assert pred.per_head["artifact"].shape == (0, 6)
    assert pred.primary.shape == (0, 4)


def test_missing_declared_head_output_fails_loudly(tmp_path):
    # Card declares type_logits, but the graph only emits q_logits + bin_logits.
    card = {**VALID_CARD, "heads": V2_HEADS}
    with pytest.raises(ModelCardError):
        _runner(tmp_path, card, [("q_logits", 4), ("bin_logits", 2)])


def test_legacy_single_output_model_still_runs(tmp_path):
    # No `heads` in the card + a single-output graph == the old contract.
    runner = _runner(tmp_path, VALID_CARD, [("q_logits", 4)])
    windows = np.random.default_rng(3).standard_normal((4, L)).astype(np.float32)
    assert runner.predict_windows(windows).shape == (4, 4)
    pred = runner.predict_windows_multihead(windows)
    assert set(pred.per_head) == {"grade"}
    assert np.allclose(pred.per_head["grade"].sum(axis=1), 1.0, atol=1e-5)
