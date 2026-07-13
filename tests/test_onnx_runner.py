"""Multi-head ONNX inference (Plan 1 §12.1) -- runs a tiny real multi-output
graph through ``OnnxRunner`` so the head->activation->probability mapping and the
backward-compatible single-head path are both exercised end-to-end.

Skipped if ``onnx``/``onnxruntime`` are unavailable (both are app deps, and the
maintainer's repo-root venv used to run this suite carries them -- see
app/README.md)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from biosqa.inference import onnx_runner as onnx_runner_mod  # noqa: E402
from biosqa.inference.onnx_runner import _MAX_BATCH, OnnxRunner, _softmax  # noqa: E402
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


# --------------------------------------------------------------- batch cap (Plan 2 §7.2)
# An uncapped batch peaks in the GBs on a long record at 0.9 overlap (the window stack plus
# the float64 STFT workspace behind the spectral branch). _run_raw slices it; the slicing
# must be EXACT, so these compare against the pre-cap behaviour (one session.run) bit for bit.

SPEC_L = 64  # long enough that the STFT lands >1 frame and exercises the interp-to-L path
SPEC_CARD = {
    **VALID_CARD,
    "L_m": SPEC_L,
    "heads": V2_HEADS,
    "spectral_preprocessing": {"bands_hz": [[15, 50], [50, 110]], "frame_s": 0.05, "hop_s": 0.02},
}


def _build_dual_onnx(path, outputs, length, n_bands):
    """Dual-branch graph (the 2-input ECG contract): x_raw [B,1,L] + x_spec [B,C,L].

    Flattens both inputs and sums a MatMul of each, so every head's output depends on
    BOTH branches -- a mis-sliced spectral branch would change the result rather than
    being masked by a raw-only path.
    """
    rng = np.random.default_rng(4)
    raw = helper.make_tensor_value_info("window", TensorProto.FLOAT, ["batch", 1, length])
    spec = helper.make_tensor_value_info("spec", TensorProto.FLOAT, ["batch", n_bands, length])
    nodes = [helper.make_node("Flatten", ["window"], ["flat"], axis=1),
             helper.make_node("Flatten", ["spec"], ["sflat"], axis=1)]
    inits, value_infos = [], []
    for name, dim in outputs:
        w_raw, w_spec = f"Wr_{name}", f"Ws_{name}"
        inits.append(numpy_helper.from_array(rng.standard_normal((length, dim)).astype(np.float32), w_raw))
        inits.append(numpy_helper.from_array(
            rng.standard_normal((n_bands * length, dim)).astype(np.float32), w_spec))
        nodes.append(helper.make_node("MatMul", ["flat", w_raw], [f"{name}_r"]))
        nodes.append(helper.make_node("MatMul", ["sflat", w_spec], [f"{name}_s"]))
        nodes.append(helper.make_node("Add", [f"{name}_r", f"{name}_s"], [name]))
        value_infos.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", dim]))
    graph = helper.make_graph(nodes, "dualbranch", [raw, spec], value_infos, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _spec_runner(tmp_path):
    (tmp_path / "ecg.model_card.json").write_text(json.dumps(SPEC_CARD))
    _build_dual_onnx(tmp_path / "ecg.onnx", _v2_outputs(), SPEC_L,
                     len(SPEC_CARD["spectral_preprocessing"]["bands_hz"]))
    runner = OnnxRunner("ecg", tmp_path)
    runner.load()
    return runner


def _ragged_n():
    """More than two full chunks, with a partial tail (the case that catches an
    off-by-one slice or a dropped remainder)."""
    return 2 * _MAX_BATCH + 37


def _assert_chunking_is_exact(runner, windows, monkeypatch):
    """predict_windows / predict_windows_multihead under the cap == one unchunked run."""
    monkeypatch.setattr(onnx_runner_mod, "_MAX_BATCH", len(windows) + 1)  # pre-cap: single session.run
    ref_raw = runner.predict_windows(windows)
    ref_heads = {k: v.copy() for k, v in runner.predict_windows_multihead(windows).per_head.items()}
    monkeypatch.undo()

    got_raw = runner.predict_windows(windows)
    got_heads = runner.predict_windows_multihead(windows).per_head
    assert got_raw.shape == (len(windows), 4)
    np.testing.assert_array_equal(got_raw, ref_raw)
    assert set(got_heads) == set(ref_heads)
    for name, ref in ref_heads.items():
        np.testing.assert_array_equal(got_heads[name], ref)


def test_chunked_run_is_bit_exact_plain_card(tmp_path, monkeypatch):
    runner = _runner(tmp_path, {**VALID_CARD, "heads": V2_HEADS}, _v2_outputs())
    windows = np.random.default_rng(11).standard_normal((_ragged_n(), L)).astype(np.float32)
    _assert_chunking_is_exact(runner, windows, monkeypatch)


def test_chunked_run_is_bit_exact_spectral_card(tmp_path, monkeypatch):
    # The dual-branch path recomputes the host-side spectral channels PER CHUNK;
    # spectral_band_channels is elementwise over the batch axis, so that must be exact too.
    runner = _spec_runner(tmp_path)
    windows = np.random.default_rng(12).standard_normal((_ragged_n(), SPEC_L)).astype(np.float32)
    _assert_chunking_is_exact(runner, windows, monkeypatch)


class _SpySession:
    """Records the batch size of every session.run (the thing the cap bounds)."""

    def __init__(self, session, sizes):
        self._session = session
        self._sizes = sizes

    def run(self, output_names, feed):
        self._sizes.append(len(next(iter(feed.values()))))
        return self._session.run(output_names, feed)


def test_batch_is_capped_at_max_batch(tmp_path):
    runner = _runner(tmp_path, {**VALID_CARD, "heads": V2_HEADS}, _v2_outputs())
    n = _ragged_n()
    windows = np.random.default_rng(13).standard_normal((n, L)).astype(np.float32)
    sizes = []
    runner._session = _SpySession(runner._session, sizes)

    runner.predict_windows(windows)
    # never more than _MAX_BATCH windows resident in one run, and none dropped
    assert sizes == [_MAX_BATCH, _MAX_BATCH, 37]
    assert sum(sizes) == n


def test_short_stack_still_runs_in_one_batch(tmp_path):
    # At or below the cap the driver must not add a concatenate round-trip.
    runner = _runner(tmp_path, {**VALID_CARD, "heads": V2_HEADS}, _v2_outputs())
    windows = np.random.default_rng(14).standard_normal((_MAX_BATCH, L)).astype(np.float32)
    sizes = []
    runner._session = _SpySession(runner._session, sizes)

    assert runner.predict_windows(windows).shape == (_MAX_BATCH, 4)
    assert sizes == [_MAX_BATCH]


# --- the runner must ACTUALLY USE the card's integrity + precision facts ------------------------
# ModelCard.verify_onnx() was written, tested at the card level, and then never called: the runner
# opened the session straight after parsing the card, so a swapped or corrupted .onnx loaded happily
# as long as the card said the right words. These pin the WIRING, which is what was missing.

def test_load_refuses_a_model_whose_digest_does_not_match_the_card(tmp_path):
    """Swap the .onnx behind a card that declares a digest -> the runner must refuse to open it."""
    card = dict(VALID_CARD, heads=V2_HEADS)
    _build_onnx(tmp_path / "ecg.onnx", _v2_outputs())
    card["onnx_sha256"] = hashlib.sha256((tmp_path / "ecg.onnx").read_bytes()).hexdigest()
    (tmp_path / "ecg.model_card.json").write_text(json.dumps(card))

    OnnxRunner("ecg", tmp_path).load()                      # honest artifact: loads

    # Swap the artifact behind the card's back. ModelCardError (not an onnxruntime protobuf error)
    # is what proves the digest is checked BEFORE the session is opened -- refuse to run rather than
    # predict from an unknown model.
    (tmp_path / "ecg.onnx").write_bytes(b"not the model the card was written for")
    with pytest.raises(ModelCardError, match="SHA-256"):
        OnnxRunner("ecg", tmp_path).load()


def test_load_reads_precision_off_the_graph_rather_than_assuming_it(tmp_path):
    """The status bar used to hardcode "FP32", which was true only by luck. It is now observed."""
    runner = _runner(tmp_path, dict(VALID_CARD, heads=V2_HEADS), _v2_outputs())
    assert runner.precision == "FP32"          # this graph carries no INT8 quantization ops


def test_a_card_with_no_digest_still_loads_verification_is_skipped_never_faked(tmp_path):
    """Every card shipped before onnx_sha256 existed carries no digest. They must keep working."""
    runner = _runner(tmp_path, dict(VALID_CARD, heads=V2_HEADS), _v2_outputs())
    assert runner.card.onnx_sha256 is None
