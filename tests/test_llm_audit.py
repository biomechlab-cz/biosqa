"""Tests for the on-demand LLM audit (pure parts + graceful failure; no ollama dependency)."""
import numpy as np

import biosqa.inference.llm_audit as la
from biosqa.inference.llm_audit import audit_segment, universal_sqis

_OK = '{"grade": 1, "confidence": 0.8, "artifacts": [], "prefilter_suspected": false, "rationale": "x"}'


def _ecg(fs=250.0, secs=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = sum(np.exp(-0.5 * ((t - c) / 0.02) ** 2) for c in range(int(secs)))
    return (x + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def test_universal_sqis_finite_dict():
    s = universal_sqis(_ecg(), 250.0)
    assert set(s) >= {"hf_ratio", "spectral_flatness", "kurtosis", "flatline_frac"}
    assert all(np.isfinite(v) for v in s.values())


def test_universal_sqis_multichannel():
    x = np.stack([_ecg(seed=i) for i in range(3)])
    s = universal_sqis(x, 250.0)
    assert np.isfinite(s["hf_ratio"])


def test_audit_segment_graceful_when_ollama_unreachable():
    # unreachable host -> must degrade to {"error": ...}, never raise (caller keeps model verdict)
    out = audit_segment(_ecg(), 250.0, "ecg", model_grade={"grade": 3, "p_usable": 0.9},
                        guard={"prefiltered": True, "reasons": ["hf suppressed"]},
                        host="http://127.0.0.1:1", samples=1, timeout=2.0)
    assert "error" in out and "sqi" in out


def test_audit_keeps_partial_samples_on_late_error(monkeypatch):
    """A mid-loop timeout must NOT discard the samples already collected (else a 2-of-3 majority collapses
    into a hard error)."""
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                                   # the last sample times out
            raise TimeoutError("hang")
        return _OK

    monkeypatch.setattr(la, "_ollama", fake)
    out = audit_segment(_ecg(), 250.0, "ecg", samples=3)
    assert "error" not in out
    assert out["grade"] == 1 and out["votes"] == [1, 1] and out["agreement"] == 1.0


def test_audit_errors_only_when_all_samples_fail(monkeypatch):
    monkeypatch.setattr(la, "_ollama",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("ollama down")))
    out = audit_segment(_ecg(), 250.0, "ecg", samples=3)
    assert "error" in out and "ConnectionError" in out["error"]
