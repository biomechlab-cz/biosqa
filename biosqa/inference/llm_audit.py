"""On-demand LLM AUDIT of a flagged segment — a human-facing second opinion (off the decision path).

The quality model + guards make the decision; this reasons, in plain language, over a flagged window's
numeric SQI summary + the guard verdict and returns a structured second opinion the UI can show. It is
NEVER a gate and its self-reported grade/confidence is never consumed as a probability: benchmarks show
a calibrated model over the SQIs beats the LLM at grading (LLM ~0.55 vs tree ~0.94 AUROC), while the LLM
is a competent EXPLAINER that catches structured failures (e.g. false-clean on pre-filtered input).

Runs against a LOCAL ollama server (stdlib urllib, no extra dependency; default qwen3:32b — 7-9B models
fail structured JSON far more often). Self-consistency: sample k times, majority-vote the grade. Only ever
call this on the handful of guard-flagged windows, never per-window at scale.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter

import numpy as np

__all__ = ["audit_segment", "universal_sqis", "AUDIT_SYSTEM"]

AUDIT_SYSTEM = (
    "You are a biosignal signal-quality expert giving a SECOND OPINION on one {modality} window that an "
    "automated quality model flagged. You get the window's numeric signal-quality indices and the runtime "
    "guard's findings. Judge whether the automated verdict is trustworthy. Watch for FALSE-CLEAN: if the "
    "input looks pre-filtered (near-zero high-frequency energy) the spectral indices can read clean even "
    "though the signal is corrupted — but pre-filtering ALONE is not corruption; only doubt a clean reading "
    "when a filter-robust cue (beat-detector agreement bSQI) is ALSO low. Respond with ONLY JSON: "
    '{"grade": <0|1|2|3>, "confidence": <0..1>, "artifacts": [<strings>], "prefilter_suspected": '
    '<true|false>, "rationale": "<one or two sentences>"}. 0=unacceptable,1=poor,2=acceptable,3=excellent.'
)


def universal_sqis(window: np.ndarray, fs: float) -> dict:
    """Compact modality-agnostic SQI summary (pure numpy) for the audit prompt."""
    x = np.asarray(window, dtype=np.float64)
    x = x.mean(0) if x.ndim == 2 else x
    z = (x - x.mean()) / (x.std() + 1e-9)
    P = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    f = np.fft.rfftfreq(len(z), d=1.0 / fs)
    tot = P.sum() + 1e-12
    hi = P[f > min(40.0, fs / 2 - 1)].sum() / tot
    pm = P + 1e-12
    flat = float(np.exp(np.log(pm).mean()) / pm.mean())          # spectral flatness (noise-like ~1)
    dz = np.diff(z)
    return {
        "hf_ratio": round(float(hi), 4),
        "spectral_flatness": round(flat, 4),
        "kurtosis": round(float((z ** 4).mean() - 3.0), 3),
        "skewness": round(float((z ** 3).mean()), 3),
        "flatline_frac": round(float((np.abs(dz) < 1e-3).mean()), 4),
    }


def _parse(raw: str):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "grade" in d:
        try:
            d["grade"] = int(d["grade"])
        except (ValueError, TypeError):
            return None
    return d


def _ollama(system: str, user: str, model: str, host: str, temperature: float, timeout: float) -> str:
    payload = {"model": model, "stream": False, "options": {"temperature": float(temperature)},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def audit_segment(window: np.ndarray, fs: float, modality: str, *, model_grade=None, guard=None,
                  model: str = "qwen3:32b", host: str = "http://localhost:11434", samples: int = 3,
                  timeout: float = 180.0) -> dict:
    """Audit ONE flagged window. ``model_grade`` e.g. {"grade":3,"p_usable":0.94}; ``guard`` = the dict
    from ``OnnxRunner.guard_record`` (prefiltered / reasons). Self-consistency over ``samples`` (majority
    grade). Returns {grade, confidence, artifacts, prefilter_suspected, rationale, votes, agreement}.
    On any transport error returns {"error": ...} — the caller keeps the model's own verdict."""
    sqi = universal_sqis(window, fs)
    parts = [f"Modality: {modality}", f"Sampling rate: {fs:g} Hz", "Signal-quality indices:"]
    parts += [f"  - {k}: {v}" for k, v in sqi.items()]
    if model_grade:
        parts += ["Automated model verdict: " + json.dumps(model_grade)]
    if guard:
        parts += [f"Runtime guard: prefiltered={guard.get('prefiltered')}; findings={guard.get('reasons', [])}"]
    parts += ["Respond with the JSON object only."]
    system, user = AUDIT_SYSTEM.replace("{modality}", modality), "\n".join(parts)
    temp = 0.0 if samples <= 1 else 0.7
    judgments = []
    last_err = None
    for _ in range(max(1, samples)):
        # catch PER SAMPLE: one mid-loop transport error/timeout must not discard the samples already
        # collected (that would collapse a 2-of-3 majority into a hard error). Only error when we got none.
        try:
            j = _parse(_ollama(system, user, model, host, temp, timeout))
        except Exception as e:  # noqa: BLE001 - ollama down / timeout for this sample; try the rest
            last_err = e
            continue
        if j:
            judgments.append(j)
    if not judgments:
        if last_err is not None:
            return {"error": f"{type(last_err).__name__}: {str(last_err)[:100]}", "sqi": sqi}
        return {"error": "no valid JSON from LLM", "sqi": sqi}
    grades = [j["grade"] for j in judgments if j.get("grade") is not None]
    maj, cnt = Counter(grades).most_common(1)[0]
    best = next(j for j in judgments if j.get("grade") == maj)
    best.update({"grade": maj, "votes": grades, "agreement": cnt / len(grades), "sqi": sqi})
    return best
