"""research3.md-derived runtime features: interpretable SQI breakdown + task-relative (rate) usability,
plus the first-batch SQA additions — tSQI, physiological-plausibility voters, PPG rate-usability, and the
ECG spectral-ratio SQIs (pSQI/basSQI)."""
import numpy as np

from biosqa.inference.conformal import aps_prediction_set, temperature_scale
from biosqa.inference.input_sanity import input_sanity
from biosqa.inference.integrity import _bandpass_fft, integrity_guard, rr_plausibility, tsqi
from biosqa.inference.novelty import novelty_distance, sqi_feature_vector
from biosqa.inference.recover import filter_for_modality
from biosqa.inference.segmenter import run_length_encode
from biosqa.inference.sqi_breakdown import sqi_breakdown, sqi_consensus
from biosqa.inference.task_usability import (
    band_usability, eda_component_usability, estimate_hr, estimate_pulse_rate, rate_usability,
    usability_verdicts,
)

FS = 250.0


def _ppg(rate_hz=1.2, noise=0.03, seed=0, fs=125.0, dur=10.0):
    """A smooth periodic pulse train (PPG-like)."""
    t = np.arange(int(fs * dur)) / fs
    x = 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t)) ** 2          # peaky positive pulse
    return (x + noise * np.random.default_rng(seed).standard_normal(x.size)).astype(np.float64)


class _PF:                                                        # stub prefilter verdict
    prefiltered = True
    reasons: list = []
    score = 1.0


def _ecg(hr_hz=1.1, noise=0.01, seed=0):
    """A realistic triphasic-QRS ECG, middle 10 s window (bSQI≈1.0 on the clean signal)."""
    n = int(12 * FS)
    idx = np.arange(n)
    x = np.zeros(n)
    for k in range(int(12 * hr_hz) + 1):
        c = int((1.0 + k / hr_hz) * FS)
        if 0 <= c < n - 3:
            x += 1.0 * np.exp(-((idx - c) ** 2) / (2 * (0.010 * FS) ** 2))                 # R
            x -= 0.18 * np.exp(-((idx - (c - int(0.03 * FS))) ** 2) / (2 * (0.008 * FS) ** 2))  # Q
            x -= 0.30 * np.exp(-((idx - (c + int(0.03 * FS))) ** 2) / (2 * (0.010 * FS) ** 2))  # S
    x += noise * np.random.default_rng(seed).standard_normal(n)
    return x[int(FS):int(FS) + 2500].astype(np.float64)


def test_sqi_breakdown_separates_clean_and_corrupt():
    clean = _ecg()
    corrupt = clean + 0.8 * np.random.default_rng(1).standard_normal(clean.size)
    sc = {d["name"]: d["value"] for d in sqi_breakdown(clean, FS, "ecg")}
    sx = {d["name"]: d["value"] for d in sqi_breakdown(corrupt, FS, "ecg")}
    assert sc["bSQI"] > sx["bSQI"]              # clean → higher beat-detector agreement
    assert sx["HF noise"] > sc["HF noise"]      # corrupt → more high-frequency energy
    for d in sqi_breakdown(clean, FS, "ecg"):
        assert {"name", "value", "hint", "bar", "desc"} <= set(d)


def test_ppg_hf_and_eda_motion_are_meaningful():
    """Review follow-up: PPG HF-noise split below its Nyquist (was structurally 0 at 40 Hz / fs 64), and
    EDA Motion is a bounded 0..1 HF fraction (was a heavy-tailed percentile/median ratio)."""
    rng = np.random.default_rng(0)
    pfs = 64.0
    pt = np.arange(int(pfs * 10)) / pfs
    clean_ppg = 0.5 * (1 + np.sin(2 * np.pi * 1.2 * pt)) ** 2 + 0.01 * rng.standard_normal(pt.size)
    noisy_ppg = clean_ppg + 0.3 * rng.standard_normal(pt.size)
    hf_c = {r["name"]: r["value"] for r in sqi_breakdown(clean_ppg, pfs, "ppg")}["HF noise"]
    hf_n = {r["name"]: r["value"] for r in sqi_breakdown(noisy_ppg, pfs, "ppg")}["HF noise"]
    assert hf_n > hf_c and hf_n > 0.0                         # PPG HF now registers noise (not stuck at 0)
    efs = 8.0
    et = np.arange(int(efs * 30)) / efs
    clean_eda = 3.0 + 0.4 * np.exp(-((et - 8) ** 2) / 2) + 0.005 * rng.standard_normal(et.size)
    m_c = {r["name"]: r["value"] for r in sqi_breakdown(clean_eda, efs, "eda")}["Motion"]
    m_m = {r["name"]: r["value"] for r in sqi_breakdown(clean_eda + 0.5 * rng.standard_normal(et.size), efs, "eda")}["Motion"]
    assert 0.0 <= m_c <= 1.0 and m_m > m_c                    # EDA Motion bounded 0..1 and separates


def test_sqi_breakdown_modality_sets():
    assert [d["name"] for d in sqi_breakdown(_ecg(), FS, "eeg")]     # non-empty EEG set
    assert [d["name"] for d in sqi_breakdown(_ecg(), FS, "eda")]     # non-empty EDA set
    assert sqi_breakdown(np.zeros(2), FS, "ecg") == []               # too short → empty


def test_estimate_hr_ecg():
    hr = estimate_hr(_ecg(hr_hz=1.1), FS)
    assert hr is not None and 60 <= hr <= 72                         # ~66 bpm


def test_rate_usable_recovers_wander_not_noise():
    clean = _ecg()
    t = np.arange(clean.size) / FS
    wander = clean + 3.0 * np.sin(2 * np.pi * 0.15 * t)              # strong drift, beats intact
    r = rate_usability(wander, FS, "ecg")
    assert r["rate_usable"] is True and 60 <= r["hr_bpm"] <= 72      # poor morphology, rate recoverable
    noise = np.random.default_rng(2).standard_normal(clean.size)
    assert rate_usability(noise, FS, "ecg")["rate_usable"] is False  # no reliable beats
    assert rate_usability(clean, FS, "eeg")["rate_usable"] is False  # not an ECG/PPG task


# ---- first-batch SQA additions ------------------------------------------------

def test_tsqi_separates_clean_and_corrupt():
    clean = _ecg()
    corrupt = clean + 1.0 * np.random.default_rng(3).standard_normal(clean.size)
    tc, tx = tsqi(clean, FS, "ecg"), tsqi(corrupt, FS, "ecg")
    assert tc > 0.8 and tc > tx                                   # clean beats correlate to the template
    assert tsqi(np.zeros(500), FS, "ecg") == 0.0                  # no beats → abstain (0.0)


def test_ecg_breakdown_has_new_indices_ppg_does_not_get_spectral():
    ecg_names = [d["name"] for d in sqi_breakdown(_ecg(), FS, "ecg")]
    assert {"tSQI", "pSQI", "basSQI"} <= set(ecg_names)           # ECG gets tSQI + QRS-band ratios
    ppg_names = [d["name"] for d in sqi_breakdown(_ppg(), 125.0, "ppg")]
    assert "tSQI" in ppg_names                                    # PPG gets tSQI …
    assert "pSQI" not in ppg_names and "basSQI" not in ppg_names  # … but not the ECG-band ratios


def test_rr_plausibility_flags_erratic_timing():
    assert rr_plausibility(_ecg(hr_hz=1.1), FS, "ecg")["plausible"] is True     # clean rhythm
    # a lone spike train with one huge gap → implausible max-RR / ratio
    x = np.zeros(2500)
    for i in (100, 180, 260, 2400):                              # 3 close beats then a >8 s gap
        x[i] = 1.0
    v = rr_plausibility(x, FS, "ecg")
    assert v["plausible"] is False and v["reasons"]


def test_ppg_rate_usability_and_pulse_rate():
    good = _ppg(rate_hz=1.2)                                      # 72 bpm
    r = rate_usability(good, 125.0, "ppg")
    assert r["rate_usable"] is True and 66 <= r["hr_bpm"] <= 78
    assert 66 <= estimate_pulse_rate(good, 125.0) <= 78
    noise = np.random.default_rng(4).standard_normal(good.size)
    assert rate_usability(noise, 125.0, "ppg")["rate_usable"] is False


def test_guard_reaches_ppg_via_tsqi():
    """PPG previously had NO false-clean voter; tSQI + plausibility make the guard reachable for PPG."""
    corrupt_ppg = _ppg(noise=1.5)
    v = integrity_guard(corrupt_ppg, 125.0, "ppg", model_p_unusable=0.1, prefilter_verdict=_PF())
    assert "tsqi" in v.voters                                     # a PPG voter now exists
    assert v.corrupt_override is True                             # confident-clean + prefiltered + dissent


# ---- batch 2: EEG blind SQIs, fused consensus, raw-vs-filtered -----------------

def test_eeg_breakdown_has_blind_sqis_and_line_index():
    fs = 256.0
    rng = np.random.default_rng(0)
    pink = np.cumsum(rng.standard_normal(int(fs * 8)))
    eeg = pink / (pink.std() + 1e-9) * 0.5
    names = [d["name"] for d in sqi_breakdown(eeg, fs, "eeg")]
    assert {"Aperiodic 1/f", "Hjorth comp.", "Line 50/60"} <= set(names)
    line = {d["name"]: d["value"] for d in sqi_breakdown(eeg, fs, "eeg")}["Line 50/60"]
    t = np.arange(int(fs * 8)) / fs
    line_dirty = {d["name"]: d["value"] for d in sqi_breakdown(eeg + 0.4 * np.sin(2 * np.pi * 50 * t), fs, "eeg")}["Line 50/60"]
    assert line_dirty > line                                      # 50 Hz mains raises the line index


def test_sqi_consensus_separates_clean_and_corrupt():
    clean = _ecg()
    corrupt = clean + 1.0 * np.random.default_rng(5).standard_normal(clean.size)
    assert sqi_consensus(sqi_breakdown(clean, FS, "ecg")) > 0.7   # clean bank → high consensus
    assert sqi_consensus(sqi_breakdown(corrupt, FS, "ecg")) < 0.5  # corrupt → discordance-triggering
    assert sqi_consensus([]) == -1.0                              # sentinel, not a real (0.0=corrupt) value


def test_sqi_bar_is_quality_fill_not_badness():
    """Regression: the display ``bar`` must be a QUALITY fill (1.0 = good), so a clean bSQI≈1.0 renders a
    FULL bar — fixing the confusing old 'badness' fill that showed an (near-)empty bar next to a max-looking
    value. Direction is uniform across 'higher=cleaner' and 'higher=noisier' indices (all: full = good)."""
    clean = _ecg()
    corrupt = clean + 0.8 * np.random.default_rng(1).standard_normal(clean.size)
    bc = {d["name"]: d["bar"] for d in sqi_breakdown(clean, FS, "ecg")}
    bx = {d["name"]: d["bar"] for d in sqi_breakdown(corrupt, FS, "ecg")}
    assert bc["bSQI"] > 0.66                                       # clean → (near-)full green bar
    assert bx["bSQI"] < bc["bSQI"]                                 # corrupt → shorter fill
    assert bx["HF noise"] < bc["HF noise"]                         # 'higher=noisier' too: corrupt fills less
    assert all(0.0 <= d["bar"] <= 1.0 for d in sqi_breakdown(clean, FS, "ecg"))


def test_sqi_consensus_excludes_informational_and_flags_worst_case():
    """Regression (review batch 2): informational rows (e.g. EDA skewness — inherently skewed on clean
    signals) must be EXCLUDED from the consensus so they can't force false discordance; and a consensus
    of exactly 0.0 (maximally corrupt) must remain a real flag, distinct from the -1 'no data' sentinel."""
    # a low-quality informational row must not drag the consensus down (bar is quality-fill now: 0 = poor;
    # if the informational 0.0 were counted this would be mean(0.9, 0.0) = 0.45 instead of 0.9)
    rows = [{"name": "Motion", "bar": 0.9, "informational": False},
            {"name": "Skewness", "bar": 0.0, "informational": True}]
    assert abs(sqi_consensus(rows) - 0.9) < 1e-9                 # informational row ignored → mean(0.9)
    # a real EDA breakdown actually tags its skewness row informational
    fs = 32.0
    eda = 2.0 + 0.3 * np.sin(2 * np.pi * 0.05 * np.arange(int(fs * 10)) / fs) \
        + 0.02 * np.random.default_rng(7).standard_normal(int(fs * 10))
    assert any(r.get("informational") for r in sqi_breakdown(eda, fs, "eda"))
    assert sqi_consensus([{"name": "x", "bar": 0.0}]) == 0.0     # maximally corrupt (empty fill) → real flag
    assert sqi_consensus([]) == -1.0                            # no data → sentinel


# ---- batch 3: per-modality task usability (EEG per-band, EDA tonic/phasic) -----

def test_eeg_band_usability_recovers_low_bands_under_muscle():
    fs = 256.0
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * 8)) / fs
    pink = np.cumsum(rng.standard_normal(t.size)); pink /= pink.std()
    clean = 0.5 * pink + 1.2 * np.sin(2 * np.pi * 10 * t)               # 1/f + real alpha
    bands = band_usability(clean, fs)
    assert [b["band"] for b in bands] == ["delta", "theta", "alpha", "beta", "gamma"]
    assert all(b["usable"] for b in bands)                             # clean → every band usable
    muscle = _bandpass_fft(rng.standard_normal(t.size), fs, 20.0, 120.0); muscle /= muscle.std()
    m = {b["band"]: b["usable"] for b in band_usability(0.8 * pink + 2.0 * muscle, fs)}
    assert m["delta"] and m["theta"] and m["alpha"]                    # low bands survive broadband muscle
    assert not m["beta"] and not m["gamma"]                            # high bands swamped → not usable


def test_eeg_band_usability_line_flags_only_gamma():
    fs = 256.0
    rng = np.random.default_rng(1)
    t = np.arange(int(fs * 8)) / fs
    pink = np.cumsum(rng.standard_normal(t.size)); pink /= pink.std()
    line = {b["band"]: b["usable"] for b in band_usability(0.5 * pink + 0.6 * np.sin(2 * np.pi * 50 * t), fs)}
    assert line["beta"] and not line["gamma"]                          # narrowband 50 Hz hits only γ, not β


def test_eda_component_usability_states():
    fs = 32.0
    rng = np.random.default_rng(2)
    et = np.arange(int(fs * 20)) / fs
    clean = 3.0 + 0.2 * np.sin(2 * np.pi * 0.01 * et) + 0.4 * np.exp(-((et - 6) ** 2) / (2 * 1.0 ** 2)) \
        + 0.005 * rng.standard_normal(et.size)
    verdicts = eda_component_usability(clean, fs)
    assert [v["component"] for v in verdicts] == ["tonic", "phasic"]
    assert all(v["usable"] for v in verdicts)                          # clean EDA → both usable
    flat = {v["component"]: v["usable"] for v in eda_component_usability(np.zeros(int(fs * 20)), fs)}
    assert not flat["tonic"] and not flat["phasic"]                    # flatline → neither
    motion = 3.0 + 0.2 * np.sin(2 * np.pi * 0.01 * et) + 2.0 * rng.standard_normal(et.size)
    assert eda_component_usability(motion, fs)[1]["usable"] is False   # motion → phasic not usable


def test_batch3_review_regressions():
    """Lock in the batch-3 adversarial-review fixes."""
    fs_e = 256.0
    rng = np.random.default_rng(9)
    t = np.arange(int(fs_e * 8)) / fs_e
    # Finding 2: pure broadband noise must NOT pass the low bands (no 1/f structure)
    noise = {b["band"]: b["usable"] for b in band_usability(rng.standard_normal(t.size), fs_e)}
    assert not any(noise[k] for k in ("delta", "theta", "alpha", "beta", "gamma"))
    efs = 32.0
    et = np.arange(int(efs * 20)) / efs
    # Finding 1: a physiological 0.1 µS/s SCL rise must stay tonic-usable (not falsely 3.2 µS/s)
    rise = 2.0 + 0.1 * et + 0.005 * rng.standard_normal(et.size)
    assert eda_component_usability(rise, efs)[0]["usable"] is True
    # Finding 3: a pure drift ramp (no real SCRs) must not report spurious SCRs
    ph = eda_component_usability(2.0 + 0.5 * et, efs)[1]
    assert "SCR" not in ph["detail"] or ph["detail"].startswith("quiet")


CLASS_ORDER = ["Q0", "Q1", "Q2", "Q3"]


def test_temperature_scale_recovers_scaled_softmax():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((5, 4))
    raw = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    T = 0.4
    direct = np.exp(logits / T) / np.exp(logits / T).sum(1, keepdims=True)   # temperature on the logits
    via_probs = temperature_scale(raw, T)                                    # temperature from probs alone
    assert np.allclose(direct, via_probs, atol=1e-9)                         # exact recovery
    assert np.allclose(temperature_scale(raw, 1.0), raw)                     # T=1 is a no-op
    assert np.allclose(temperature_scale(raw, None), raw)


def test_aps_prediction_set_confident_vs_ambiguous():
    assert aps_prediction_set([0.01, 0.01, 0.01, 0.97], CLASS_ORDER, 0.9682) == ("Q3",)   # peaked → size 1
    amb = aps_prediction_set([0.02, 0.03, 0.47, 0.48], CLASS_ORDER, 0.9682)
    assert len(amb) >= 2 and "Q2" in amb and "Q3" in amb                    # can't separate Q2/Q3
    assert aps_prediction_set([0.25] * 4, CLASS_ORDER, None) == ()          # no threshold → empty
    assert aps_prediction_set([], CLASS_ORDER, 0.9) == ()
    # float-summation slop: 0.6+0.3 = 0.8999… must still count as reaching 0.9 (review nit)
    assert aps_prediction_set([0.6, 0.3, 0.05, 0.05], CLASS_ORDER, 0.9) == ("Q0", "Q1")


def test_conformal_set_flows_through_rle():
    tiers = np.array(["Q3", "Q3", "Q1"])
    conf = np.array([0.9, 0.9, 0.5])
    gp = temperature_scale(np.array([[0.01, 0.01, 0.01, 0.97], [0.02, 0.02, 0.02, 0.94],
                                     [0.03, 0.04, 0.45, 0.48]]), 0.4)
    ivs = run_length_encode(tiers, conf, 10.0, 10.0, grade_probs_per_window=gp,
                            class_order=CLASS_ORDER, conformal_threshold=0.9682)
    assert ivs[0].conformal_set == ("Q3",) and not ivs[0].ambiguous
    assert ivs[1].ambiguous and len(ivs[1].conformal_set) >= 2
    # no threshold → empty sets, ambiguous False (feature gracefully absent)
    plain = run_length_encode(tiers, conf, 10.0, 10.0)
    assert all(iv.conformal_set == () and not iv.ambiguous for iv in plain)


def test_model_card_parses_conformal_only_where_shipped():
    from pathlib import Path

    from biosqa.model.model_card import load_model_card
    models = Path(__file__).resolve().parents[1] / "models"
    ecg = load_model_card(models / "ecg.model_card.json")
    assert ecg.conformal_threshold is not None and abs(ecg.conformal_threshold - 0.9682) < 1e-6
    assert abs(ecg.grade_temperature - 0.4) < 1e-9 and abs(ecg.conformal_alpha - 0.1) < 1e-9
    # PPG app model predates the conformal calibration → no threshold (feature stays off, no crash)
    assert load_model_card(models / "ppg.model_card.json").conformal_threshold is None


def test_novelty_distance_flags_ood_and_explains():
    """Feature-space Mahalanobis novelty: pure noise fed as ECG is far outside the training SQI manifold
    and is NAMED by its dominant SQI; a real-ish beat is not; guards handle missing/mismatched blocks."""
    from pathlib import Path

    from biosqa.model.model_card import load_model_card
    blk = load_model_card(Path(__file__).resolve().parents[1] / "models" / "ecg.model_card.json").novelty
    assert blk and blk["method"] == "mahalanobis_sqi" and len(blk["feature_names"]) == 8
    fs = 250.0
    rng = np.random.default_rng(1)
    fb, nb = sqi_feature_vector(_ecg(), fs, "ecg")
    fn, nn = sqi_feature_vector(rng.standard_normal(2500), fs, "ecg")
    assert nb == blk["feature_names"]                  # runtime SQI names match the shipped reference
    d2_beat, _ = novelty_distance(fb, blk, nb)
    d2_noise, top = novelty_distance(fn, blk, nn)
    assert d2_noise > blk["d2_threshold"] > 0          # pure noise is novel (beyond the calibrated cutoff)
    assert d2_noise > d2_beat                          # …and more novel than a real-ish beat
    assert top in blk["feature_names"]                 # the explaining SQI is a real feature name
    # fail-safe guards
    assert novelty_distance([1.0, 2.0], blk) == (0.0, "")            # wrong length → skip
    assert novelty_distance([], None) == (0.0, "")                  # no block → skip
    reordered = list(reversed(blk["feature_names"]))                # same length, different ORDER
    assert novelty_distance(fb, blk, reordered) == (0.0, "")        # reorder → skip, not silent mis-score
    assert novelty_distance(fb, {"mean": [0, 0]}) == (0.0, "")      # partial block → skip, no KeyError


def test_input_sanity_domain_shift_index():
    """DSI flags ALIASING (robust across modalities) WITHOUT false-positiving on narrow-band signals."""
    fs = 250.0
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * 30)) / fs
    ecg = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(t.size)
    for i in range(int(0.4 * fs), t.size - 3, int(fs * 60 / 70)):
        ecg[i:i + 3] += 1.3
    assert input_sanity(ecg, fs).dsi < 0.2                              # broadband in-regime → low DSI
    aliased = ecg + 0.8 * np.sin(2 * np.pi * 120 * t)                   # strong near-Nyquist energy
    ra = input_sanity(aliased, fs)
    assert ra.dsi > 0.3 and ra.near_nyquist_frac > 0.2 and ra.flags     # aliasing flagged
    # narrow-band NATURAL modalities must NOT false-positive (the review-caught bug: EEG/PPG were flagged)
    et = np.arange(int(256 * 20)) / 256.0
    pink = np.cumsum(rng.standard_normal(et.size)); pink /= pink.std()
    assert input_sanity(0.5 * pink + 1.2 * np.sin(2 * np.pi * 10 * et), 256.0).dsi < 0.1   # clean EEG
    pt = np.arange(int(125 * 20)) / 125.0
    assert input_sanity(0.5 * (1 + np.sin(2 * np.pi * 1.2 * pt)) ** 2, 125.0).dsi < 0.1    # clean PPG
    # over-filtered ECG is owned by the pre-filter detector, not this aliasing check
    assert input_sanity(filter_for_modality(ecg, fs, "ecg"), fs).dsi < 0.1
    # broadband white noise (flat spectrum, ~10% power in the top decile) must NOT read as aliasing
    assert input_sanity(rng.standard_normal(int(fs * 20)), fs).dsi < 0.1
    assert input_sanity(np.ones(int(fs * 5)), fs).dsi == 0.0            # constant → safe, no false alarm
    assert input_sanity(np.ones(8), fs).dsi == 0.0                      # too short → safe


def test_usability_verdicts_dispatch_shapes():
    fs_e = 256.0
    eeg = np.random.default_rng(3).standard_normal(int(fs_e * 8))
    assert len(usability_verdicts(eeg, fs_e, "eeg")) == 5              # per-band
    assert len(usability_verdicts(np.zeros(640), 32.0, "eda")) == 2    # tonic/phasic
    assert usability_verdicts(_ecg(), FS, "ecg") == []                 # ECG/PPG use the rate-usable card
    for v in usability_verdicts(eeg, fs_e, "eeg"):
        assert {"label", "usable", "detail"} <= set(v)


def test_filtered_view_lifts_baseline_wander():
    """The Raw/Filtered toggle's value: a filter clears baseline wander (basSQI ↑) but not in-band noise."""
    base = _ecg()
    wander = base + 3.0 * np.sin(2 * np.pi * 0.2 * np.arange(base.size) / FS)
    raw = {d["name"]: d["value"] for d in sqi_breakdown(wander, FS, "ecg")}["basSQI"]
    filt_sig = filter_for_modality(wander, FS, "ecg")
    filt = {d["name"]: d["value"] for d in sqi_breakdown(filt_sig, FS, "ecg")}["basSQI"]
    assert filt > raw + 0.3                                       # filtering restores the baseline SQI
