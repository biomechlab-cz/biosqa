"""Synthetic test recordings for exercising the app end to end (Plan 2 §14 test aid).

``write_test_ecg`` generates a multi-minute single-lead ECG with realistic PQRST
morphology and a sequence of deliberately quality-varying regions — clean, gross
motion, baseline wander, muscle/EMG, and amplitude clipping — so the whole pipeline
(decimation, pan/zoom, sliding-window inference, run-length segmentation) has
something long and heterogeneous to chew on. Writes a WFDB record to a temp dir and
returns the ``.hea`` path, ready for ``RecordingListModel.open``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def _gauss(phase: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-(((phase - center) / width) ** 2))


def make_long_ecg(minutes: float = 5.0, fs: int = 360, hr_bpm: float = 68.0, seed: int = 7):
    """Return ``(signal, fs)`` — a quality-varying single-lead ECG."""
    rng = np.random.default_rng(seed)
    n = int(round(minutes * 60 * fs))
    t = np.arange(n) / fs

    # --- base ECG: a periodic PQRST built from gaussians, with mild HR variation ----
    beat = 60.0 / hr_bpm
    # slowly-varying instantaneous phase so beats aren't perfectly metronomic
    hr_wobble = 1.0 + 0.04 * np.sin(2 * np.pi * 0.05 * t)
    phase = np.mod(np.cumsum(hr_wobble) / (fs * beat), 1.0)
    ecg = (
        _gauss(phase, 0.18, 0.022, 0.12)     # P
        + _gauss(phase, 0.37, 0.008, -0.16)  # Q
        + _gauss(phase, 0.40, 0.009, 1.00)   # R
        + _gauss(phase, 0.43, 0.012, -0.28)  # S
        + _gauss(phase, 0.62, 0.045, 0.32)   # T
    )
    ecg = ecg + 0.015 * rng.standard_normal(n)

    def seg(a: float, b: float) -> slice:
        return slice(int(a * n), int(b * n))

    # --- quality-varying regions (fractions of the recording) -----------------------
    # clean .00-.22 | motion .22-.32 | clean .32-.50 | baseline wander .50-.60 |
    # clean .60-.74 | muscle/EMG .74-.82 | clean .82-.90 | clipping .90-.95 | clean .95-1
    mot = seg(0.22, 0.32)
    ecg[mot] += 2.6 * rng.standard_normal(ecg[mot].shape[0])
    ecg[mot] += 1.4 * np.sin(2 * np.pi * 3.0 * t[mot])

    bw = seg(0.50, 0.60)
    ecg[bw] += 0.9 * np.sin(2 * np.pi * 0.33 * t[bw]) + 0.5 * np.sin(2 * np.pi * 0.15 * t[bw])

    emg = seg(0.74, 0.82)
    ecg[emg] += 0.55 * rng.standard_normal(ecg[emg].shape[0])  # broadband muscle

    clip = seg(0.90, 0.95)
    ecg[clip] = np.clip(ecg[clip] * 6.0, -1.6, 1.6)  # saturation

    return ecg.astype(np.float64), fs


def make_long_ppg(minutes: float = 5.0, fs: int = 64, hr_bpm: float = 72.0, seed: int = 11):
    """Quality-varying pulsatile PPG (systolic peak + dicrotic wave)."""
    rng = np.random.default_rng(seed)
    n = int(round(minutes * 60 * fs)); t = np.arange(n) / fs
    beat = 60.0 / hr_bpm
    phase = np.mod(np.cumsum(1.0 + 0.05 * np.sin(2 * np.pi * 0.06 * t)) / (fs * beat), 1.0)
    ppg = _gauss(phase, 0.30, 0.13, 1.0) + _gauss(phase, 0.58, 0.10, 0.34)
    ppg = ppg - 0.42 + 0.01 * rng.standard_normal(n)

    def seg(a, b):
        return slice(int(a * n), int(b * n))
    mot = seg(0.24, 0.33); ppg[mot] += 1.8 * rng.standard_normal(ppg[mot].shape[0])
    clp = seg(0.55, 0.62); ppg[clp] = np.clip(ppg[clp] * 4.0, -1.0, 1.0)
    drp = seg(0.80, 0.86); ppg[drp] = 0.02 * rng.standard_normal(ppg[drp].shape[0])
    return ppg.astype(np.float64), fs


def make_long_eeg(minutes: float = 5.0, fs: int = 256, seed: int = 13):
    """Quality-varying EEG (alpha/beta/theta mix) with muscle / eye / motion regions."""
    rng = np.random.default_rng(seed)
    n = int(round(minutes * 60 * fs)); t = np.arange(n) / fs
    eeg = (0.5 * np.sin(2 * np.pi * 10 * t) + 0.22 * np.sin(2 * np.pi * 20 * t + 1)
           + 0.16 * np.sin(2 * np.pi * 6 * t + 2) + 0.22 * rng.standard_normal(n))

    def seg(a, b):
        return slice(int(a * n), int(b * n))
    mus = seg(0.22, 0.30); eeg[mus] += 1.3 * rng.standard_normal(eeg[mus].shape[0])
    eye = seg(0.50, 0.57)  # slow high-amplitude ocular deflections
    tt = t[eye] - t[eye][0]
    eeg[eye] += 1.6 * np.sin(2 * np.pi * 0.5 * tt) * np.exp(-((np.mod(tt, 2.0) - 1.0) ** 2) / 0.1)
    mot = seg(0.74, 0.81); eeg[mot] += 2.0 * np.sin(2 * np.pi * 2.0 * t[mot]) \
        + 0.7 * rng.standard_normal(eeg[mot].shape[0])
    return eeg.astype(np.float64), fs


def make_long_eda(minutes: float = 5.0, fs: int = 8, seed: int = 17):
    """Quality-varying EDA: slow tonic level + phasic SCR bursts, with a motion region."""
    rng = np.random.default_rng(seed)
    n = int(round(minutes * 60 * fs)); t = np.arange(n) / fs
    eda = 2.0 + 0.5 * np.sin(2 * np.pi * t / 130.0)
    idx = np.arange(n)
    for _ in range(max(3, int(minutes * 3))):
        onset = int(rng.integers(0, n)); rel = idx - onset
        scr = np.where(rel >= 0, (1 - np.exp(-rel / (1.0 * fs))) * np.exp(-rel / (5.0 * fs)), 0.0)
        eda = eda + scr * float(rng.uniform(0.6, 1.6))

    def seg(a, b):
        return slice(int(a * n), int(b * n))
    mot = seg(0.30, 0.40); eda[mot] += 1.6 * rng.standard_normal(eda[mot].shape[0])
    return eda.astype(np.float64), fs


#: modality -> (generator, WFDB channel name that ``detect_modality`` recognizes, unit)
_GENERATORS = {
    "ecg": (make_long_ecg, "II", "mV"),
    "ppg": (make_long_ppg, "PLETH", "NU"),
    "eeg": (make_long_eeg, "Fp1", "uV"),
    "eda": (make_long_eda, "EDA", "uS"),
}


def write_test_recording(modality: str = "ecg", dirpath: str | Path | None = None,
                         minutes: float = 5.0) -> str:
    """Synthesize a long quality-varying recording for ``modality`` (ecg|ppg|eeg|eda),
    write it as WFDB (channel named so ``detect_modality`` routes it), return the ``.hea``."""
    import wfdb

    modality = (modality or "ecg").lower()
    gen, chname, unit = _GENERATORS.get(modality, _GENERATORS["ecg"])
    sig, fs = gen(minutes=minutes)
    # stable per-modality path so re-clicking a test-data button re-selects the existing
    # row (dedup in RecordingListModel.open) instead of appending a new duplicate each time.
    out_dir = Path(dirpath) if dirpath else (Path(tempfile.gettempdir()) / "biosqa_testdata")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"test_{modality}_{int(minutes)}min"
    wfdb.wrsamp(name, fs=fs, units=[unit], sig_name=[chname],
                p_signal=sig[:, None], write_dir=str(out_dir))
    return str(out_dir / f"{name}.hea")


def write_test_ecg(dirpath: str | Path | None = None, minutes: float = 5.0) -> str:
    """Back-compat: synthesize + write a long ECG (delegates to write_test_recording)."""
    return write_test_recording("ecg", dirpath=dirpath, minutes=minutes)
