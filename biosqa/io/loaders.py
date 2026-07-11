"""Foreign-format readers: WFDB / the whole MNE family / Parquet / Zarr, ``preload=False``.

A single ``mne.io.read_raw`` opener covers EDF/EDF+, BDF/BDF+, GDF, BrainVision (.vhdr),
EEGLAB (.set), and FIF — MNE auto-dispatches by content. WFDB (.hea/.dat) is read by wfdb.
The canonical Zarr store + Parquet interval/signal tables are read directly (deps already present).
Opening is header-only (``<2s`` regardless of length); sample windows are read lazily via
``read_window``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Eager, shallow-stack import of the heavy readers. The per-format openers below still write
# ``import wfdb`` / ``import mne`` locally, but once these have run that is a trivial ``sys.modules``
# cache hit. Importing them HERE — at module-import time, i.e. the shallow process-startup stack
# (main.py imports this transitively before the window exists) — is what makes those cache hits
# safe: in the frozen (PyInstaller) build the FIRST-EVER import of the wfdb→pandas→pyarrow chain,
# triggered from DEEP inside the native file-dialog accept callback, overflowed the call stack and
# killed the app with exit 0xC00000FD (STACK_OVERFLOW). Loading them up front removes that deep
# first import entirely (a background pre-warm thread was racy — this is deterministic).
try:  # best-effort: a stripped test env may lack one; the local in-function import re-raises clearly.
    import wfdb  # noqa: F401
    import mne  # noqa: F401
except Exception:  # noqa: BLE001
    pass


@dataclass
class RecordingHandle:
    """A lazily-opened recording: enough metadata to build the file tree / channel list
    without materializing samples."""

    path: Path
    format: str  # "wfdb" | "mne" | "zarr" | "parquet"
    channel_names: list[str]
    fs_hz: dict[str, float]
    n_samples: dict[str, int]
    backend: object  # the underlying wfdb/mne/zarr/pyarrow handle, format-specific
    units: dict[str, str] = field(default_factory=dict)  # channel -> physical unit (uV/mV/uS/...)


@dataclass
class FormatGuess:
    """Result of content-sniffing a file: which reader to use + how sure we are."""

    format: str          # "wfdb" | "mne" | "zarr" | "parquet" | "dicom" | "unknown"
    confidence: float    # 0..1
    reason: str = ""     # what matched (magic bytes / extension / sibling)


# --- companion-file resolution: a dropped side-car resolves to its header -------------
_SIBLING = {".dat": ".hea", ".eeg": ".vhdr", ".vmrk": ".vhdr", ".fdt": ".set"}

# extension -> reader family (final fallback if content sniffing is inconclusive)
_EXT_FMT = {
    ".hea": "wfdb", ".dat": "wfdb",
    ".edf": "mne", ".bdf": "mne", ".gdf": "mne", ".vhdr": "mne", ".set": "mne",
    ".fif": "mne", ".fif.gz": "mne",
    ".zarr": "zarr", ".parquet": "parquet", ".dcm": "dicom",
}
_MNE_EXTS = (".edf", ".bdf", ".gdf", ".vhdr", ".set", ".fif")


def _resolve_companion(path: Path) -> Path:
    """A dropped side-car file (.dat/.eeg/.vmrk/.fdt) resolves to its text header."""
    sib = _SIBLING.get(path.suffix.lower())
    if sib:
        cand = path.with_suffix(sib)
        if cand.exists():
            return cand
    return path


def sniff_format(path: str | Path) -> str:
    """Back-compat thin wrapper: return just the reader family string."""
    return guess_format(path).format


def guess_format(path: str | Path) -> FormatGuess:
    """Content-sniff a recording: resolve companions, test binary magics + text structure,
    fall back to the extension map. Returns a :class:`FormatGuess`."""
    p = _resolve_companion(Path(path))
    suffix = "".join(p.suffixes[-2:]).lower() if p.name.lower().endswith(".fif.gz") else p.suffix.lower()

    if p.is_dir() or suffix == ".zarr":
        if (p / "zarr.json").exists() or (p / ".zgroup").exists() or suffix == ".zarr":
            return FormatGuess("zarr", 0.95, "zarr store dir")

    head = b""
    try:
        with open(p, "rb") as fh:
            head = fh.read(512)
    except OSError:
        pass

    if len(head) >= 132 and head[128:132] == b"DICM":
        return FormatGuess("dicom", 0.99, "DICM@128")
    if head[:4] == b"PAR1":
        return FormatGuess("parquet", 0.99, "PAR1")
    if head[:1] == b"\xff" and head[1:8] == b"BIOSEMI":
        return FormatGuess("mne", 0.99, "BDF BIOSEMI")            # 24-bit BioSemi -> read_raw_bdf
    if head[:8] == b"0       ":                                   # '0' + 7 spaces = EDF version field
        return FormatGuess("mne", 0.97, "EDF header")
    if head[:3] == b"GDF":
        return FormatGuess("mne", 0.97, "GDF header")
    if head[:8].startswith(b"MATLAB 5") or head[:8] == b"\x89HDF\r\n\x1a\n":
        return FormatGuess("mne", 0.6, "MAT/HDF5 (EEGLAB .set)") if suffix == ".set" else FormatGuess("unknown", 0.3, "mat/hdf5")
    try:
        txt = head.decode("latin-1", "ignore")
        if txt.startswith("Brain Vision Data Exchange Header"):
            return FormatGuess("mne", 0.97, "BrainVision header")
        toks = txt.split()
        if len(toks) >= 3 and all(t.replace(".", "", 1).isdigit() for t in toks[1:3]):
            if Path(path).with_suffix(".dat").exists() or suffix in (".hea", ".dat"):
                return FormatGuess("wfdb", 0.9, "WFDB record line")
    except Exception:  # noqa: BLE001
        pass

    fmt = _EXT_FMT.get(suffix, "unknown")
    return FormatGuess(fmt, 0.5 if fmt != "unknown" else 0.0, f"extension {suffix}")


# --- WFDB ------------------------------------------------------------------------------
def open_wfdb(path: str | Path) -> RecordingHandle:
    """Open a WFDB record header (header-only, <2s regardless of length)."""
    import wfdb

    record_path = _resolve_companion(Path(path))
    header = wfdb.rdheader(str(record_path.with_suffix("")))
    fs = float(header.fs)
    names = list(header.sig_name)
    raw_units = list(getattr(header, "units", []) or [])
    units = {n: (raw_units[i] if i < len(raw_units) else "") for i, n in enumerate(names)}
    return RecordingHandle(
        path=record_path, format="wfdb", channel_names=names,
        fs_hz={n: fs for n in names}, n_samples={n: int(header.sig_len) for n in names},
        backend=header, units=units,
    )


def read_wfdb_window(handle, channels, start_sample, end_sample):
    import wfdb
    signal, _fields = wfdb.rdsamp(
        str(handle.path.with_suffix("")), sampfrom=start_sample, sampto=end_sample,
        channel_names=channels,
    )
    return np.asarray(signal)


# --- MNE family (EDF/BDF/GDF/BrainVision/EEGLAB/FIF) ------------------------------------
def open_mne(path: str | Path) -> RecordingHandle:
    """Open any MNE-readable file with ``preload=False``. ``mne.io.read_raw`` auto-dispatches
    (read_raw_edf / _bdf / _gdf / _brainvision / _eeglab / _fif) by extension+content."""
    import mne

    p = _resolve_companion(Path(path))
    raw = mne.io.read_raw(str(p), preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    orig_units = getattr(raw, "_orig_units", {}) or {}
    units = {n: str(orig_units.get(n, "")) for n in names}
    return RecordingHandle(
        path=p, format="mne", channel_names=names,
        fs_hz={n: fs for n in names}, n_samples={n: int(raw.n_times) for n in names},
        backend=raw, units=units,
    )


def read_mne_window(handle, channels, start_sample, end_sample):
    raw = handle.backend
    picks = [raw.ch_names.index(ch) for ch in channels]
    data, _times = raw[picks, start_sample:end_sample]
    return np.asarray(data).T


# --- Parquet (signal table: numeric columns = channels) --------------------------------
_TIME_COLS = ("time", "t", "timestamp", "seconds", "sec")


def open_parquet(path: str | Path) -> RecordingHandle:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    names = [f.name for f in schema if f.name.lower() not in _TIME_COLS]
    n = pf.metadata.num_rows
    fs = 250.0
    tcol = next((f.name for f in schema if f.name.lower() in _TIME_COLS), None)
    if tcol is not None and n > 1:
        # Estimate fs from the FIRST row group only (keeps open header-only / <2 s); reading the
        # whole time column on a multi-million-row Parquet would block the GUI thread.
        rg0 = pf.read_row_group(0, columns=[tcol]) if pf.metadata.num_row_groups else pf.read([tcol])
        t = rg0.column(0).to_numpy()
        dt = float(np.median(np.diff(t[: min(len(t), 4096)]))) if len(t) > 1 else 0.0
        if dt > 0:
            fs = 1.0 / dt
    return RecordingHandle(
        path=Path(path), format="parquet", channel_names=names,
        fs_hz={c: fs for c in names}, n_samples={c: int(n) for c in names},
        backend=str(path), units={c: "" for c in names},
    )


def read_parquet_window(handle, channels, start_sample, end_sample):
    """Windowed Parquet read: load ONLY the row groups overlapping ``[start_sample, end_sample)``
    (not the whole column), so large Parquet files are actually lazy/windowed. Falls back to the
    single row group a file may have."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(handle.backend)
    md = pf.metadata
    start_sample, end_sample = int(start_sample), int(end_sample)
    offsets = [0]
    for i in range(md.num_row_groups):
        offsets.append(offsets[-1] + md.row_group(i).num_rows)
    groups = [i for i in range(md.num_row_groups)
              if offsets[i] < end_sample and offsets[i + 1] > start_sample]
    if not groups:
        return np.empty((0, len(channels)), dtype=np.float64)
    tbl = pf.read_row_groups(groups, columns=list(channels))
    base = offsets[groups[0]]                       # global sample index of the first read row
    lo, hi = start_sample - base, end_sample - base
    cols = [tbl.column(c).to_numpy(zero_copy_only=False)[lo:hi] for c in channels]
    return np.asarray(cols, dtype=np.float64).T


# --- Zarr (canonical store: top-level arrays = channels) -------------------------------
def open_zarr(path: str | Path) -> RecordingHandle:
    import zarr

    g = zarr.open_group(str(path), mode="r")
    attrs = dict(g.attrs)
    fs = float(attrs.get("fs", attrs.get("sfreq", 250.0)) or 250.0)
    names = [k for k, v in g.arrays()] if hasattr(g, "arrays") else list(g.array_keys())
    n_samples = {k: int(g[k].shape[0]) for k in names}
    return RecordingHandle(
        path=Path(path), format="zarr", channel_names=names,
        fs_hz={c: fs for c in names}, n_samples=n_samples,
        backend=g, units={c: "" for c in names},
    )


def read_zarr_window(handle, channels, start_sample, end_sample):
    g = handle.backend
    cols = [np.asarray(g[c][start_sample:end_sample]) for c in channels]
    return np.asarray(cols, dtype=np.float64).T


# --- dispatch --------------------------------------------------------------------------
_OPENERS = {"wfdb": open_wfdb, "mne": open_mne, "parquet": open_parquet, "zarr": open_zarr}
_READERS = {"wfdb": read_wfdb_window, "mne": read_mne_window,
            "parquet": read_parquet_window, "zarr": read_zarr_window}


def open_recording(path: str | Path) -> RecordingHandle:
    """Content-sniff then dispatch to the right lazy opener."""
    fmt = guess_format(path).format
    opener = _OPENERS.get(fmt)
    if opener is None:
        raise NotImplementedError(
            f"open_recording: format {fmt!r} not supported (DICOM/E4/MAT need an optional extra)"
        )
    return opener(path)


def read_window(handle, channels, start_sample, end_sample):
    """Format-dispatching windowed read -> ``[n_samples, n_channels]`` float array."""
    reader = _READERS.get(handle.format)
    if reader is None:
        raise NotImplementedError(f"read_window: format {handle.format!r} not supported")
    return reader(handle, channels, start_sample, end_sample)


# Back-compat aliases (EDF was the old name for the whole mne family).
open_edf = open_mne
read_edf_window = read_mne_window


# --- modality detection: weighted multi-cue vote ---------------------------------------
_SUBSTR_TOKENS = {
    "ecg": ("ecg", "ekg", "mlii"),
    "ppg": ("ppg", "pleth", "bvp", "pulse", "spo2"),
    "eda": ("eda", "gsr", "scl", "scr", "electroderm", "skin cond"),
    "eeg": ("eeg", "fp1", "fp2", "fpz", "cz", "oz", "pz", "fz", "o1", "o2", "c3", "c4", "eog", "emg"),
}
_WORD_TOKENS = {
    "ecg": ("i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6", "avr", "avl", "avf", "lead"),
}
_MODALITY_FS = {"ecg": 250.0, "ppg": 64.0, "eeg": 256.0, "eda": 8.0}


def _unit_to_modality(u: str) -> str | None:
    """Map a physical unit string to a modality (the strongest cue)."""
    u = (u or "").strip().lower().replace("µ", "u").replace(" ", "")
    if u in ("uv", "microvolt", "microvolts"):
        return "eeg"
    if u in ("mv", "millivolt", "millivolts"):
        return "ecg"
    if u in ("us", "microsiemens", "umho", "umhos", "us/cm"):
        return "eda"
    if u in ("na", "nampere", "adu", "counts", "arb", "arbitrary"):
        return "ppg"
    return None


def modality_vote(handle: RecordingHandle) -> tuple[str, float]:
    """Weighted vote over physical units (strongest), channel-name tokens, and fs (weak
    tie-break). Returns ``(modality, confidence 0..1)``. Confidence is low/0 when nothing
    but the fs tie-break fired — the caller should then ask the user instead of guessing."""
    scores = {m: 0.0 for m in _MODALITY_FS}
    # (1) physical units — highest weight
    for u in handle.units.values():
        m = _unit_to_modality(u)
        if m:
            scores[m] += 3.0
    # (2) channel-name tokens
    padded = " " + " ".join(n.lower().replace("-", " ") for n in handle.channel_names) + " "
    for m, toks in _SUBSTR_TOKENS.items():
        scores[m] += 2.0 * sum(padded.count(t) for t in toks)
    for m, toks in _WORD_TOKENS.items():
        scores[m] += 2.0 * sum(padded.count(f" {t} ") for t in toks)
    strong_total = sum(scores.values())
    # (3) sampling-rate nearest canonical — weak tie-break only
    fs = float(next(iter(handle.fs_hz.values()), 250.0)) or 250.0
    nearest = min(_MODALITY_FS, key=lambda m: abs(np.log(fs) - np.log(_MODALITY_FS[m])))
    scores[nearest] += 0.5
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = (scores[best] / total) if total > 0 else 0.0
    # if only the fs tie-break fired, confidence is genuinely low
    if strong_total == 0:
        confidence = min(confidence, 0.35)
    return best, float(confidence)


def detect_modality(handle: RecordingHandle) -> str:
    """Back-compat: the winning modality string from :func:`modality_vote`."""
    return modality_vote(handle)[0]
