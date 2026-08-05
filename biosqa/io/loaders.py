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
    # channel name -> POSITIONAL index in the underlying file. Every dict above is keyed by name, so a
    # format that allows duplicate signal names (WFDB does) would alias two leads onto one entry; the
    # openers de-duplicate the display names and record the true index here so reads stay positional.
    channel_index: dict[str, int] = field(default_factory=dict)


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
def _unique_names(names: list[str]) -> list[str]:
    """Make signal names unique by appending a positional ``#k`` to repeats.

    WFDB headers do NOT guarantee distinct ``sig_name`` entries — 44 records in the Long Term ST
    Database are literally ``['ECG', 'ECG']``. Every per-channel dict on :class:`RecordingHandle` is
    keyed by name, so duplicates collapse onto one entry and the second lead becomes unreachable (the
    Coordinator's ``[n for n in names if n != analyzed]`` dedup drops it entirely). Suffixing keeps one
    key per physical channel; :attr:`RecordingHandle.channel_index` carries the true position."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def open_wfdb(path: str | Path) -> RecordingHandle:
    """Open a WFDB record header (header-only, <2s regardless of length)."""
    import wfdb

    record_path = _resolve_companion(Path(path))
    header = wfdb.rdheader(str(record_path.with_suffix("")))
    fs = float(header.fs)
    # Refuse a header that is syntactically valid but describes something no analysis
    # can consume, rather than handing the UI a confident-looking channel list.
    # BUT PPG's records declare `<name> 10000 1000 1` -- TEN THOUSAND signals of ONE
    # sample each, with no signal names, because the cohort stores one record per
    # matrix row. That opened cleanly and populated the channel panel with 10 000
    # entries named None, None#2 ... None#10000, every one of them a single sample.
    # This is the same failure class as the zarr path (accept and return garbage) but
    # on WFDB, which is the app's primary advertised input and is reachable straight
    # from the file dialog.
    # The floor is 2, not a window length: "short but real" is already reported loudly
    # downstream with a better message. This rejects only what cannot be a signal.
    n_samples = int(header.sig_len)
    n_sig = int(getattr(header, "n_sig", 0) or 0)
    if n_samples < 2:
        raise ValueError(
            f"{record_path.name}: the header declares {n_sig} signal(s) of {n_samples} "
            f"sample(s) each at {fs:g} Hz — that is not a recording. This layout usually "
            "means the file stores one record per row (BUT PPG does); open an individual "
            "record instead."
        )
    names = _unique_names(list(header.sig_name))
    raw_units = list(getattr(header, "units", []) or [])
    units = {n: (raw_units[i] if i < len(raw_units) else "") for i, n in enumerate(names)}
    return RecordingHandle(
        path=record_path, format="wfdb", channel_names=names,
        fs_hz={n: fs for n in names}, n_samples={n: int(header.sig_len) for n in names},
        backend=header, units=units, channel_index={n: i for i, n in enumerate(names)},
    )


def read_wfdb_window(handle, channels, start_sample, end_sample):
    import wfdb
    # Address channels by INDEX, not by name: ``channel_names=`` resolves through
    # ``sig_name.index(name)`` and so always returns the FIRST match, silently serving lead 0's
    # samples for every repeat of a duplicated name.
    idx = getattr(handle, "channel_index", None) or {}
    picks = [int(idx[c]) if c in idx else handle.channel_names.index(c) for c in channels]
    signal, _fields = wfdb.rdsamp(
        str(handle.path.with_suffix("")), sampfrom=start_sample, sampto=end_sample,
        channels=picks,
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
# Plausible band for a derived sampling rate. Nothing this app grades is slower than a 0.5 Hz EDA
# logger or faster than a 20 kHz intracranial/audio-rate acquisition, so a value outside this is a
# UNIT error in the time column, never a real rate.
_FS_MIN_HZ, _FS_MAX_HZ = 0.5, 20000.0
# Decimal units a bare numeric time column is ever written in, and the seconds-per-tick of each.
_TIME_UNITS = (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9))


def _fs_from_time_column(t: np.ndarray, tcol: str, path) -> float:
    """Derive the sampling rate from a time column, with the unit resolved EXPLICITLY.

    ``np.median(np.diff(t))`` taken at face value is wrong by orders of magnitude on the two most
    common real exports: a pandas ``datetime64[ns]`` timestamp reads as a 2.5e-07 Hz recording (which
    then asks ``resample_signal`` for a ~2e10-tap FIR and hangs the load worker outright), and a
    millisecond column — the normal wearable export — reads as 0.25 Hz, resamples cleanly, and makes
    every displayed and exported time wrong by exactly 1000x with no error anywhere.

    So: datetime64/timedelta64 carry their unit in the dtype and are converted exactly. A bare numeric
    column does not, so its tick is resolved by trying s -> ms -> us -> ns and taking the FIRST that
    lands the rate in the plausible band (seconds is tried first, so a genuine seconds column is never
    reinterpreted). When none does, the rate is genuinely unknown and that is raised, not defaulted.

    A TEXT time column (ISO-8601 timestamps — what a CSV -> Parquet round trip produces) is parsed as
    ``datetime64``, i.e. treated exactly like a native timestamp column. It used to fall through to
    ``np.asarray(t, dtype=float)`` and surface numpy's raw ``could not convert string to float:
    '1970-01-01 00:00:00.000'`` instead of this module's own wording."""
    if np.issubdtype(t.dtype, np.datetime64) or np.issubdtype(t.dtype, np.timedelta64):
        t = t.astype("timedelta64[ns]" if np.issubdtype(t.dtype, np.timedelta64)
                     else "datetime64[ns]").astype(np.int64)
        units = (("ns", 1e-9),)
    else:
        try:                                  # numeric first: an object column of Decimals still works
            t = np.asarray(t, dtype=np.float64)
        except (ValueError, TypeError):
            try:                              # ...then ISO-8601 text, converted exactly (ns ticks)
                t = np.asarray(t.astype("U"), dtype="datetime64[ns]").astype(np.int64)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{Path(path).name}: sampling rate unknown — the {tcol!r} column is neither "
                    f"numeric nor ISO-8601 timestamps (first value {t[0]!r}: {exc}). Provide a "
                    f"monotonically increasing time column in seconds."
                ) from None
            units = (("ns", 1e-9),)
        else:
            units = _TIME_UNITS
    dt = float(np.median(np.diff(t[: min(len(t), 4096)])))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"{Path(path).name}: sampling rate unknown — the {tcol!r} column does not increase "
            f"(median step {dt}). Provide a monotonically increasing time column."
        )
    for unit, tick in units:
        fs = 1.0 / (dt * tick)
        if _FS_MIN_HZ <= fs <= _FS_MAX_HZ:
            return fs
    raise ValueError(
        f"{Path(path).name}: sampling rate unknown — a median step of {dt:g} in the {tcol!r} column "
        f"gives no plausible rate in {_FS_MIN_HZ}-{_FS_MAX_HZ:g} Hz under any of "
        f"{', '.join(u for u, _ in units)}. Rewrite the time column in seconds."
    )


def open_parquet(path: str | Path) -> RecordingHandle:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    # Channels must be NUMERIC: a string/bool annotation column exposed as a channel only fails later,
    # deep inside read_parquet_window ("could not convert string to float").
    names = [f.name for f in schema
             if f.name.lower() not in _TIME_COLS
             and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    if not names:
        raise ValueError(f"{Path(path).name}: no numeric signal columns (have {schema.names!r})")
    n = pf.metadata.num_rows
    tcol = next((f.name for f in schema if f.name.lower() in _TIME_COLS), None)
    if tcol is None or n <= 1:
        raise ValueError(
            f"{Path(path).name}: sampling rate unknown — no time column (looked for "
            f"{'/'.join(_TIME_COLS)}). Silently assuming a rate would mis-time every segment."
        )
    # Estimate fs from the FIRST row group only (keeps open header-only / <2 s); reading the
    # whole time column on a multi-million-row Parquet would block the GUI thread.
    rg0 = pf.read_row_group(0, columns=[tcol]) if pf.metadata.num_row_groups else pf.read([tcol])
    t = rg0.column(0).to_numpy(zero_copy_only=False)
    if len(t) < 2:
        raise ValueError(f"{Path(path).name}: sampling rate unknown — {tcol!r} has < 2 rows")
    fs = _fs_from_time_column(t, tcol, path)
    # No channel_index: Parquet column names are unique by construction, so reads stay by name.
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
# Sampling-rate keys read off an array's own attrs first, then the group's. ``fs_hz``/``unit`` are
# what this app's own writer (``io.store.RecordingStore.create``) stamps on each channel array;
# ``fs``/``sfreq`` cover foreign stores.
_ZARR_FS_KEYS = ("fs_hz", "fs", "sfreq")


def _zarr_fs(name: str, arr_attrs: dict, group_attrs: dict, path) -> float:
    """Resolve ONE channel's sampling rate from its own attrs, else the group's. Never defaults.

    This was ``float(attrs.get("fs", attrs.get("sfreq", 250.0)) or 250.0)`` read off the GROUP only,
    so a store carrying no rate anywhere -- which is every Zarr store this repo actually writes --
    was graded at 250 Hz. That is 31.25x wrong for 8 Hz EDA, and being one rate for the whole store
    it cannot be right at all for a mixed-modality one. Same reasoning as ``open_parquet``: an
    unknown rate is raised, not assumed, because a wrong rate mis-times every segment silently.
    """
    for src in (arr_attrs, group_attrs):
        for key in _ZARR_FS_KEYS:
            if key not in src:
                continue
            try:
                fs = float(src[key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{Path(path).name}: channel {name!r} has a non-numeric {key!r} attribute "
                    f"({src[key]!r}). Write the sampling rate in Hz."
                ) from None
            if not np.isfinite(fs) or not (_FS_MIN_HZ <= fs <= _FS_MAX_HZ):
                raise ValueError(
                    f"{Path(path).name}: channel {name!r} declares {key}={fs:g}, outside the "
                    f"plausible {_FS_MIN_HZ}-{_FS_MAX_HZ:g} Hz band -- that is a unit error, not a rate."
                )
            return fs
    raise ValueError(
        f"{Path(path).name}: sampling rate unknown -- channel {name!r} carries no "
        f"{'/'.join(_ZARR_FS_KEYS)} attribute and neither does the group. Silently assuming a rate "
        f"would mis-time every segment (the old 250.0 default is 31x wrong for 8 Hz EDA)."
    )


def open_zarr(path: str | Path) -> RecordingHandle:
    """Open a Zarr store whose TOP-LEVEL arrays are 1-D per-channel sample vectors.

    The two structural guards below exist because this opener accepted the research corpus
    ``data/store_v8/waveforms.zarr`` without a murmur: it reported the four ``[N,1,L]`` modality
    arrays as four channels of one recording, all at the 250 Hz default, with ``n_samples`` set to
    the WINDOW COUNT -- so a 181,580 s ECG corpus opened as a 72.632 s recording (2500x understated)
    and ``read_window`` handed back a transposed 4-D array. Nothing raised anywhere, so the failure
    mode was confident garbage rather than a refusal.
    """
    import zarr

    g = zarr.open_group(str(path), mode="r")
    group_attrs = dict(g.attrs)
    names = [k for k, v in g.arrays()] if hasattr(g, "arrays") else list(g.array_keys())
    if not names:
        raise ValueError(
            f"{Path(path).name}: no top-level arrays. This opener reads a flat <store>/<channel> "
            f"layout of 1-D sample vectors; a nested layout (raw/<channel>, as written by "
            f"io.store.RecordingStore) is not read yet. It used to open as a zero-channel recording."
        )
    fs_hz: dict[str, float] = {}
    n_samples: dict[str, int] = {}
    units: dict[str, str] = {}
    for name in names:
        arr = g[name]
        if arr.ndim != 1:
            raise ValueError(
                f"{Path(path).name}: array {name!r} is {arr.ndim}-D {tuple(arr.shape)}, not a 1-D "
                f"vector of samples. An [n_windows, n_channels, n_samples] array is a research "
                f"SegmentStore (src/biosqa/data/store.py::SegmentStore), not a recording -- opening "
                f"it here would grade a window COUNT as a sample count. Read it with SegmentStore."
            )
        arr_attrs = dict(arr.attrs)
        fs_hz[name] = _zarr_fs(name, arr_attrs, group_attrs, path)
        n_samples[name] = int(arr.shape[0])
        units[name] = str(arr_attrs.get("unit", group_attrs.get("unit", "")))
    return RecordingHandle(
        path=Path(path), format="zarr", channel_names=names,
        fs_hz=fs_hz, n_samples=n_samples,
        backend=g, units=units,
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
