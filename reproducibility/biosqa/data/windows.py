"""Physiological-time windowing + the shared loader contract (Plan 1 §7.3, §6.3).

Loaders (``biosqa.data.datasets.<name>.load``) return a list of
:class:`RecordSample` — raw signal + native quality labels (per-record or as
time spans) in a common structure. :func:`windowize` then turns any record into
fixed **time-normalized** windows with harmonized Q0..Q3 labels, so every dataset
feeds the same segment index regardless of sampling rate or label granularity.

Windows are normalized by *time*, not sample count (ECG 10 s, PPG 30 s, EEG 1-5 s,
EDA 30-60 s), and :func:`patch_params` derives per-modality ``patch_len``/``stride``
so token counts are comparable across modalities (~32-64 tokens/window).
"""
from __future__ import annotations

import hashlib

from dataclasses import dataclass, field

import numpy as np

from .harmonize import (
    ARTIFACT_TYPE_INDEX,
    Q0,
    Q3,
    normalize_artifact_type,
    to_harmonized,
    window_fraction_to_q,
)

__all__ = ["RecordSample", "windowize", "patch_params"]


@dataclass
class RecordSample:
    """One loaded recording (or record) in the common loader contract.

    Exactly one of ``label_native`` (whole-record) or ``label_spans`` (interval
    labels, sample-indexed) should be set. ``dataset_key`` selects the
    harmonization table in :mod:`biosqa.data.harmonize`.

    Artifact-TYPE (level-3) supervision is optional and orthogonal to the Q
    grade: set ``artifact_type`` for a whole-record type (e.g. macecgdb
    ``"walking"``, NSTDB ``"em"``), or emit 4-tuple ``label_spans`` elements
    ``(start, end, native_q, type_token)`` for interval types (TUAR, PTB-XL).
    Tokens are normalized to :data:`harmonize.ARTIFACT_TYPES` at windowing time;
    combination tokens (``"eyem_musc"``) are split on ``_`` (multi-label).
    """

    dataset: str                       # display name, e.g. "BUT QDB"
    dataset_key: str                   # harmonization key, e.g. "but_qdb"
    modality: str                      # ecg | ppg | eeg | eda
    subject_id: str
    fs: float
    signal: np.ndarray                 # [C, L] float32
    sig_names: list[str] | None = None  # per-channel names (e.g. ["MLII", "V1"]); enables
                                        # store.select_channel to pick an ECG lead BY NAME
    label_native: object | None = None
    label_spans: list[tuple] | None = None  # (start, end, native) or (start, end, native, type_token)
    artifact_type: object | None = None      # whole-record artifact-type token (level-3), or None
    # C8 (Plan 09). A subject can contribute several RECORDINGS/sessions, and until
    # 2026-08-18 that identity was dropped at windowing: store_v8 has 2,776
    # (dataset, subject, t_start) keys colliding over 14,400 rows, so same-recording
    # leakage was unprovable after construction. Loaders that know their record/session
    # should set these; both stay optional so no loader is forced to invent one.
    recording_id: str | None = None          # stable per source record/session
    source_path: str | None = None           # source-relative path or opaque raw ID
    device_id: str | None = None             # acquisition device, when the source states it
    meta: dict = field(default_factory=dict)


def _canonical_types(token) -> set[str]:
    """Native/combination artifact token -> set of canonical ARTIFACT_TYPES.

    Accepts already-canonical tokens (``electrode_leadoff``), native tokens
    (``bckg``, ``bw``), and combination tokens (``eyem_musc``, split on ``_``).
    Returns an empty set for tokens that carry no type info.
    """
    if token is None:
        return set()
    tok = str(token).strip()
    if tok in ARTIFACT_TYPE_INDEX:            # already a canonical class
        return {tok}
    whole = normalize_artifact_type(tok)      # try the whole token first
    if whole is not None:
        return {whole}
    out: set[str] = set()                     # else a combination token: split on '_'
    for part in tok.split("_"):
        p = normalize_artifact_type(part)
        if p is not None:
            out.add(p)
    return out


def patch_params(fs: float, window_s: float, target_tokens: int = 48, overlap: float = 0.5):
    """Return ``(patch_len, stride, window_len, n_tokens)`` for a modality so a
    window yields ~``target_tokens`` patch tokens."""
    window_len = int(round(window_s * fs))
    stride = max(1, window_len // target_tokens)
    patch_len = max(stride, int(round(stride / (1 - overlap)))) if overlap else stride
    n_tokens = (window_len - patch_len) // stride + 1
    return patch_len, stride, window_len, n_tokens


def _native_label_vector(rec: RecordSample, length: int) -> np.ndarray | None:
    """Per-sample native quality-label array from spans, or None if per-record."""
    if rec.label_spans is None:
        return None
    v = np.full(length, fill_value=None, dtype=object)
    for span in rec.label_spans:
        start, end, lab = span[0], span[1], span[2]
        s, e = max(0, int(start)), min(length, int(end))
        if e > s:
            v[s:e] = lab
    return v


def _type_label_vector(rec: RecordSample, length: int) -> np.ndarray | None:
    """Per-sample native artifact-TYPE token array from 4-tuple spans, or None
    if no span carries a type token (level-3 supervision absent)."""
    if rec.label_spans is None or not any(len(s) >= 4 for s in rec.label_spans):
        return None
    v = np.full(length, fill_value=None, dtype=object)
    for span in rec.label_spans:
        if len(span) < 4:
            continue
        start, end, _lab, typ = span[0], span[1], span[2], span[3]
        s, e = max(0, int(start)), min(length, int(end))
        if e > s:
            v[s:e] = typ
    return v


def windowize(
    rec: RecordSample,
    window_s: float,
    stride_s: float | None = None,
    *,
    agg: str = "worst",        # "worst" (min Q dominates) | "majority" | "fraction" | "burden"
    min_coverage: float = 0.5,  # min fraction of a window that must carry a label
    drop_unlabeled: bool = True,
) -> list[dict]:
    """Slice ``rec`` into time windows, assign a harmonized Q label to each, and
    (when present) an artifact-TYPE token.

    ``agg='worst'`` takes the minimum harmonized Q over the window (a window is as
    bad as its worst part — the conservative, clinically-safe choice). ``'majority'``
    takes the most-common harmonized label. ``'fraction'`` (for per-sample binary
    artifact masks like EDABE) computes the fraction of non-clean (non-Q3) samples
    and buckets it via :func:`harmonize.window_fraction_to_q`, so the full Q0..Q3
    scale is recovered from a binary source. ``'burden'`` is the ordinal
    generalization of ``'fraction'``: it averages the per-sample artifact *burden*
    ``(Q3 - q) / Q3``, buckets that, and then clamps the result to be **no worse
    than the window's own worst sample**, so a graded source (eda_artifact's
    3-expert vote) lands on the same window scale as a binary one without
    collapsing its vote and without ever grading a window below any part of it.
    Semantics on a uniform 60 s eda_artifact window (native = #experts voting
    artifact): 0/3 -> Q3, 1/3 -> Q2, 2/3 -> Q1, 3/3 -> Q0, i.e. a uniform window
    keeps its per-5 s grade exactly; partial contamination moves it down by
    extent. On a purely binary Q0/Q3 source the clamp never binds and
    ``'burden'`` is identical to ``'fraction'``.
    Windows whose labels can't be determined (all excluded/None) are dropped when
    ``drop_unlabeled``.

    Each emitted row also carries ``label_native`` (the record's native label,
    for per-record datasets) and ``artifact_type`` (a ``|``-joined set of
    canonical :data:`harmonize.ARTIFACT_TYPES` present in the window, ``"clean"``
    when the record has type supervision but this window has none, or ``None``
    when the dataset carries no artifact-type labels at all).
    """
    C, L = rec.signal.shape
    win = int(round(window_s * rec.fs))
    step = int(round((stride_s if stride_s is not None else window_s) * rec.fs))
    if win <= 0 or win > L:
        return []
    label_vec = _native_label_vector(rec, L)
    type_vec = _type_label_vector(rec, L)
    has_types = type_vec is not None or rec.artifact_type is not None
    record_type_tokens = _canonical_types(rec.artifact_type)  # whole-record type (may be empty)
    rows: list[dict] = []
    n_flatline_regraded = [0]   # C10, reported by build_store
    for start in range(0, L - win + 1, step):
        end = start + win
        # resolve harmonized label for this window
        if label_vec is None:  # per-record native label
            q = to_harmonized(rec.dataset_key, rec.label_native)
            qs = None if q is None else np.array([q])
        else:
            native = label_vec[start:end]
            mapped = [to_harmonized(rec.dataset_key, n) for n in native if n is not None]
            mapped = [m for m in mapped if m is not None]
            if len(mapped) < min_coverage * win:
                qs = None
            else:
                qs = np.array(mapped)
        if qs is None or qs.size == 0:
            if drop_unlabeled:
                continue
            q_final = None
        elif agg == "fraction":
            q_final = window_fraction_to_q(float(np.mean(qs != Q3)))
        elif agg == "burden":
            # Never grade a window BELOW its own worst constituent sample: the
            # burden bucket measures *extent*, and clamping keeps it from also
            # re-penalising *severity* that the per-sample Q already encodes.
            # (Without the clamp, a window every sample of which carries a
            # unanimous-minus-one vote -> all-Q1 has mean burden 2/3 > 0.50 and
            # lands on Q0, i.e. worse than any sample in it.) The clamp is a
            # no-op on a binary Q0/Q3 source, so 'burden' still coincides
            # exactly with 'fraction' there.
            q_final = max(window_fraction_to_q(float(np.mean((Q3 - qs) / Q3))),
                          int(qs.min()))
        else:
            q_final = int(qs.min()) if agg == "worst" else int(np.bincount(qs).argmax())
        if q_final is None:
            continue
        # resolve artifact-type token(s) for this window
        artifact_type = None
        if has_types:
            types: set[str] = set(record_type_tokens)
            if type_vec is not None:
                for tok in type_vec[start:end]:
                    types |= _canonical_types(tok)
            types.discard("clean")
            artifact_type = "|".join(sorted(types)) if types else "clean"
        win_sig = rec.signal[:, start:end].astype(np.float32)
        # C10 (Plan 09, 2026-08-18). A zero-variance window is a detached electrode,
        # a saturated amplifier or a dropout -- never an excellent recording. store_v8
        # carried 17 constant windows of which 4 were graded USABLE (3 EDABE Q3 from one
        # subject, 1 TUAR Q3), teaching the model that a flat trace is pristine while
        # every other cohort teaches the opposite. The corpus already has a convention
        # for this -- MIMIC's own SQI maps -2 (flatline/repeated extremes) to Q0 -- so
        # applying it uniformly is consistent harmonization, not an invented label.
        if q_final is not None and int(q_final) > Q0 and float(np.ptp(win_sig)) == 0.0:
            q_final = Q0
            n_flatline_regraded[0] += 1
        rows.append(
            {
                "dataset": rec.dataset,
                "dataset_key": rec.dataset_key,
                "subject_id": rec.subject_id,
                # C7 (Plan 09). Hash of the exact stored bytes. store_v8 carried 4
                # byte-identical content groups over 73 rows, 67 of which spanned
                # train/val/test while every row had a DISTINCT subject_id -- so
                # subject-disjointness passed and identical waveforms still sat on
                # both sides of the partition. The modality is mixed in so two
                # different signals can never collide across modality arrays.
                "content_sha1": hashlib.sha1(
                    rec.modality.encode() + win_sig.tobytes()).hexdigest(),
                "recording_id": rec.recording_id or rec.subject_id,
                "source_path": rec.source_path,
                "device_id": rec.device_id,
                "modality": rec.modality,
                "fs_hz": rec.fs,
                "t_start_s": start / rec.fs,
                "t_end_s": end / rec.fs,
                "label_harmonized": q_final,
                # str-coerced: within one modality the native scheme differs per
                # dataset (int / str / None), and a mixed-type object column is
                # not parquet-safe. None stays null.
                "label_native": None if rec.label_native is None else str(rec.label_native),
                "artifact_type": artifact_type,
                "flatline_regraded": bool(
                    q_final == Q0 and float(np.ptp(win_sig)) == 0.0),
                "signal": win_sig,
            }
        )
    return rows
