"""Quality intervals -> CSV / TSV / JSON / Parquet / WFDB-annotation / MAT (Plan 2 §8.2/§11).

All writers are plain-data (no Qt) and testable here. They share one shape via
:func:`interval_records` and are registered in :data:`EXPORTERS` (``fmt -> (writer, ext)``) so
``ExportController.exportToPath`` is a single lookup and adding a format is one function.

Format notes:
  * **csv / parquet** — the flat interval table (the original schema, now incl. an ``artifacts`` column).
  * **tsv** — BIDS ``_events.tsv`` columns (``onset``/``duration``/``trial_type``) for BIDS pipelines.
  * **json** — the RICHEST carrier: nested, schema-versioned report incl. artifacts + model provenance;
    round-trips into Plan 1's active-learning reverse channel.
  * **wfdb** — a PhysioNet annotation file (``.qual`` via ``wfdb.wrann``): sample-indexed ``~`` (signal-
    quality-change) annotations, tier in ``subtype`` + full info in ``aux_note``. Needs the recording ``fs``.
  * **mat** — a MATLAB struct of the interval columns (``scipy.io.savemat``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from biosqa.inference.segmenter import QualityInterval

SCHEMA_VERSION = "1.0"

#: Flat CSV/Parquet column order (expected on import by Plan 1's active-learning ingestion,
#: Plan 2 §11 "reverse channel"). ``artifacts`` is ";"-joined in CSV, a list in Parquet/JSON.
#: ``tier`` is the EFFECTIVE grade (the reviewer's correction when relabeled, else the model's);
#: ``model_tier`` always preserves the model's original prediction so a correction is auditable.
INTERVAL_COLUMNS = ("start_sec", "end_sec", "tier", "model_tier", "confidence", "artifacts",
                    "overridden", "note", "recoverable", "recovered_tier",
                    "uncertainty", "rate_usable", "hr_bpm", "conformal_set")

#: Ordinal tier -> WFDB annotation subtype (a per-annotation int slot).
_TIER_SUBTYPE = {"Q0": 0, "Q1": 1, "Q2": 2, "Q3": 3}

#: Human labels for the QML export menu (kept next to the registry so they stay in sync).
FORMAT_LABELS = {
    "csv": "CSV table",
    "tsv": "TSV (BIDS events)",
    "json": "JSON report",
    "parquet": "Parquet",
    "wfdb": "WFDB annotation",
    "mat": "MATLAB .mat",
}


def interval_records(
    intervals: list[QualityInterval],
    overridden: dict[int, bool] | None = None,
    notes: dict[int, str] | None = None,
    corrected: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """One rich dict per interval (the shared source every writer formats from).

    ``overridden``/``notes``/``corrected`` are keyed by interval position and come from
    ``SelectionController`` (design spec (d)) -- kept separate so ``QualityInterval`` stays a
    plain model-prediction record. When ``corrected[i]`` is set (a reviewer relabeled that
    segment), ``tier`` becomes that corrected grade and ``model_tier`` keeps the model's original;
    otherwise both equal the model's prediction.
    """
    overridden = overridden or {}
    notes = notes or {}
    corrected = corrected or {}
    recs: list[dict[str, object]] = []
    for i, iv in enumerate(intervals):
        new_tier = corrected.get(i) or ""
        recs.append(
            {
                "start_sec": float(iv.start_sec),
                "end_sec": float(iv.end_sec),
                "duration_sec": float(iv.duration_sec),
                "tier": new_tier or iv.tier,        # effective grade (human correction wins)
                # model's ORIGINAL prediction: iv.model_tier survives an in-place reclassify (which
                # mutates iv.tier); "" → never reclassified, so fall back to the current tier.
                "model_tier": getattr(iv, "model_tier", "") or iv.tier,
                "confidence": float(iv.confidence),
                "artifacts": list(iv.artifacts),
                "overridden": bool(overridden.get(i, False)) or bool(new_tier),
                "note": notes.get(i, ""),
                "recoverable": bool(getattr(iv, "recoverable", False)),
                "recovered_tier": getattr(iv, "recovered_tier", ""),
                "uncertainty": round(float(getattr(iv, "uncertainty", 0.0)), 4),
                "rate_usable": bool(getattr(iv, "rate_usable", False)),
                "hr_bpm": round(float(getattr(iv, "hr_bpm", 0.0)), 1),
                "conformal_set": list(getattr(iv, "conformal_set", ())),
            }
        )
    return recs


def intervals_to_rows(intervals, overridden=None, notes=None, corrected=None) -> list[dict[str, object]]:
    """Flat rows in ``INTERVAL_COLUMNS`` order (artifacts ";"-joined), for CSV/Parquet."""
    return [
        {
            "start_sec": r["start_sec"], "end_sec": r["end_sec"], "tier": r["tier"],
            "model_tier": r["model_tier"], "confidence": r["confidence"],
            "artifacts": ";".join(r["artifacts"]), "overridden": r["overridden"], "note": r["note"],
            "recoverable": r["recoverable"], "recovered_tier": r["recovered_tier"],
            "uncertainty": r["uncertainty"], "rate_usable": r["rate_usable"], "hr_bpm": r["hr_bpm"],
            "conformal_set": ";".join(r["conformal_set"]),
        }
        for r in interval_records(intervals, overridden, notes, corrected)
    ]


def export_intervals_csv(intervals, path, *, overridden=None, notes=None, corrected=None, **_) -> Path:
    out = Path(path)
    rows = intervals_to_rows(intervals, overridden, notes, corrected)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INTERVAL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return out


def export_intervals_tsv(intervals, path, *, overridden=None, notes=None, corrected=None, **_) -> Path:
    """BIDS-style ``_events.tsv`` (onset / duration / trial_type + extras)."""
    out = Path(path)
    recs = interval_records(intervals, overridden, notes, corrected)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["onset", "duration", "trial_type", "model_tier", "confidence",
                    "artifacts", "overridden", "note"])
        for r in recs:
            w.writerow([r["start_sec"], r["duration_sec"], r["tier"], r["model_tier"], r["confidence"],
                        ";".join(r["artifacts"]), int(r["overridden"]), r["note"]])
    return out


def export_intervals_json(intervals, path, *, overridden=None, notes=None, corrected=None,
                          provenance=None, **_) -> Path:
    """Schema-versioned quality report: full per-interval records + model provenance."""
    out = Path(path)
    recs = interval_records(intervals, overridden, notes, corrected)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": "BioSQA Studio",
        "provenance": provenance or {},
        "n_intervals": len(recs),
        "total_duration_sec": recs[-1]["end_sec"] if recs else 0.0,
        "intervals": [
            {k: r[k] for k in ("start_sec", "end_sec", "tier", "model_tier", "confidence",
                               "artifacts", "overridden", "note", "recoverable", "recovered_tier",
                               "uncertainty", "rate_usable", "hr_bpm", "conformal_set")}
            for r in recs
        ],
    }
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def export_intervals_parquet(intervals, path, *, overridden=None, notes=None, corrected=None, **_) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(path)
    recs = interval_records(intervals, overridden, notes, corrected)
    # Build every INTERVAL_COLUMNS column from the shared records so the flat table can't silently
    # drift from CSV (artifacts stays a list<str> in Parquet, ";"-joined only in CSV/TSV).
    cols = {c: [(r["artifacts"] if c == "artifacts" else r[c]) for r in recs] for c in INTERVAL_COLUMNS}
    pq.write_table(pa.table(cols), out)
    return out


def export_intervals_wfdb(intervals, path, *, fs=None, inference_channel=0, corrected=None, **_) -> Path:
    """Write a PhysioNet WFDB annotation file (``<record>.qual``) via ``wfdb.wrann``.

    Each interval -> a ``~`` (change-in-signal-quality) annotation at its start sample; the ordinal
    tier goes in ``subtype`` (Q0..Q3 -> 0..3) and the full ``tier|conf|artifacts`` string in
    ``aux_note``. Needs the recording ``fs`` to map seconds -> sample indices.
    """
    if not fs or fs <= 0:
        raise ValueError("WFDB annotation export needs the recording's sample rate — open a recording first.")
    import numpy as np
    import wfdb

    out = Path(path)
    record = out.stem
    ext = out.suffix.lstrip(".") or "qual"
    recs = interval_records(intervals, corrected=corrected)  # annotate the effective (corrected) tier
    if not recs:
        raise ValueError("Nothing to export.")
    sample = np.asarray([int(round(r["start_sec"] * fs)) for r in recs], dtype=np.int64)
    symbol = ["~"] * len(recs)
    subtype = np.asarray([_TIER_SUBTYPE.get(r["tier"], 0) for r in recs], dtype=np.int64)
    chan = np.asarray([int(inference_channel)] * len(recs), dtype=np.int64)
    aux_note = [f'{r["tier"]}|conf={r["confidence"]:.2f}|{",".join(r["artifacts"])}' for r in recs]
    wfdb.wrann(record, ext, sample=sample, symbol=symbol, subtype=subtype, chan=chan,
               aux_note=aux_note, fs=float(fs), write_dir=str(out.parent))
    return out.parent / f"{record}.{ext}"


def export_intervals_mat(intervals, path, *, overridden=None, notes=None, corrected=None, **_) -> Path:
    import numpy as np
    from scipy.io import savemat

    out = Path(path)
    rows = intervals_to_rows(intervals, overridden, notes, corrected)   # all INTERVAL_COLUMNS, artifacts joined
    _num = {"start_sec", "end_sec", "confidence", "uncertainty", "hr_bpm"}
    _bool = {"overridden", "recoverable", "rate_usable"}
    data = {}
    for c in INTERVAL_COLUMNS:
        vals = [r[c] for r in rows]
        dtype = float if c in _num else (bool if c in _bool else object)
        data[c] = np.asarray(vals, dtype=dtype)
    savemat(str(out), {"quality_intervals": data})
    return out


#: fmt -> (writer, canonical extension). ExportController.exportToPath is one lookup on this.
EXPORTERS: dict[str, tuple] = {
    "csv": (export_intervals_csv, ".csv"),
    "tsv": (export_intervals_tsv, ".tsv"),
    "json": (export_intervals_json, ".json"),
    "parquet": (export_intervals_parquet, ".parquet"),
    "wfdb": (export_intervals_wfdb, ".qual"),
    "mat": (export_intervals_mat, ".mat"),
}


def export_figure(quick_window: object, path: str | Path, image_format: str = "png") -> Path:
    """Grab the current plot view as a PNG/SVG snapshot (TODO Plan2 §8.2, needs a live QQuickWindow)."""
    raise NotImplementedError(
        "export_figure: PNG/SVG figure export not yet implemented (TODO Plan2 §8.2, Phase 4)"
    )
