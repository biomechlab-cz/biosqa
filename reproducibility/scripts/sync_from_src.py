"""Regenerate this folder from the monorepo — ``biosqa/`` **and** ``scripts/``.

The reproducibility tree is a *generated* subset of the research package, not a
hand-maintained fork. Two trees are covered:

``biosqa/``
    every module is a **verbatim byte copy** of ``src/biosqa/<same path>``.
    One documented exception: ``biosqa/utils/paths.py``, whose ``REPO_ROOT``
    must resolve to *this* folder (the package sits at ``<root>/biosqa``, one
    level shallower than ``src/biosqa``).

``scripts/``
    every script is a copy of ``<monorepo>/scripts/<same name>`` with exactly
    one mechanical transform: the ``sys.path`` bootstrap is repointed from
    ``<root>/src`` to ``<root>`` (same layout reason), and a GENERATED banner is
    prepended. ``sync_from_src.py`` itself is reproducibility-only.

Hand-copying is what let this package rot twice: first ``eval/protocols.py``
fell behind and lost the cluster-bootstrap CI helpers, then nine modules plus
both export scripts shipped pre-fix code (the 2.08x-inflated artifact-type
``pos_weight`` and the un-masked ``class_weights(ytr)``) after the engine had
already been fixed. Nothing under ``biosqa/`` or ``scripts/`` may be edited
here; edit the monorepo copy and re-run this script.

Only runnable from the monorepo checkout — the public app repo has no ``src/``.

    python scripts/sync_from_src.py --check     # report drift, exit 1 if any
    python scripts/sync_from_src.py             # copy monorepo -> here
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PKG = HERE / "biosqa"
SCRIPTS = HERE / "scripts"
# app/reproducibility -> app -> <monorepo root>
MONOREPO = HERE.parents[1]
SRC = MONOREPO / "src" / "biosqa"
SRC_SCRIPTS = MONOREPO / "scripts"

# Package modules that must differ (layout-dependent), relative to the package root.
PKG_EXCEPTIONS = {Path("utils/paths.py")}
# Scripts that are reproducibility-only (no monorepo counterpart).
SCRIPT_EXCEPTIONS = {Path("sync_from_src.py")}

# The single mechanical transform applied to every synced script.
SRC_PATH_EXPR = 'str(Path(__file__).resolve().parents[1] / "src")'
DST_PATH_EXPR = "str(Path(__file__).resolve().parents[1])"

BANNER = (
    "# ---------------------------------------------------------------------------\n"
    "# GENERATED FILE — do not edit here.\n"
    "# Verbatim copy of <monorepo>/scripts/<this name>, with one transform: the\n"
    "# sys.path bootstrap points at <root> (this package's layout puts biosqa/ at\n"
    "# the root) instead of <root>/src. Regenerate: python scripts/sync_from_src.py\n"
    "# ---------------------------------------------------------------------------\n"
)


class SyncError(RuntimeError):
    """The transform could not be applied — the script would import the wrong tree."""


def transform_script(text: str, rel: Path) -> str:
    """Monorepo script text -> reproducibility script text."""
    if SRC_PATH_EXPR not in text:
        raise SyncError(
            f"{rel}: expected sys.path bootstrap {SRC_PATH_EXPR!r} not found in the "
            "monorepo copy — the transform is stale, fix sync_from_src.py before syncing"
        )
    return BANNER + text.replace(SRC_PATH_EXPR, DST_PATH_EXPR)


def _read_text(path: Path) -> str:
    # universal newlines: CRLF/LF differences never count as drift
    with path.open(encoding="utf-8", newline=None) as fh:
        return fh.read()


def _write_text(path: Path, text: str) -> None:
    # preserve the destination's existing newline convention to avoid whole-file churn
    nl = "\r\n" if (path.is_file() and b"\r\n" in path.read_bytes()) else "\n"
    with path.open("w", encoding="utf-8", newline=nl) as fh:
        fh.write(text)


def audit(apply: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Compare both trees against the monorepo.

    Returns ``(drift, missing, errors)`` as display strings. With ``apply=True``
    the drifted files are rewritten from the monorepo copy.
    """
    drift: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    for dst in sorted(PKG.rglob("*.py")):
        rel = dst.relative_to(PKG)
        if rel in PKG_EXCEPTIONS:
            continue
        src = SRC / rel
        if not src.is_file():
            missing.append(f"biosqa/{rel.as_posix()}")
            continue
        if not filecmp.cmp(src, dst, shallow=False):
            drift.append(f"biosqa/{rel.as_posix()}")
            if apply:
                shutil.copyfile(src, dst)

    for dst in sorted(SCRIPTS.glob("*.py")):
        rel = dst.relative_to(SCRIPTS)
        if rel in SCRIPT_EXCEPTIONS:
            continue
        src = SRC_SCRIPTS / rel
        if not src.is_file():
            missing.append(f"scripts/{rel.as_posix()}")
            continue
        try:
            want = transform_script(_read_text(src), rel)
        except SyncError as exc:
            errors.append(str(exc))
            continue
        if _read_text(dst) != want:
            drift.append(f"scripts/{rel.as_posix()}")
            if apply:
                _write_text(dst, want)

    return drift, missing, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift, do not copy")
    args = ap.parse_args()

    if not SRC.is_dir() or not SRC_SCRIPTS.is_dir():
        print(f"[sync] monorepo not found at {MONOREPO} — run this from the monorepo checkout")
        return 2

    drift, missing, errors = audit(apply=not args.check)

    for rel in missing:
        print(f"[sync] reproducibility-only (no monorepo counterpart): {rel}")
    for msg in errors:
        print(f"[sync] ERROR: {msg}")
    for rel in drift:
        print(f"[sync] {'DRIFT' if args.check else 'updated'}: {rel}")
    if not drift and not errors:
        print("[sync] in sync with the monorepo")
    if errors:
        return 2
    return 1 if (args.check and drift) else 0


if __name__ == "__main__":
    sys.exit(main())
