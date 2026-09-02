"""The published reproducibility package must reproduce the FIXED engine.

`app/reproducibility/` is a generated snapshot of the research monorepo (`src/biosqa/`
plus `scripts/`). It has silently rotted twice: first `eval/protocols.py` fell behind
and lost the cluster-bootstrap CI helpers, then nine modules and both export scripts
shipped pre-fix code -- so a reviewer running the Data-availability package reproduced
bugs the engine had already fixed.

Two layers of defence here:

* :func:`test_snapshot_is_in_sync_with_the_monorepo` runs the sync script's own audit
  and fails on ANY content drift. It is skipped in the public app checkout, which has
  no `src/` -- it is a monorepo-side guard.
* the remaining tests are content assertions that hold in the public checkout too:
  the shipped harness still exposes the CI helpers, and the two export defects that
  were fixed upstream are not present in the shipped export scripts. Those pin the
  behaviour rather than the file bytes, so they also catch a snapshot regenerated
  from a stale source.

Nothing here imports the snapshot's `biosqa` -- `tests/conftest.py` puts the APP's
`biosqa` on `sys.path`, and importing the research one would shadow it (and pull in
torch). Everything is done by reading and parsing the files.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
REPRO = APP_ROOT / "reproducibility"
MONOREPO_SRC = APP_ROOT.parent / "src" / "biosqa"


def _load_sync_module():
    path = REPRO / "scripts" / "sync_from_src.py"
    spec = importlib.util.spec_from_file_location("_repro_sync_from_src", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toplevel_defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_reproducibility_package_is_present():
    assert (REPRO / "biosqa" / "eval" / "protocols.py").is_file()
    assert (REPRO / "scripts" / "sync_from_src.py").is_file()


@pytest.mark.skipif(
    not MONOREPO_SRC.is_dir(),
    reason="public app checkout has no src/biosqa -- monorepo-side guard",
)
def test_snapshot_is_in_sync_with_the_monorepo():
    """Fails the moment app/reproducibility/ drifts from src/biosqa or scripts/.

    Fix with: ``cd app/reproducibility && python scripts/sync_from_src.py``.
    This compares CONTENT, not just symbol names -- the previous guard only checked
    ``__all__`` and passed against nine files of stale code.
    """
    sync = _load_sync_module()
    drift, _missing, errors = sync.audit(apply=False)
    assert not errors, f"sync transform is broken: {errors}"
    assert not drift, (
        "reproducibility snapshot has drifted from the monorepo: "
        + ", ".join(drift)
        + " -- regenerate with `python app/reproducibility/scripts/sync_from_src.py`"
    )


def _fake_trees(tmp_path, monkeypatch, sync):
    """Point the sync module at a throwaway src/dst pair (never touches the real tree)."""
    src_pkg = tmp_path / "src" / "biosqa" / "eval"
    dst_pkg = tmp_path / "repro" / "biosqa" / "eval"
    src_scripts = tmp_path / "src_scripts"
    dst_scripts = tmp_path / "repro" / "scripts"
    for d in (src_pkg, dst_pkg, src_scripts, dst_scripts):
        d.mkdir(parents=True)
    monkeypatch.setattr(sync, "PKG", tmp_path / "repro" / "biosqa")
    monkeypatch.setattr(sync, "SRC", tmp_path / "src" / "biosqa")
    monkeypatch.setattr(sync, "SCRIPTS", dst_scripts)
    monkeypatch.setattr(sync, "SRC_SCRIPTS", src_scripts)
    return src_pkg, dst_pkg, src_scripts, dst_scripts


def test_sync_audit_detects_content_drift(tmp_path, monkeypatch):
    """The drift detector must react to CONTENT, not only to missing symbols.

    The guard this replaces compared ``__all__`` names only, so it passed against
    nine files of stale code.
    """
    sync = _load_sync_module()
    src_pkg, dst_pkg, _, _ = _fake_trees(tmp_path, monkeypatch, sync)
    body = "def cluster_bootstrap_ci(x):\n    return x\n"
    (src_pkg / "protocols.py").write_text(body, encoding="utf-8")
    (dst_pkg / "protocols.py").write_text(body, encoding="utf-8")
    assert sync.audit(apply=False) == ([], [], [])

    # same public symbol, different body -> must still be flagged
    (src_pkg / "protocols.py").write_text(body.replace("return x", "return x + 1"), encoding="utf-8")
    drift, _missing, errors = sync.audit(apply=False)
    assert drift == ["biosqa/eval/protocols.py"] and not errors

    # ...and the copy repairs it
    assert sync.audit(apply=True)[0] == ["biosqa/eval/protocols.py"]
    assert sync.audit(apply=False) == ([], [], [])


def test_sync_applies_exactly_one_transform_to_scripts(tmp_path, monkeypatch):
    """Scripts are copied with the sys.path bootstrap repointed and a banner added."""
    sync = _load_sync_module()
    _, _, src_scripts, dst_scripts = _fake_trees(tmp_path, monkeypatch, sync)
    original = 'import sys\nsys.path.insert(0, %s)\nX = 1\n' % sync.SRC_PATH_EXPR
    (src_scripts / "export_thing.py").write_text(original, encoding="utf-8")
    (dst_scripts / "export_thing.py").write_text("stale\n", encoding="utf-8")

    assert sync.audit(apply=True)[0] == ["scripts/export_thing.py"]
    got = (dst_scripts / "export_thing.py").read_text(encoding="utf-8")
    assert sync.SRC_PATH_EXPR not in got, "snapshot script would import <root>/src (absent here)"
    assert sync.DST_PATH_EXPR in got
    assert "GENERATED FILE" in got
    assert "X = 1" in got, "the transform must not change anything else"
    assert sync.audit(apply=False) == ([], [], [])


def test_sync_errors_instead_of_writing_an_unbootstrappable_script(tmp_path, monkeypatch):
    """If the monorepo script stops matching the transform, fail loudly.

    Copying it verbatim would leave a ``sys.path`` entry pointing at ``<root>/src``,
    which does not exist in the published package -- an unrunnable snapshot.
    """
    sync = _load_sync_module()
    _, _, src_scripts, dst_scripts = _fake_trees(tmp_path, monkeypatch, sync)
    (src_scripts / "export_thing.py").write_text("import sys\nX = 1\n", encoding="utf-8")
    (dst_scripts / "export_thing.py").write_text("stale\n", encoding="utf-8")

    drift, _missing, errors = sync.audit(apply=True)
    assert errors and "export_thing.py" in errors[0]
    assert drift == []
    assert (dst_scripts / "export_thing.py").read_text(encoding="utf-8") == "stale\n"


def test_shipped_eval_harness_still_defines_the_ci_helpers():
    """The two functions that produce the paper's confidence intervals."""
    defs = _toplevel_defs(REPRO / "biosqa" / "eval" / "protocols.py")
    assert {"cluster_bootstrap_ci", "dump_raw_points"} <= defs


def test_shipped_export_scripts_do_not_carry_the_fixed_defects():
    """Regression guard for the two export defects fixed upstream.

    * artifact-type ``pos_weight`` must count negatives over the TYPE-LABELLED rows
      (``mask.sum() - pos``), not over every row (``len(Xall) - ...``): the loss is
      masked to those rows, so the whole-set count inflated every entry ~2.1x.
    * grade ``class_weights`` must be computed on the grade-supervised rows
      (``ytr[gm_tr]``), not after the grade-masked synthetic windows are appended.
    """
    ecg = (REPRO / "scripts" / "export_ecg_dualbranch.py").read_text(encoding="utf-8")
    assert "len(Xall) - Tall.sum(0)" not in ecg, "inflated artifact-type pos_weight is back"
    assert "def type_pos_weight(" in ecg
    assert "type_pos_weight(Tall, tmask)" in ecg

    allmod = (REPRO / "scripts" / "export_all_modalities.py").read_text(encoding="utf-8")
    assert "class_weights=class_weights(ytr)" not in allmod, "un-masked class_weights is back"
    assert "class_weights=class_weights(ytr[gm_tr])" in allmod
    # calibration ECE must be reported on a partition it was not fitted on
    assert "fit_partition" in allmod and "evaluation_partition" in allmod


def _unresolvable_biosqa_imports(path: Path, pkg_root: Path) -> set[str]:
    """Module-level ``biosqa.*`` imports in ``path`` that this snapshot cannot satisfy."""

    def have(mod: str) -> bool:
        p = pkg_root.joinpath(*mod.split(".")[1:])
        return p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()

    try:
        rel_parts = path.relative_to(pkg_root).with_suffix("").parts[:-1]
        base = ".".join(("biosqa",) + rel_parts)
    except ValueError:  # a script, not a package module -> no relative imports
        base = ""

    missing: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = base.split(".")
                anchor = ".".join(parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts)
                target = f"{anchor}.{node.module}" if node.module else anchor
            elif node.module and node.module.startswith("biosqa"):
                target = node.module
            else:
                continue
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("biosqa") and not have(alias.name):
                    missing.add(alias.name)
            continue
        else:
            continue
        if target.startswith("biosqa") and not have(target):
            missing.add(target)
    return missing


def test_readme_is_honest_about_what_the_snapshot_cannot_import():
    """The package deliberately omits the data-acquisition layer, so some shipped
    modules cannot be imported. That list is documented in the README and pinned
    here: if it GROWS the docs are silently wrong, if it SHRINKS they are stale.
    """
    pkg = REPRO / "biosqa"
    broken = {
        p.relative_to(pkg).as_posix(): sorted(_unresolvable_biosqa_imports(p, pkg))
        for p in sorted(pkg.rglob("*.py"))
        if _unresolvable_biosqa_imports(p, pkg)
    }
    assert broken == {
        "data/artifact_labels.py": ["biosqa.data.harmonize"],
        "data/artifact_synth.py": ["biosqa.data.datasets", "biosqa.data.harmonize"],
        "data/windows.py": ["biosqa.data.harmonize"],
    }, f"documented import gaps changed: {broken}"

    readme = (REPRO / "README.md").read_text(encoding="utf-8")
    for name in ("data/windows.py", "data/artifact_labels.py", "data/artifact_synth.py"):
        assert name.replace("data/", "data/") in readme

    # the export scripts are documented as reference-only for the same reason
    scripts_gap = sorted(
        _unresolvable_biosqa_imports(REPRO / "scripts" / "export_all_modalities.py", pkg)
    )
    assert scripts_gap == ["biosqa.data.harmonize", "biosqa.data.store", "biosqa.xdomain"]
    assert "not runnable as shipped" in readme


def test_snapshot_modules_the_docs_promise_do_import(tmp_path):
    """The pieces the README says are usable must actually import off a clean path.

    Run in a subprocess with only the snapshot prepended, so the monorepo's editable
    `biosqa` install cannot satisfy the import and mask a missing module.
    """
    import subprocess

    for dep in ("numpy", "scipy", "sklearn", "torch"):
        if importlib.util.find_spec(dep) is None:
            pytest.skip(f"research stack not installed here ({dep} missing) — app venv is torch-free")

    code = (
        "import sys, pathlib\n"
        f"here = pathlib.Path(r'{REPRO}')\n"
        "sys.path.insert(0, str(here))\n"
        "import biosqa.eval.protocols, biosqa.eval.metrics, biosqa.data.synthetic\n"
        "import biosqa.data.sqa_features, biosqa.data.sqi, biosqa.data.nonlinear_features\n"
        "bad = [m.__file__ for n, m in sys.modules.items()\n"
        "       if n.startswith('biosqa') and getattr(m, '__file__', None)\n"
        "       and not str(pathlib.Path(m.__file__).resolve()).startswith(str(here))]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_generated_trees_say_they_are_generated():
    """A reader who opens the snapshot must be told not to hand-edit it."""
    assert "GENERATED" in (REPRO / "biosqa" / "GENERATED.md").read_text(encoding="utf-8")
    assert "GENERATED" in (REPRO / "README.md").read_text(encoding="utf-8")
    banner = (REPRO / "scripts" / "export_all_modalities.py").read_text(encoding="utf-8")
    assert "GENERATED FILE" in banner.splitlines()[1]
