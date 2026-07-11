# -----------------------------------------------------------------------------
# PyInstaller spec — BioSQA Studio (dev/CI build, Plan 2 §13)
#
# PLACEHOLDER: this spec is a structurally-correct starting point, not a
# verified build. TODO(Plan2 §13/Phase 5): validate PySide6/QML hooks, test
# the frozen app's relative-path resolution for `ui/` (QML data files) and
# `models/` (ONNX artifacts) on a clean machine, and switch to --onedir
# (one-folder) for faster startup as recommended in the plan.
#
# Build with:
#   pyinstaller build/biosqa.spec --distpath build/dist --workpath build/work
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

APP_ROOT = Path(SPECPATH).resolve().parent  # app/  (SPECPATH = app/build, a PyInstaller spec global)
PACKAGE = APP_ROOT / "biosqa"

# QML files are plain data as far as PyInstaller is concerned -- PySide6
# does not import them as Python modules, so they must be listed explicitly.
# TODO(Plan2 §13): once biosqa/ui/qml.qrc is compiled with
# pyside6-rcc for release builds, this can shrink to a single generated
# qml_rc.py module instead of shipping loose .qml files.
qml_datas = [
    (str(f), str(f.parent.relative_to(APP_ROOT)))
    for f in (PACKAGE / "ui").rglob("*")
    # Everything under ui/ EXCEPT Python: *.qml AND `qmldir` (no extension — CRITICAL: the
    # Theme `pragma Singleton` needs it or the whole QML tree fails to load), fonts, icon, qrc.
    if f.is_file() and f.suffix not in {".py", ".pyc"} and "__pycache__" not in f.parts
]

# ONNX artifacts + model cards (Plan 1 handshake, see app/models/README.md).
# Empty on a clean checkout -- populated at release-build time.
model_datas = [
    (str(f), "models")
    for f in (APP_ROOT / "models").glob("*")
    if f.is_file() and f.name not in {"README.md", ".gitkeep"}
]

a = Analysis(
    [str(PACKAGE / "main.py")],
    pathex=[str(APP_ROOT)],
    binaries=[],
    datas=qml_datas + model_datas + collect_data_files("PySide6", subdir="qml"),
    hiddenimports=[
        "biosqa.viewmodels",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "mlflow"],  # keep Plan 1 deps out, see §5
    noarchive=False,
)

pyz = PYZ(a.pure)  # PyInstaller 6.x removed cipher / a.zipped_data

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BioSQA Studio",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=(str(APP_ROOT / "build" / "icon.ico") if sys.platform == "win32" else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="BioSQA Studio",
)
