# Nuitka production build, notes (Plan 2 §13, §14)

Nuitka compiles the app to C for a ~2-4x runtime speedup over the
PyInstaller dev build and a smaller reverse-engineering surface. Use it for
the *shipped* production build; use `build/biosqa.spec` (PyInstaller) for
fast dev/CI iteration.

## Prerequisites

- A working C toolchain (MSVC on Windows, `nuitka` will prompt to download
  a minimal one via `--assume-yes-for-downloads` if none is found).
- `pip install -e ".[packaging]"` inside the **app's own venv** (never the
  Plan 1 training venv, see `app/README.md`).

## Command

See `build/nuitka.cfg` for the full flag list. In short:

```powershell
nuitka --standalone --enable-plugin=pyside6 `
  --include-qt-plugins=platforms,imageformats `
  --include-data-dir=biosqa\ui=biosqa\ui `
  --include-data-dir=models=models `
  --output-dir=build\dist-nuitka `
  biosqa\main.py
```

## Known pitfalls (do not skip these when this becomes a real build)

1. **QML data files are not Python modules.** Nuitka's dependency analysis
   only follows imports; `biosqa/ui/**/*.qml` must be shipped via
   `--include-data-dir` (or compiled into `qml.qrc` first, see
   `biosqa/ui/resources.py`) or the frozen app will fail at
   `engine.load()` with a "file not found" style error and a blank window.
   This is the QML-specific analog of the `models/` path issue already
   flagged in Plan 2 §13.
2. **`models/` must resolve relative to the frozen executable**, not the
   development working directory. `biosqa/main.py` resolves paths
   via `Path(__file__).resolve().parent`, which survives freezing as long
   as the data dirs above are included next to the executable.
3. **Keep `--nofollow-import-to` for Plan 1's heavy/optional deps** (torch,
   transformers, mlflow, snorkel, ...) even though they should never be
   importable from `biosqa` in the first place, this is a
   belt-and-suspenders guard against accidental cross-imports blowing up
   build size.
4. **PySide6 plugin coverage**: `--enable-plugin=pyside6` handles most of
   it, but verify `platforms` (needed just to open a window) and
   `imageformats` (needed for exported PNG previews) are actually bundled;
   missing platform plugins is the #1 "works here, blank-screens there"
   failure mode for Qt freezes.
5. Nuitka builds are slow (minutes, not seconds), do not use it in the
   inner dev loop; that's what `python -m biosqa.main` and the
   PyInstaller spec are for.
