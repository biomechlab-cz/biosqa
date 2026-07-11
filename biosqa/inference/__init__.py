"""ONNX inference: runner, model-card-driven preprocessing, and segmentation.

Framework-agnostic (no PySide6 imports); ``workers.qt_threads`` dispatches
calls into this package from a ``QRunnable`` and marshals results back to
the GUI thread via ``workers.signals``.
"""
