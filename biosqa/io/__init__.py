"""Out-of-core data layer: readers, Zarr store, decimation pyramid, LRU cache.

Framework-agnostic (no PySide6 imports) so it can be unit-tested without a
Qt event loop -- see ``tests/test_pyramid.py`` and ``tests/test_cache.py``.
"""
