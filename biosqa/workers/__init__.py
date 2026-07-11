"""Background workers (Plan 2 §9): QThreadPool/QRunnable + result signals.

The #1 correctness risk in this app (Plan 2 §14): **no code in this package
may import Qt Quick scene-graph classes or call ``.update()`` on a
``QQuickItem``.** Workers only compute numpy arrays and emit them via
``workers.signals``; only ``viewmodels``/``scenegraph`` code running back on
the GUI thread (via ``Qt.QueuedConnection``) is allowed to touch the plot
item.
"""
