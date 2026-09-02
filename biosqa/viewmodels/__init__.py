"""Python QObject/QAbstractItemModel backends exposed to QML (design spec (b)/(d)).

Each module here is registered once as a singleton context property in
``biosqa.main.build_engine`` (``AppController``, ``recordings``,
``channels``, ``signalView``, ``segments``, ``selection``, ``inference``,
``modelCard``, ``exporter``) -- QML binds to the instance directly rather
than constructing one per use, per the design spec's registration pattern.

The one exception is ``biosqa.scenegraph.decimated_series_item``,
which *is* registered as an actual QML type (``qmlRegisterType``-equivalent)
because it is instantiated once per visible channel lane by a ``Repeater``.
"""
