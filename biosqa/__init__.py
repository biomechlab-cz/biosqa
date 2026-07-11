"""BioSQA Studio — desktop application package.

PySide6 + QML front end for interactive inspection and ONNX-driven quality
segmentation of GB-scale biosignal recordings. See ``app/README.md`` and
``plans/02_DESKTOP_APP_ENGINEERING_PLAN.md`` for the full design.

This package is deliberately independent of the repo-root ``biosqa``
training package (Plan 1): it consumes Plan 1's exported ``models/*.onnx``
+ ``models/*.model_card.json`` artifacts through the contract in
``biosqa.model.model_card`` and never imports Plan 1's training
code.
"""

__version__ = "0.0.1"
