"""Parse and validate `model_card.json`, the Plan 1 <-> Plan 2 handshake (Plan 2 §11).

The card is the *only* place normalization constants, window length, sample
rate, and class order should ever live. Silently hard-coding these in the
app instead of reading them from the card is exactly the failure mode Plan 2
§14 calls out ("preprocessing drift ... silently degrades predictions") --
so validation here is intentionally strict and fails loudly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "modality",
    "L_m",
    "fs_hz",
    "class_order",
    "normalization",
    "training_data_hash",
    "model_version",
)

# v2 multi-head contract (Plan 1 §12.1). A card MAY declare an optional ``heads``
# array; when absent we synthesize a single ordinal head from ``class_order`` so
# the rest of the app never special-cases legacy single-head cards.
HEAD_KINDS = ("ordinal", "binary", "multilabel")
ACTIVATIONS = ("softmax", "sigmoid")


class ModelCardError(ValueError):
    """Raised when a model_card.json is missing, malformed, or mismatched."""


@dataclass(frozen=True)
class Normalization:
    """Normalization contract applied in ``inference.preprocess``."""

    method: str  # e.g. "zscore" | "minmax" | "none"
    mean: float | None = None
    std: float | None = None


@dataclass(frozen=True)
class Head:
    """One output head of a (possibly multi-head) exported model (Plan 1 §12.1).

    ``kind`` picks the semantics/activation the app applies to this ONNX output:

    * ``ordinal`` — the Q0..Q3 grade (softmax). Exactly one per card; it is the
      *primary* head and its ``class_order`` MUST equal the card's top-level
      ``class_order`` (so ``n_classes`` keeps its meaning).
    * ``binary`` — usable/unusable OK/BAD (softmax, exactly two classes).
    * ``multilabel`` — artifact TYPE (independent per-class sigmoids, NOT softmax:
      artifacts co-occur). Requires ``threshold`` in ``[0, 1]``.
    """

    name: str
    output_name: str  # the ONNX graph output tensor name this head reads
    kind: str  # one of HEAD_KINDS
    activation: str  # one of ACTIVATIONS
    class_order: tuple[str, ...]
    threshold: float | None = None  # multilabel only

    @property
    def n_labels(self) -> int:
        return len(self.class_order)


@dataclass(frozen=True)
class ModelCard:
    """Validated, typed view of a `<modality>.model_card.json` file."""

    modality: str
    l_m: int
    fs_hz: float
    class_order: tuple[str, ...]
    normalization: Normalization
    training_data_hash: str
    model_version: str
    source_path: Path
    heads: tuple[Head, ...] = ()
    spectral_preprocessing: dict | None = None  # 2-input dual-branch models: 2nd input = spectral CHANNELS
    feature_preprocessing: dict | None = None   # 2-input fusion models: 2nd input = a hand-crafted SQI VECTOR
    calibration: dict | None = None             # {"temperatures": {"grade": T, "usable": T}, "grade_ece": ...}
    ood: dict | None = None                      # {"method": "conformal_aps", "grade_nonconformity_threshold": τ, "alpha": α}
    novelty: dict | None = None                  # {"method": "mahalanobis_sqi", mean, std, inv_corr, d2_threshold, feature_names}
    feature_attribution: dict | None = None      # {"feature_fn": "combined_vector", feature_names, reference_mean, reference_std} — background for group-Shapley grade attribution (fusion models only)

    @property
    def n_classes(self) -> int:
        return len(self.class_order)

    @property
    def grade_temperature(self) -> float:
        """Temperature-scaling factor for the grade head (1.0 = none). The conformal threshold was
        calibrated on temperature-scaled grade probabilities, so it MUST be applied before APS decoding."""
        try:
            return float(self.calibration["temperatures"]["grade"])  # type: ignore[index]
        except (TypeError, KeyError, ValueError):
            return 1.0

    @property
    def conformal_threshold(self) -> float | None:
        """APS nonconformity threshold τ for the grade head, or ``None`` when the card ships no
        conformal block (the app then simply skips prediction sets)."""
        try:
            if self.ood and self.ood.get("method") == "conformal_aps":
                return float(self.ood["grade_nonconformity_threshold"])
        except (TypeError, KeyError, ValueError):
            pass
        return None

    @property
    def conformal_alpha(self) -> float:
        """Miscoverage level α (target coverage = 1−α, e.g. 0.1 → 90%)."""
        try:
            return float(self.ood["alpha"]) if self.ood else 0.1  # type: ignore[index]
        except (TypeError, KeyError, ValueError):
            return 0.1

    @property
    def primary_head(self) -> Head:
        """The ordinal Q-grade head (the card's primary output)."""
        for head in self.heads:
            if head.kind == "ordinal":
                return head
        return self.heads[0]

    @property
    def artifact_head(self) -> Head | None:
        """The multilabel artifact-type head, or ``None`` if the card has none."""
        for head in self.heads:
            if head.kind == "multilabel":
                return head
        return None

    def head(self, name: str) -> Head:
        """Look up a head by name, raising ``ModelCardError`` if it is absent."""
        for head in self.heads:
            if head.name == name:
                return head
        raise ModelCardError(
            f"{self.source_path}: no head named {name!r} (have {[h.name for h in self.heads]!r})"
        )


def _validate_raw(raw: dict[str, Any], source_path: Path) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ModelCardError(
            f"{source_path}: missing required field(s) {missing!r} "
            f"(Plan 2 §11 contract: {REQUIRED_FIELDS!r})"
        )
    if not isinstance(raw["class_order"], list) or not raw["class_order"]:
        raise ModelCardError(f"{source_path}: 'class_order' must be a non-empty list")
    if not isinstance(raw["normalization"], dict) or "method" not in raw["normalization"]:
        raise ModelCardError(f"{source_path}: 'normalization' must be an object with a 'method' key")
    if not isinstance(raw["L_m"], int) or raw["L_m"] <= 0:
        raise ModelCardError(f"{source_path}: 'L_m' must be a positive integer")


def _parse_heads(raw: dict[str, Any], class_order: list[str], source_path: Path) -> tuple[Head, ...]:
    """Parse the optional v2 ``heads`` array (Plan 1 §12.1), or synthesize a
    single ordinal head from ``class_order`` for legacy (single-head) cards.

    Validation is strict and loud (Plan 2 §11/§14): unknown kind/activation,
    a multilabel head without a threshold in ``[0, 1]``, a binary head that is
    not two-class, duplicate head names/output tensors, or an ordinal head whose
    ``class_order`` disagrees with the card's top-level ``class_order`` are all
    hard errors rather than silently-wrong inference.
    """
    heads_raw = raw.get("heads")
    if heads_raw is None:
        # Legacy single-head card: the whole model IS the ordinal grade head.
        return (
            Head(
                name="grade",
                output_name="q_logits",
                kind="ordinal",
                activation="softmax",
                class_order=tuple(class_order),
                threshold=None,
            ),
        )
    if not isinstance(heads_raw, list) or not heads_raw:
        raise ModelCardError(f"{source_path}: 'heads' must be a non-empty list when present")

    heads: list[Head] = []
    seen_names: set[str] = set()
    seen_outputs: set[str] = set()
    for i, h in enumerate(heads_raw):
        if not isinstance(h, dict):
            raise ModelCardError(f"{source_path}: head #{i} must be an object")
        for field_name in ("name", "output_name", "kind", "class_order"):
            if field_name not in h:
                raise ModelCardError(f"{source_path}: head #{i} missing required key {field_name!r}")
        name, output_name, kind = h["name"], h["output_name"], h["kind"]
        if not isinstance(name, str) or not name:
            raise ModelCardError(f"{source_path}: head #{i} 'name' must be a non-empty string")
        if not isinstance(output_name, str) or not output_name:
            raise ModelCardError(f"{source_path}: head {name!r} 'output_name' must be a non-empty string")
        if kind not in HEAD_KINDS:
            raise ModelCardError(f"{source_path}: head {name!r} kind {kind!r} not in {HEAD_KINDS!r}")
        # Default activation follows the kind; an explicit value must be consistent.
        activation = h.get("activation") or ("sigmoid" if kind == "multilabel" else "softmax")
        if activation not in ACTIVATIONS:
            raise ModelCardError(f"{source_path}: head {name!r} activation {activation!r} not in {ACTIVATIONS!r}")
        if kind == "multilabel" and activation != "sigmoid":
            raise ModelCardError(f"{source_path}: multilabel head {name!r} must use 'sigmoid' activation")
        if kind in ("ordinal", "binary") and activation != "softmax":
            raise ModelCardError(f"{source_path}: {kind} head {name!r} must use 'softmax' activation")
        co = h["class_order"]
        if not isinstance(co, list) or not co:
            raise ModelCardError(f"{source_path}: head {name!r} 'class_order' must be a non-empty list")
        if kind == "binary" and len(co) != 2:
            raise ModelCardError(f"{source_path}: binary head {name!r} 'class_order' must have exactly 2 entries")
        threshold = h.get("threshold")
        if kind == "multilabel":
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not (0.0 <= float(threshold) <= 1.0):
                raise ModelCardError(f"{source_path}: multilabel head {name!r} needs a 'threshold' in [0, 1]")
            threshold = float(threshold)
        else:
            threshold = None
        if name in seen_names:
            raise ModelCardError(f"{source_path}: duplicate head name {name!r}")
        if output_name in seen_outputs:
            raise ModelCardError(f"{source_path}: duplicate head output_name {output_name!r}")
        seen_names.add(name)
        seen_outputs.add(output_name)
        heads.append(Head(name, output_name, kind, activation, tuple(co), threshold))

    ordinals = [h for h in heads if h.kind == "ordinal"]
    if not ordinals:
        raise ModelCardError(
            f"{source_path}: at least one 'ordinal' head is required (it maps to the top-level class_order)"
        )
    if tuple(ordinals[0].class_order) != tuple(class_order):
        raise ModelCardError(
            f"{source_path}: ordinal head class_order {ordinals[0].class_order!r} must equal "
            f"the card's top-level class_order {tuple(class_order)!r}"
        )
    return tuple(heads)


def load_model_card(path: str | Path) -> ModelCard:
    """Load and validate a `model_card.json` file, raising ``ModelCardError`` on any mismatch.

    This is the single entry point ``inference.onnx_runner`` must call before
    running any inference session -- refuse to run rather than guess.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise ModelCardError(
            f"{source_path}: model card not found -- see app/models/README.md "
            "for the expected Plan 1 handshake artifacts"
        )
    try:
        raw = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelCardError(f"{source_path}: invalid JSON ({exc})") from exc

    _validate_raw(raw, source_path)
    norm_raw = raw["normalization"]
    heads = _parse_heads(raw, raw["class_order"], source_path)

    return ModelCard(
        modality=raw["modality"],
        l_m=raw["L_m"],
        fs_hz=float(raw["fs_hz"]),
        class_order=tuple(raw["class_order"]),
        normalization=Normalization(
            method=norm_raw["method"],
            mean=norm_raw.get("mean"),
            std=norm_raw.get("std"),
        ),
        training_data_hash=raw["training_data_hash"],
        model_version=raw["model_version"],
        source_path=source_path,
        heads=heads,
        spectral_preprocessing=raw.get("spectral_preprocessing"),
        feature_preprocessing=raw.get("feature_preprocessing"),
        calibration=raw.get("calibration"),
        ood=raw.get("ood"),
        novelty=raw.get("novelty"),
        feature_attribution=raw.get("feature_attribution"),
    )


def _is_dynamic_dim(dim: Any) -> bool:
    """True if an ONNX shape dim is symbolic/dynamic (a name, ``None``, or <= 0)."""
    if isinstance(dim, str):
        return True
    if dim is None:
        return True
    return isinstance(dim, int) and dim <= 0


def validate_onnx_input_shape(card: ModelCard, onnx_input_shape: tuple[int, ...]) -> None:
    """Cross-check an ONNX session's declared input shape against the card's `L_m`.

    Contract: float32 ``[batch, 1, L_m]`` (design spec §11). The batch dim is
    exported as a *dynamic* axis (so the app can batch windows), so it is
    accepted whether it is ``1`` or a symbolic/``None``/``-1`` placeholder; the
    channel (1) and window-length (``L_m``) dims are checked strictly — a
    mismatch there means preprocessing drift and is a hard failure (Plan 2 §14).
    """
    shape = tuple(onnx_input_shape)
    if len(shape) != 3:
        raise ModelCardError(
            f"{card.source_path}: ONNX input rank {len(shape)} (shape {shape}) does not match "
            f"model card contract [batch, 1, {card.l_m}] (modality={card.modality!r})"
        )
    batch, channels, length = shape
    batch_ok = _is_dynamic_dim(batch) or batch == 1
    if not (batch_ok and channels == 1 and length == card.l_m):
        raise ModelCardError(
            f"{card.source_path}: ONNX input shape {shape} does not match "
            f"model card contract [batch, 1, {card.l_m}] (modality={card.modality!r})"
        )
