"""Parse and validate `model_card.json`, the Plan 1 <-> Plan 2 handshake (Plan 2 §11).

The card is the *only* place normalization constants, window length, sample
rate, and class order should ever live. Silently hard-coding these in the
app instead of reading them from the card is exactly the failure mode Plan 2
§14 calls out ("preprocessing drift ... silently degrades predictions") --
so validation here is intentionally strict and fails loudly.
"""

from __future__ import annotations

import hashlib
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
    onnx_sha256: str | None = None               # OPTIONAL integrity digest of the sibling .onnx; canonical lowercase hex

    @property
    def n_classes(self) -> int:
        return len(self.class_order)

    @property
    def grade_temperature(self) -> float:
        """Temperature-scaling factor for the grade head (1.0 = none) — DOCUMENTATION of the constant
        baked into the ONNX graph, not an instruction. The graph's final grade path divides the
        log-probabilities by this value, so the softmax the session returns is already the scaled one
        the conformal threshold was calibrated on; re-applying it host-side would scale twice."""
        try:
            return float(self.calibration["temperatures"]["grade"])  # type: ignore[index]
        except (TypeError, KeyError, ValueError):
            return 1.0

    @property
    def usable_temperature(self) -> float:
        """Temperature-scaling factor for the binary usable/unusable head (1.0 = none).

        Every shipped card calibrates this alongside ``grade``, and — like ``grade`` — the constant is
        BAKED into the graph, so this documents what the head's probabilities already mean rather than
        something to apply. Absent/unparseable -> 1.0 (identity), which is the honest no-op, not a guess.
        """
        try:
            return float(self.calibration["temperatures"]["usable"])  # type: ignore[index]
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

    def verify_onnx(self, onnx_path: str | Path) -> str | None:
        """Check the .onnx artifact against the card's OPTIONAL ``onnx_sha256`` digest.

        Model identity is otherwise a free-text label (``model_version``): without this, a
        swapped, truncated or corrupted .onnx loads happily as long as the card says the right
        words. Callers should run this before opening the session -- refuse to run rather than
        predict from an unknown model.

        Returns the verified digest, or ``None`` when the card carries no digest: verification is
        then SKIPPED, never faked. A mismatch is a hard ``ModelCardError``.

        Every shipped card carries a digest, so this is live on every real load. It catches the one
        substitution the runner's other checks cannot, as an EEG build once demonstrated: a stale v4
        ``eeg.onnx`` under ``app/dist/`` had exactly the same byte SIZE (2,298,439) as the v5 graph,
        with the same modality, L_m (1280), head names and head widths -- identical under every
        structural check, and distinguishable only by digest (2f10ee74... vs cf80f542...). Those EEG
        weights are no longer shipped (see LICENSE-MODELS), but the failure mode is generic and the
        gate applies to whatever is in ``models/``, including weights a user supplies themselves.
        """
        if self.onnx_sha256 is None:
            return None
        path = Path(onnx_path)
        if not path.exists():
            raise ModelCardError(
                f"{path}: ONNX model not found -- cannot verify the card's onnx_sha256 digest"
            )
        actual = sha256_file(path)
        if actual != self.onnx_sha256:
            raise ModelCardError(
                f"{path}: ONNX SHA-256 mismatch -- {self.source_path.name} declares "
                f"{self.onnx_sha256} but the file hashes to {actual}. The model artifact does not "
                f"match its card (swapped, corrupted, or truncated) -- refusing to load."
            )
        return actual


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 a file in streaming chunks, returning lowercase hex.

    Chunked because the .onnx artifacts are megabytes and this runs on the load path.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_onnx_sha256(raw: dict[str, Any], source_path: Path) -> str | None:
    """Parse the OPTIONAL ``onnx_sha256`` integrity field into canonical lowercase hex.

    Absent -> ``None`` (every card shipped before this field existed keeps loading; the app
    just skips verification). Present but malformed is a HARD error rather than a silent
    downgrade to "unverified" -- a typo'd digest must not read as "no integrity check".
    Accepts a bare 64-char hex digest or a ``sha256:``-prefixed one (as ``training_data_hash``
    is written).
    """
    value = raw.get("onnx_sha256")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelCardError(f"{source_path}: 'onnx_sha256' must be a string")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:"):]
    if len(digest) != 64 or digest.strip("0123456789abcdef"):
        raise ModelCardError(
            f"{source_path}: 'onnx_sha256' must be a 64-char hex SHA-256 digest "
            f"(optionally 'sha256:'-prefixed), got {value!r}"
        )
    return digest


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
        onnx_sha256=_parse_onnx_sha256(raw, source_path),
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
