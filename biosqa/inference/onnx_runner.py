"""onnxruntime session management for Plan 1's exported quality models (Plan 2 §7.1/§7.2).

One ``InferenceSession`` per modality, CPU execution provider (Plan 2 §5:
the FP32 ``<modality>.onnx`` models are small/fast enough that GPU EPs aren't
required). Loading a session always goes through the model-card handshake in
``biosqa.model.model_card`` first -- refuse to run rather than
silently mis-predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biosqa.inference.preprocess import make_windows, normalize_window
from biosqa.model.model_card import (
    Head,
    ModelCard,
    ModelCardError,
    load_model_card,
    validate_onnx_input_shape,
)

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - onnxruntime is a required app dep; guarded for import-time safety
    ort = None  # type: ignore[assignment]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis (numerically stable)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Elementwise logistic sigmoid (multilabel activation)."""
    return 1.0 / (1.0 + np.exp(-logits))


def _activate(head: Head, logits: np.ndarray) -> np.ndarray:
    """Map a head's raw ONNX output to probabilities per its declared activation."""
    return _softmax(logits) if head.activation == "softmax" else _sigmoid(logits)


@dataclass(frozen=True)
class MultiHeadPrediction:
    """Per-head probabilities for a batch of windows (Plan 1 §12.1 multi-head).

    ``per_head`` maps each model-card head *name* -> ``[n_windows, n_labels]``
    probability array (softmax rows for ordinal/binary, independent sigmoids for
    the multilabel artifact head). ``primary`` is the ordinal Q-grade head that
    legacy single-head callers care about.
    """

    per_head: dict[str, np.ndarray]
    primary_name: str

    @property
    def primary(self) -> np.ndarray:
        """The ordinal Q-grade head probabilities ``[n_windows, n_classes]``."""
        return self.per_head[self.primary_name]

    def get(self, head_name: str) -> np.ndarray | None:
        """Probabilities for ``head_name``, or ``None`` if the card has no such head."""
        return self.per_head.get(head_name)


class OnnxRunner:
    """Loads and runs one modality's ONNX quality model.

    TODO(Plan2 §7.2): batch windows instead of looping one-at-a-time once a
    real model is available to benchmark against; this skeleton favors
    clarity over throughput.
    """

    def __init__(self, modality: str, models_dir: str | Path):
        self.modality = modality
        self.models_dir = Path(models_dir)
        self.card: ModelCard | None = None
        self._session: "ort.InferenceSession | None" = None
        self._input_name: str | None = None
        self._spec_input_name: str | None = None  # 2nd input for dual-branch models (spectral channels)
        self._spec_params: dict | None = None
        self._feat_input_name: str | None = None  # 2nd input for fusion models (SQI feature vector)
        self._feat_params: dict | None = None
        self._output_index: dict[str, int] = {}  # ONNX output tensor name -> run() position

    def load(self) -> None:
        """Validate the model card and open the ONNX session (fail loudly on mismatch)."""
        if ort is None:
            raise RuntimeError("onnxruntime is required to load an OnnxRunner (see app/pyproject.toml)")

        card_path = self.models_dir / f"{self.modality}.model_card.json"
        onnx_path = self.models_dir / f"{self.modality}.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path}: ONNX model not found -- see app/models/README.md "
                "for the expected Plan 1 handshake artifacts"
            )

        self.card = load_model_card(card_path)
        # The runner's modality (from the filename) drives the host-side feature pack / SQI / novelty /
        # guard, while the card's declared modality drives the fused-vector routing. A mismatch would split
        # them silently (e.g. compute PPG features for an EEG-labelled card), so pin them together.
        if self.card.modality and self.card.modality != self.modality:
            raise ModelCardError(
                f"{card_path}: card modality {self.card.modality!r} != runner modality {self.modality!r} "
                f"— the filename and the card disagree on the signal type"
            )
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = self._session.get_inputs()
        input_meta = inputs[0]
        validate_onnx_input_shape(self.card, tuple(input_meta.shape))
        self._input_name = input_meta.name

        # Two-input dual-branch model (x_raw + precomputed spectral channels): the app
        # computes the second input host-side (numpy STFT band power). Require the card's
        # spectral_preprocessing spec so we reproduce the exact training transform.
        if len(inputs) == 2:
            # The 2nd input is either precomputed spectral CHANNELS (dual-branch) or a
            # hand-crafted SQI feature VECTOR (fusion); the card says which and how.
            if self.card.spectral_preprocessing:
                self._spec_input_name = inputs[1].name
                self._spec_params = self.card.spectral_preprocessing
            elif self.card.feature_preprocessing:
                self._feat_input_name = inputs[1].name
                self._feat_params = self.card.feature_preprocessing
                self._validate_feature_contract(inputs[1], onnx_path)
            else:
                raise ModelCardError(
                    f"{onnx_path}: ONNX has 2 inputs but the model card has neither a "
                    f"'spectral_preprocessing' nor a 'feature_preprocessing' spec to compute the "
                    f"second input (modality={self.modality!r})"
                )
        elif len(inputs) != 1:
            raise ModelCardError(f"{onnx_path}: unexpected input count {len(inputs)} (modality={self.modality!r})")

        # Every head declared by the card must correspond to a real graph output;
        # a card/graph mismatch is a hard handshake failure (Plan 2 §11/§14), not
        # a silent wrong prediction.
        self._output_index = {out.name: i for i, out in enumerate(self._session.get_outputs())}
        missing = [h.output_name for h in self.card.heads if h.output_name not in self._output_index]
        if missing:
            raise ModelCardError(
                f"{onnx_path}: model card declares head output(s) {missing!r} absent from the ONNX "
                f"graph outputs {list(self._output_index)!r} (modality={self.modality!r})"
            )
        # Also validate each head's OUTPUT WIDTH against its class_order length: a card listing
        # Q0..Q3 (4) against a 3-wide graph output would silently never emit the last class (argmax
        # stays in-range, no error). Skip symbolic/dynamic last dims (not statically checkable).
        outs = self._session.get_outputs()
        for h in self.card.heads:
            n = h.n_labels
            shp = outs[self._output_index[h.output_name]].shape
            w = shp[-1] if shp else None
            if isinstance(w, int) and n > 0 and w != n:
                raise ModelCardError(
                    f"{onnx_path}: head {h.output_name!r} graph output width {w} != card class_order "
                    f"length {n} (modality={self.modality!r}) — the extra class(es) would never be emitted"
                )

    def _validate_feature_contract(self, feat_input, onnx_path) -> None:
        """Pin the fusion 2nd-input feature vector to the card, at load time. The graph bakes per-feature
        standardization BY POSITION, so a same-length reorder/rename in the research ``combined_vector`` would
        silently permute the fused input and corrupt every grade/usable prediction with no error. ``novelty``
        and ``feature_attribution`` already guard the order against the card; the primary inference path must
        too. Asserts (a) runtime feature NAMES == card ``feat_names``, and (b) the graph 2nd-input width ==
        the card feature count (when statically known)."""
        p = self._feat_params or {}
        if p.get("fn") != "combined_vector":
            return
        card_names = list(p.get("feat_names") or [])
        rt_names = None
        try:
            from biosqa.inference.sqa_features import combined_vector
            probe = np.sin(2.0 * np.pi * 5.0 * np.arange(self.card.l_m) / max(1, self.card.l_m))
            _, rt_names = combined_vector(probe.reshape(1, 1, -1).astype(np.float32),
                                          float(self.card.fs_hz), p.get("modality", self.modality))
            rt_names = list(rt_names)
        except Exception:  # noqa: BLE001 - a probe failure shouldn't block load; the width check still applies
            rt_names = None
        if card_names and rt_names is not None and rt_names != card_names:
            raise ModelCardError(
                f"{onnx_path}: combined_vector feature names/order do not match the card's "
                f"feature_preprocessing.feat_names (modality={self.modality!r}) — the graph standardizes the "
                f"fused input by POSITION, so a reordered vector would silently corrupt every prediction"
            )
        w = feat_input.shape[-1] if getattr(feat_input, "shape", None) else None
        n_feat = p.get("n_features") or len(card_names)
        if isinstance(w, int) and n_feat and w != n_feat:
            raise ModelCardError(
                f"{onnx_path}: fusion 2nd-input width {w} != card feature count {n_feat} "
                f"(modality={self.modality!r})"
            )

    def _run_raw(self, windows: np.ndarray) -> list[np.ndarray]:
        """Run all ONNX outputs once over a ``[n_windows, L_m]`` batch.

        Returns the raw output tensors in graph order. Batches the whole window
        stack in a single ``session.run`` (the exported batch dim is dynamic).
        """
        if self._session is None or self.card is None or self._input_name is None:
            raise RuntimeError("OnnxRunner.load() must be called before inference")
        batch = np.asarray(windows, dtype=np.float32).reshape(-1, 1, self.card.l_m)
        if self._feat_input_name is not None:
            # 2-input fusion: compute the SQI feature VECTOR host-side from the raw window.
            # The features are scale-invariant and the graph bakes their standardization, so
            # we feed the RAW vector (and x_raw raw — its instance-norm is baked too). The card
            # names the numpy fn: 'combined_vector' (pack ++ advanced) or legacy 'ppg_sqi_vector'.
            p = self._feat_params or {}
            fn = p.get("fn", "ppg_sqi_vector")
            if fn == "combined_vector":
                from biosqa.inference.sqa_features import combined_vector
                feat = combined_vector(batch, float(self.card.fs_hz), p.get("modality", self.modality))[0]
            else:
                from biosqa.inference.ppg_features import ppg_sqi_vector
                feat = ppg_sqi_vector(batch, float(self.card.fs_hz))
            return self._session.run(None, {self._input_name: batch, self._feat_input_name: feat.astype(np.float32)})
        if self._spec_input_name is None:
            return self._session.run(None, {self._input_name: batch})
        # 2-input dual-branch: compute spectral channels on the PER-WINDOW z-scored signal
        # (band power is scale-dependent -> match the training normalization; the graph then
        # z-scores both inputs again). x_raw is fed raw (its instance-norm is scale-invariant).
        from biosqa.inference.spectral import spectral_band_channels
        p = self._spec_params or {}
        zb = (batch - batch.mean(-1, keepdims=True)) / (batch.std(-1, keepdims=True) + 1e-6)
        spec = spectral_band_channels(zb, float(self.card.fs_hz), [tuple(b) for b in p["bands_hz"]],
                                      frame_s=p.get("frame_s", 0.25), hop_s=p.get("hop_s", 0.0625))
        return self._session.run(None, {self._input_name: batch, self._spec_input_name: spec.astype(np.float32)})

    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        """Run inference and return the PRIMARY (ordinal Q-grade) head's raw logits.

        Backward-compatible with the single-head skeleton: returns a
        ``[n_windows, n_classes]`` array of the primary head's raw output.
        Multi-output models are handled by selecting the primary head's output
        tensor by name (rather than assuming a single output).

        Args:
            windows: ``[n_windows, L_m]`` float32 array, already normalized
                via ``inference.preprocess.normalize_window``.
        """
        if self._session is None or self.card is None:
            raise RuntimeError("OnnxRunner.load() must be called before predict_windows()")
        if len(windows) == 0:
            return np.empty((0, self.card.n_classes), dtype=np.float32)
        outputs = self._run_raw(windows)
        primary = self.card.primary_head
        return np.asarray(outputs[self._output_index[primary.output_name]], dtype=np.float32)

    def predict_windows_multihead(self, windows: np.ndarray) -> MultiHeadPrediction:
        """Run inference over all declared heads and return per-head probabilities.

        Each head's raw ONNX output is mapped to probabilities via its declared
        activation (softmax for ordinal/binary, independent sigmoids for the
        multilabel artifact head).

        Args:
            windows: ``[n_windows, L_m]`` float32 array, already normalized.
        """
        if self._session is None or self.card is None:
            raise RuntimeError("OnnxRunner.load() must be called before predict_windows_multihead()")
        primary_name = self.card.primary_head.name
        if len(windows) == 0:
            per_head = {
                h.name: np.empty((0, h.n_labels), dtype=np.float32) for h in self.card.heads
            }
            return MultiHeadPrediction(per_head=per_head, primary_name=primary_name)
        outputs = self._run_raw(windows)
        per_head = {}
        for head in self.card.heads:
            raw = np.asarray(outputs[self._output_index[head.output_name]], dtype=np.float32)
            per_head[head.name] = _activate(head, raw).astype(np.float32)
        return MultiHeadPrediction(per_head=per_head, primary_name=primary_name)

    def has_feature_attribution(self) -> bool:
        """True iff this model fuses the combined SQI+dynamics vector as its 2nd input AND the card ships a
        reference background for it — the precondition for group-Shapley grade attribution (the dual-branch
        ECG model uses spectral channels and grade<-raw, so it returns False)."""
        return bool(
            self.card is not None
            and self._feat_input_name is not None
            and (self._feat_params or {}).get("fn") == "combined_vector"
            and self.card.feature_attribution
        )

    def combined_feature_vector(self, window_norm: np.ndarray) -> "tuple[np.ndarray, list[str]]":
        """The fused SQI+dynamics vector (+ its feature names) for ONE already-normalized window — the exact
        2nd input the fusion model receives at inference."""
        from biosqa.inference.sqa_features import combined_vector
        batch = np.asarray(window_norm, dtype=np.float32).reshape(1, 1, self.card.l_m)
        v, names = combined_vector(batch, float(self.card.fs_hz),
                                   (self._feat_params or {}).get("modality", self.modality))
        return np.asarray(v[0], dtype=np.float32), list(names)

    def grade_probs_with_feat(self, window_norm: np.ndarray, feat: np.ndarray) -> np.ndarray:
        """Primary (grade) probabilities for one normalized window run with a CUSTOM feature vector — the
        counterfactual forward pass group-Shapley attribution needs (perturb the fused 2nd input while
        holding the raw window fixed). Bypasses the host-side combined_vector so the caller controls the
        fused features."""
        if self._session is None or self.card is None or self._feat_input_name is None:
            raise RuntimeError("grade_probs_with_feat requires a loaded fusion (combined_vector) model")
        batch = np.asarray(window_norm, dtype=np.float32).reshape(1, 1, self.card.l_m)
        f = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        outs = self._session.run(None, {self._input_name: batch, self._feat_input_name: f})
        primary = self.card.primary_head
        raw = np.asarray(outs[self._output_index[primary.output_name]], dtype=np.float32)
        return _activate(primary, raw)[0]

    def _normalized_windows(self, signal: np.ndarray, overlap: float) -> np.ndarray:
        """Window a full-length signal and apply the card's normalization."""
        if self.card is None:
            raise RuntimeError("OnnxRunner.load() must be called before inference")
        windows = make_windows(signal, self.card, overlap=overlap)
        if len(windows) == 0:
            return windows
        return np.stack([normalize_window(w, self.card.normalization) for w in windows])

    def run_sliding_window(self, signal: np.ndarray, overlap: float = 0.0) -> np.ndarray:
        """End-to-end: window -> normalize -> predict for one channel's full signal.

        Returns the primary (ordinal) head's raw logits, backward-compatible with
        the single-head skeleton. This is the function a ``workers.qt_threads``
        ``QRunnable`` should call on a background thread; it never touches Qt/QML.
        """
        return self.predict_windows(self._normalized_windows(signal, overlap))

    def run_sliding_window_multihead(self, signal: np.ndarray, overlap: float = 0.0) -> MultiHeadPrediction:
        """End-to-end multi-head variant: window -> normalize -> all heads.

        The inference worker uses this so the segmenter can carry the artifact
        head's per-window tags alongside the Q-grade track.
        """
        return self.predict_windows_multihead(self._normalized_windows(signal, overlap))

    def guard_record(self, signal: np.ndarray, prediction: "MultiHeadPrediction | None" = None,
                     overlap: float = 0.0, bsqi_corrupt: float = 0.72) -> dict:
        """False-clean guard for a full-signal inference (Tier 1 detector + Tier 2 integrity override).

        The models are trained on RAW signals; a pre-filtered but still-corrupted input can be scored
        clean because filtering strips the high-frequency cues the model keys on. This returns a guard
        report the UI can surface: a per-RECORD pre-filter warning, plus a per-WINDOW integrity override
        that re-flags windows the (filter-robust) bSQI voter marks corrupt where the model read clean —
        active ONLY on pre-filtered input, so raw-signal behaviour is unchanged.

        ``signal`` is the (single-channel) signal that was scored; pass ``prediction`` to reuse an
        already-computed :meth:`run_sliding_window_multihead` result. Returns
        ``{"prefiltered", "reasons", "override_mask" [n_windows] bool, "n_overridden", "score"}``.
        """
        from biosqa.inference.integrity import integrity_guard
        from biosqa.inference.prefilter import detect_prefiltering

        if self.card is None:
            raise RuntimeError("OnnxRunner.load() must be called before guard_record()")
        fs = float(self.card.fs_hz)
        pf = detect_prefiltering(signal, fs, self.modality)
        windows = make_windows(signal, self.card, overlap=overlap)
        if len(windows) == 0:
            return {"prefiltered": pf.prefiltered, "reasons": pf.reasons,
                    "override_mask": np.zeros(0, dtype=bool), "n_overridden": 0, "score": pf.score}
        if prediction is None:
            prediction = self.predict_windows_multihead(np.stack(
                [normalize_window(w, self.card.normalization) for w in windows]))
        # per-window P(unusable): prefer the 'usable' head [unusable, usable]; else derive from grade (Q<2)
        usable = prediction.get("usable")
        if usable is not None and len(usable):
            p_unusable = usable[:, 0]
        else:
            g = prediction.primary
            p_unusable = 1.0 - (g[:, 2] + g[:, 3]) if g.shape[1] >= 4 else g[:, 0]
        mask = np.array([
            integrity_guard(w, fs, self.modality, float(pu), pf, bsqi_corrupt=bsqi_corrupt).corrupt_override
            for w, pu in zip(windows, p_unusable)
        ], dtype=bool)
        return {"prefiltered": pf.prefiltered, "reasons": pf.reasons,
                "override_mask": mask, "n_overridden": int(mask.sum()), "score": pf.score}
