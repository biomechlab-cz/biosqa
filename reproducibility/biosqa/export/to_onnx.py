"""ONNX export + parity gate + CPU latency (Plan 1 §11 — deployment contract).

The app (Plan 2) consumes ONLY the exported per-modality ONNX graph:
``float32 [B, 1, L_m] -> Q0..Q3 logits``. This module exports it, verifies logit
parity against PyTorch (target <1% drift), optionally applies ORT dynamic
quantization (guidance for transformer backbones), and times CPU inference.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from ..models.model import BioSQAModel, SingleModalityExport

__all__ = ["export_onnx", "parity_check", "quantize_dynamic_onnx", "cpu_latency_ms", "export_and_verify",
           "export_multihead_onnx", "parity_check_multihead", "write_model_card", "export_and_verify_multihead",
           "sha256_file"]

_HEAD_OUTPUT_NAME = {"q": "q_logits", "binary": "bin_logits", "type": "type_logits"}


def export_onnx(
    model: BioSQAModel,
    modality: str,
    length: int,
    path: str | Path,
    *,
    c_in: int = 1,
    opset: int = 17,
    batch: int = 1,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SingleModalityExport(model.eval().cpu(), modality).eval()
    dummy = torch.randn(batch, c_in, length)
    # dummy batch=2 (not 1) so the batch dim can't be trivially specialised.
    dummy = torch.randn(max(2, batch), c_in, length)
    # Legacy exporter (dynamo=False): reliably honours dynamic_axes for the
    # transformer/Mamba reshapes. The new dynamo path specialised the batch dim
    # to the dummy size and broke Reshape at batch>1 (Phase-0 finding).
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["window"],
        output_names=["q_logits"],
        opset_version=opset,
        dynamic_axes={"window": {0: "batch"}, "q_logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    return path


def parity_check(
    model: BioSQAModel, onnx_path: str | Path, modality: str, length: int, *, c_in: int = 1, n: int = 8
) -> dict:
    """Max/relative logit deviation between PyTorch and onnxruntime (CPU)."""
    import onnxruntime as ort

    wrapper = SingleModalityExport(model.eval().cpu(), modality).eval()
    x = torch.randn(n, c_in, length)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(["q_logits"], {"window": x.numpy().astype(np.float32)})[0]
    abs_err = float(np.max(np.abs(ref - out)))
    denom = float(np.max(np.abs(ref))) + 1e-8
    return {
        "max_abs_logit_diff": abs_err,
        "max_rel_logit_diff": abs_err / denom,
        "argmax_agree": float(np.mean(ref.argmax(1) == out.argmax(1))),
    }


def quantize_dynamic_onnx(in_path: str | Path, out_path: str | Path) -> Path:
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    in_path, out_path = Path(in_path), Path(out_path)
    # ORT-recommended: shape-infer + fold constants first so the dynamic quantizer
    # doesn't trip the version converter on dynamo-exported graphs. The pre-processed
    # graph is still FP32, so it goes to a scratch dir — writing it next to the
    # deliverables left `<modality>.int8.pre.onnx` files that a packaging glob would
    # pick up as INT8 artifacts.
    with tempfile.TemporaryDirectory() as tmp:
        src = in_path
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            pre = Path(tmp) / f"{out_path.stem}.pre.onnx"
            quant_pre_process(str(in_path), str(pre), skip_symbolic_shape=True)
            src = pre
        except Exception:
            pass  # fall back to quantizing the raw graph
        quantize_dynamic(str(src), str(out_path), weight_type=QuantType.QInt8)
    return out_path


def cpu_latency_ms(onnx_path: str | Path, length: int, *, c_in: int = 1, n_iters: int = 200, warmup: int = 20) -> float:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # single-thread = honest per-window edge latency
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, c_in, length).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"window": x})
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess.run(None, {"window": x})
    return (time.perf_counter() - t0) / n_iters * 1000.0


def export_and_verify(
    model: BioSQAModel, modality: str, length: int, out_dir: str | Path, *, c_in: int = 1, quantize: bool = True
) -> dict:
    """Full gate: export -> parity -> (quantize -> parity) -> latency. Returns a
    verdict dict; ``passes_parity`` is True when rel drift < 1% (Plan 1 §11.1)."""
    out_dir = Path(out_dir)
    fp32 = export_onnx(model, modality, length, out_dir / f"{modality}.onnx", c_in=c_in)
    verdict = {"modality": modality, "length": length, "onnx": str(fp32)}
    verdict["fp32_parity"] = parity_check(model, fp32, modality, length, c_in=c_in)
    verdict["fp32_latency_ms"] = cpu_latency_ms(fp32, length, c_in=c_in)
    if quantize:
        try:
            q8 = quantize_dynamic_onnx(fp32, out_dir / f"{modality}.int8.onnx")
            verdict["int8_onnx"] = str(q8)
            verdict["int8_parity"] = parity_check(model, q8, modality, length, c_in=c_in)
            verdict["int8_latency_ms"] = cpu_latency_ms(q8, length, c_in=c_in)
        except Exception as e:  # quantization can fail on some ops; report, don't crash
            verdict["int8_error"] = repr(e)
    verdict["passes_parity"] = verdict["fp32_parity"]["max_rel_logit_diff"] < 0.01
    return verdict


# --- multi-head (3-level) deployment: q_logits + bin_logits + type_logits -----

def export_multihead_onnx(
    model: BioSQAModel, modality: str, length: int, path: str | Path, *,
    c_in: int = 1, opset: int = 17, output_order=("q", "binary", "type"),
    instance_norm: bool = True, temperature: dict | None = None,
) -> tuple[Path, list[str]]:
    """Export the multi-head graph: ``[B,1,L] -> (q_logits[, bin_logits][, type_logits])``
    as named RAW-logit outputs with a dynamic batch axis on the input and every
    output (the app batches windows + applies softmax/sigmoid per head). Only the
    heads present on the model are emitted. ``temperature`` (head->T) is baked in
    so the app's softmax is already calibrated. Legacy exporter (``dynamo=False``)."""
    from ..models.model import MultiHeadExport

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = MultiHeadExport(model.eval().cpu(), modality, output_order, instance_norm, temperature).eval()
    out_names = [_HEAD_OUTPUT_NAME[h] for h in wrapper.active]
    dummy = torch.randn(2, c_in, length)
    dynamic = {"window": {0: "batch"}, **{n: {0: "batch"} for n in out_names}}
    torch.onnx.export(
        wrapper, (dummy,), str(path), input_names=["window"], output_names=out_names,
        opset_version=opset, dynamic_axes=dynamic, do_constant_folding=True, dynamo=False,
    )
    return path, out_names


def parity_check_multihead(
    model: BioSQAModel, onnx_path: str | Path, modality: str, length: int, *,
    c_in: int = 1, n: int = 8, output_order=("q", "binary", "type"), instance_norm: bool = True,
    temperature: dict | None = None,
) -> dict:
    """Per-output logit parity (PyTorch vs onnxruntime); ``passes`` if every head
    is under 1% relative drift."""
    import onnxruntime as ort

    from ..models.model import MultiHeadExport

    wrapper = MultiHeadExport(model.eval().cpu(), modality, output_order, instance_norm, temperature).eval()
    out_names = [_HEAD_OUTPUT_NAME[h] for h in wrapper.active]
    x = torch.randn(n, c_in, length)
    with torch.no_grad():
        ref = wrapper(x)
    ref = (ref,) if isinstance(ref, torch.Tensor) else tuple(ref)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(out_names, {"window": x.numpy().astype(np.float32)})
    res: dict = {}
    for name, r, o in zip(out_names, ref, outs):
        r = r.numpy()
        ae = float(np.max(np.abs(r - o)))
        res[name] = {"max_abs": ae, "max_rel": ae / (float(np.max(np.abs(r))) + 1e-8)}
    res["passes"] = all(v["max_rel"] < 0.01 for v in res.values())
    return res


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 a file in streaming chunks, returning lowercase hex.

    Mirrors ``app.biosqa.model.model_card.sha256_file`` — the app recomputes this digest at load
    time and refuses the session on a mismatch, so producer and consumer must agree byte for byte.
    Chunked because the .onnx artifacts are megabytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_card(
    path: str | Path, *, modality: str, length: int, fs_hz: float,
    class_order, out_names, binary_class_order=("BAD", "OK"),
    artifact_class_order=None, artifact_threshold: float = 0.5,
    training_data_hash: str = "", model_version: str = "v2", normalization: dict | None = None,
    calibration: dict | None = None, ood: dict | None = None, onnx_path: str | Path | None = None,
) -> dict:
    """Write the app v2 multi-head ``model_card.json`` (Plan 2 §11 handshake).

    Normalization defaults to ``{"method": "none"}`` because per-window instance
    z-score is baked into the exported graph (:class:`MultiHeadExport`).
    ``calibration`` records the baked-in per-head temperatures + ECE (provenance);
    ``ood`` records the host-side conformal abstention threshold (metrology gate).

    ``onnx_path`` pins the card to a specific graph via ``onnx_sha256``. Pass the FINAL
    artifact — the app hashes the file it is about to open and refuses on a mismatch, so a
    digest taken before a later rewrite is worse than none. ``model_version`` alone is
    free text and cannot separate two same-shape graphs: ``app/dist/.../eeg.onnx`` is a v4
    model of the identical byte size (2,298,439) and identical L_m/head contract as the
    shipped v5, so every structural check passes and only the digest differs."""
    import json

    heads = [{"name": "grade", "output_name": "q_logits", "kind": "ordinal",
              "activation": "softmax", "class_order": list(class_order)}]
    if "bin_logits" in out_names:
        heads.append({"name": "usable", "output_name": "bin_logits", "kind": "binary",
                      "activation": "softmax", "class_order": list(binary_class_order)})
    if "type_logits" in out_names:
        heads.append({"name": "artifact", "output_name": "type_logits", "kind": "multilabel",
                      "activation": "sigmoid", "class_order": list(artifact_class_order or []),
                      "threshold": artifact_threshold})
    card = {
        "modality": modality, "L_m": int(length), "fs_hz": float(fs_hz),
        "class_order": list(class_order),
        "normalization": normalization or {"method": "none"},
        "training_data_hash": training_data_hash, "model_version": model_version,
        "heads": heads,
    }
    if onnx_path is not None:
        card["onnx_sha256"] = sha256_file(onnx_path)
    if calibration is not None:
        calibration = dict(calibration)
        calibration.setdefault("location", "onnx_graph")
        card["calibration"] = calibration
    if ood is not None:
        card["ood"] = ood
    Path(path).write_text(json.dumps(card, indent=2))
    return card


def export_and_verify_multihead(
    model: BioSQAModel, modality: str, length: int, fs_hz: float, out_dir: str | Path, *,
    class_order, artifact_class_order=None, c_in: int = 1, quantize: bool = True,
    training_data_hash: str = "", model_version: str = "v2",
    temperature: dict | None = None, calibration: dict | None = None, ood: dict | None = None,
) -> dict:
    """Full multi-head gate: export (temperature baked in) -> parity -> (quantize)
    -> latency + write the v2 model card (with calibration/ood metadata)."""
    out_dir = Path(out_dir)
    onnx_path, out_names = export_multihead_onnx(
        model, modality, length, out_dir / f"{modality}.onnx", c_in=c_in, temperature=temperature)
    verdict = {"modality": modality, "length": length, "onnx": str(onnx_path), "outputs": out_names}
    verdict["parity"] = parity_check_multihead(model, onnx_path, modality, length, c_in=c_in, temperature=temperature)
    verdict["fp32_latency_ms"] = cpu_latency_ms(onnx_path, length, c_in=c_in)
    if quantize:
        try:
            q8 = quantize_dynamic_onnx(onnx_path, out_dir / f"{modality}.int8.onnx")
            verdict["int8_onnx"] = str(q8)
            # Never report an INT8 artifact we haven't measured (the single-head gate
            # already does this). `passes_parity` stays the FP32 verdict — FP32 is the
            # source of truth and INT8 drift is expected — but the INT8 result is now
            # explicit instead of absent.
            verdict["int8_parity"] = parity_check_multihead(
                model, q8, modality, length, c_in=c_in, temperature=temperature)
            verdict["passes_int8_parity"] = verdict["int8_parity"]["passes"]
            verdict["int8_latency_ms"] = cpu_latency_ms(q8, length, c_in=c_in)
        except Exception as e:
            verdict["int8_error"] = repr(e)
    # Card last, and pinned to onnx_path: quantization writes a SEPARATE {modality}.int8.onnx,
    # so the FP32 graph the card describes is final here and the digest cannot go stale.
    verdict["model_card"] = write_model_card(
        out_dir / f"{modality}.model_card.json", modality=modality, length=length, fs_hz=fs_hz,
        class_order=class_order, out_names=out_names, artifact_class_order=artifact_class_order,
        training_data_hash=training_data_hash, model_version=model_version,
        calibration=calibration, ood=ood, onnx_path=onnx_path)
    verdict["passes_parity"] = verdict["parity"]["passes"]
    return verdict


