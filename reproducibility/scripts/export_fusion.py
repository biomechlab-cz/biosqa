"""Generalized SQI-fusion export for any modality (campaign round-2, 2026-07-06).

Banks the promoted COMBINED feature bank (per-modality SQI pack ++ round-2 advanced
dynamics/HOS: spectral-kurtosis, dispersion-entropy, scalar-RQA, jump-ratio, ordinal-
transition, cepstral) fused into the trunk + the ordinal ordered-logit grade head.
Trains FeatureFusionMultiHead, calibrates on store val, exports a TWO-INPUT legacy-ONNX
(x_raw [B,1,L], x_feat [B,D]); the app computes x_feat via sqa_features.combined_vector in
numpy (pure numpy, numpy.rfft only), and the graph bakes both the raw instance-norm AND the
feature standardization, so the app feeds the raw vector.

    python scripts/export_fusion.py --modality eda --epochs 30 --copy-to-app
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosqa.data.harmonize import QUALITY_NAMES  # noqa: E402
from biosqa.data.sqa_features import combined_vector  # noqa: E402
from biosqa.data.store import CANONICAL_FS, WINDOW_S, SegmentStore  # noqa: E402
from biosqa.models.fusion import FeatureFusionExport, FeatureFusionMultiHead  # noqa: E402
from biosqa.train.losses import SORDLoss  # noqa: E402
from biosqa.train.loop import _cosine_warmup  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402
from biosqa.xdomain import calibration as cal  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = {"ecg": 50, "ppg": 32, "eeg": 32, "eda": 8}


def _inorm(X):
    return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)


def guarded_T(logits, labels):
    e0 = cal.expected_calibration_error(cal.apply_temperature(logits, 1.0), labels)
    cands = [1.0, cal.fit_temperature(logits, labels)] + list(np.linspace(0.2, 3.0, 57))
    bT, be = 1.0, e0
    for T in cands:
        e = cal.expected_calibration_error(cal.apply_temperature(logits, T), labels)
        if e < be - 1e-6:
            bT, be = float(T), e
    return bT, e0, be


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["ppg", "eda", "eeg", "ecg"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models")
    ap.add_argument("--store", default="data/store_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-to-app", action="store_true")
    args = ap.parse_args()
    mod = args.modality
    seed_everything(args.seed); rng = np.random.default_rng(args.seed)
    fs = CANONICAL_FS[mod]; L = int(round(WINDOW_S[mod] * fs)); patch = PATCH[mod]
    store = SegmentStore(args.store)

    Xtr, ytr, _, _ = store.load_modality(mod, "train"); ytr = ytr.astype(np.int64)
    Xva, yva, _, _ = store.load_modality(mod, "val"); yva = yva.astype(np.int64)
    Vtr, names = combined_vector(Xtr, fs, mod); Vva, _ = combined_vector(Xva, fs, mod)
    fmean, fstd = Vtr.mean(0), Vtr.std(0) + 1e-6
    Vtr_s = ((Vtr - fmean) / fstd).astype(np.float32); Vva_s = ((Vva - fmean) / fstd).astype(np.float32)
    Xr = _inorm(Xtr.astype(np.float32)); Xvr = _inorm(Xva.astype(np.float32))
    D = Vtr.shape[1]
    print(f"[{mod}-fusion] train={len(Xr)} val={len(Xvr)} feat_dim={D}", flush=True)

    model = FeatureFusionMultiHead(patch, patch, D, n_classes=4, max_tokens=max(16, 4000 // patch),
                                   ordinal_grade=True).to(DEVICE)
    qcrit = SORDLoss(4, tau=2.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(Xr); bs = 128; steps = args.epochs * math.ceil(n / bs)
    sched = _cosine_warmup(opt, int(0.05 * steps), steps); use_amp = DEVICE == "cuda"
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xr = torch.from_numpy(Xr[idx]).to(DEVICE); xf = torch.from_numpy(Vtr_s[idx]).to(DEVICE)
            yq = torch.from_numpy(ytr[idx]).long().to(DEVICE); ybin = (yq >= 2).long()
            opt.zero_grad(set_to_none=True)
            actx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with actx:
                out = model.forward_multitask(xr, xf)
                loss = qcrit(out["q"], yq) + 0.3 * F.cross_entropy(out["binary"], ybin)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        ql, bl = [], []
        for i in range(0, len(Xvr), 256):
            o = model.forward_multitask(torch.from_numpy(Xvr[i:i+256]).to(DEVICE),
                                        torch.from_numpy(Vva_s[i:i+256]).to(DEVICE))
            ql.append(o["q"].float().cpu().numpy()); bl.append(o["binary"].float().cpu().numpy())
    ql = np.concatenate(ql); bl = np.concatenate(bl); yb = (yva >= 2).astype(int)
    Tq, q0, q1 = guarded_T(ql, yva); Tb, b0, b1 = guarded_T(bl, yb)
    print(f"[{mod}-fusion] grade ECE {q0:.3f}->{q1:.3f} (T={Tq:.2f}) | usable ECE {b0:.3f}->{b1:.3f} (T={Tb:.2f})", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / f"{mod}_fusion.onnx"
    exp = FeatureFusionExport(model, fmean, fstd, temperature={"q": Tq, "binary": Tb}).to(DEVICE).eval()
    dr = torch.randn(2, 1, L, device=DEVICE); dfeat = torch.randn(2, D, device=DEVICE)
    torch.onnx.export(exp, (dr, dfeat), str(onnx_path), input_names=["x_raw", "x_feat"],
                      output_names=["q_logits", "bin_logits"],
                      dynamic_axes={k: {0: "batch"} for k in ["x_raw", "x_feat", "q_logits", "bin_logits"]},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    with torch.no_grad():
        tq, tb = exp(dr, dfeat)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    oq, ob = sess.run(None, {"x_raw": dr.cpu().numpy(), "x_feat": dfeat.cpu().numpy()})

    def _sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    prob_diff = float(np.abs(_sm(tq.cpu().numpy()) - _sm(oq)).max())
    argmax_ok = bool((_sm(tq.cpu().numpy()).argmax(1) == _sm(oq).argmax(1)).all())
    parity = prob_diff < 5e-3 and argmax_ok
    print(f"[{mod}-fusion] parity: grade-prob max-diff={prob_diff:.2e} argmax_ok={argmax_ok} -> {parity}", flush=True)

    QN = [QUALITY_NAMES[i] for i in range(4)]
    card = {
        "modality": mod, "L_m": L, "fs_hz": float(fs), "class_order": QN,
        "normalization": {"method": "none"},
        "training_data_hash": f"{mod}-combined-fusion-{Path(args.store).name}", "model_version": "v3-combined-fusion-ordlogit",
        "inputs": ["x_raw", "x_feat"],
        "feature_preprocessing": {"fn": "combined_vector", "modality": mod, "feat_names": names, "n_features": D,
                                  "note": "sqa_features.combined_vector (per-modality SQI pack ++ advanced dynamics/HOS); "
                                          "pure numpy (numpy.rfft only); graph z-scores x_raw and standardizes x_feat (baked)."},
        "heads": [{"name": "grade", "output_name": "q_logits", "kind": "ordinal", "activation": "softmax", "class_order": QN},
                  {"name": "usable", "output_name": "bin_logits", "kind": "binary", "activation": "softmax", "class_order": ["unusable", "usable"]}],
        "calibration": {"temperatures": {"grade": round(Tq, 4), "usable": round(Tb, 4)},
                        "grade_ece": [round(q0, 4), round(q1, 4)], "usable_ece": [round(b0, 4), round(b1, 4)]},
        "routing": "grade+usable <- raw trunk fused with host combined SQI+dynamics vector; grade = ordinal ordered-logit",
    }
    (out_dir / f"{mod}_fusion.model_card.json").write_text(json.dumps(card, indent=2))
    print(f"[{mod}-fusion] exported {onnx_path} parity={parity}", flush=True)
    if args.copy_to_app:
        import shutil
        # backup existing single-input model once
        appdir = Path("app/models")
        for suf in (".onnx", ".model_card.json"):
            cur = appdir / f"{mod}{suf}"; bak = appdir / f"{mod}_prev{suf}"
            if cur.exists() and not bak.exists():
                shutil.copy(cur, bak)
        shutil.copy(onnx_path, appdir / f"{mod}.onnx")
        shutil.copy(out_dir / f"{mod}_fusion.model_card.json", appdir / f"{mod}.model_card.json")
        print(f"  copied to app/models/{mod}.* (prev backed up)", flush=True)


if __name__ == "__main__":
    main()
