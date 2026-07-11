"""Export the EEG 3-HEAD deployable: grade + usable + artifact-TYPE (2026-07-06).

Banks the phantom type-anchor gain (ds004784 lifted EEG type macro-F1 0.19->0.25). Trains
FeatureFusionMultiHead with a multilabel artifact-type head on store_v5 EEG (4 cohorts:
TUAR + PhysioMotion + motion_eeg + phantom), fusing the combined SQI+dynamics vector; grade
uses ordinal-SORD, usable CE, type masked pos-weighted BCE (only windows carrying native
type tokens contribute). Exports a TWO-INPUT / THREE-OUTPUT legacy-ONNX (x_raw, x_feat ->
q_logits, bin_logits, type_logits); the app already renders the artifact head (chips).

    python scripts/export_eeg_multihead.py --epochs 30 --copy-to-app
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

from biosqa.data.artifact_labels import to_multihot  # noqa: E402
from biosqa.data.harmonize import ARTIFACT_TYPES, QUALITY_NAMES  # noqa: E402
from biosqa.data.sqa_features import combined_vector  # noqa: E402
from biosqa.data.store import CANONICAL_FS, WINDOW_S, SegmentStore  # noqa: E402
from biosqa.models.fusion import FeatureFusionExport, FeatureFusionMultiHead  # noqa: E402
from biosqa.train.losses import SORDLoss  # noqa: E402
from biosqa.train.loop import _cosine_warmup  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402
from biosqa.xdomain import calibration as cal  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 32
K = len(ARTIFACT_TYPES)


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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models")
    ap.add_argument("--store", default="data/store_v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--type-weight", type=float, default=1.0)
    ap.add_argument("--copy-to-app", action="store_true")
    args = ap.parse_args()
    mod = "eeg"
    seed_everything(args.seed); rng = np.random.default_rng(args.seed)
    fs = CANONICAL_FS[mod]; L = int(round(WINDOW_S[mod] * fs))
    store = SegmentStore(args.store)

    Xtr, ytr, _, _ = store.load_modality(mod, "train"); ytr = ytr.astype(np.int64)
    Xva, yva, _, _ = store.load_modality(mod, "val"); yva = yva.astype(np.int64)
    Ttr, mtr = to_multihot(store.load_column(mod, "artifact_type", split="train"))
    Vtr, names = combined_vector(Xtr, fs, mod); Vva, _ = combined_vector(Xva, fs, mod)
    fmean, fstd = Vtr.mean(0), Vtr.std(0) + 1e-6
    Vtr_s = ((Vtr - fmean) / fstd).astype(np.float32); Vva_s = ((Vva - fmean) / fstd).astype(np.float32)
    Xr = _inorm(Xtr.astype(np.float32)); Xvr = _inorm(Xva.astype(np.float32))
    D = Vtr.shape[1]
    tpw = torch.tensor(np.clip(len(Xtr) - Ttr.sum(0), 1, None) / np.clip(Ttr.sum(0), 1, None),
                       dtype=torch.float32, device=DEVICE)
    print(f"[eeg-multihead] train={len(Xr)} feat_dim={D} typed={int(mtr.sum())}/{len(mtr)} type_classes={K}", flush=True)

    model = FeatureFusionMultiHead(PATCH, PATCH, D, n_classes=4, n_type=K,
                                   max_tokens=max(16, 4000 // PATCH), ordinal_grade=True).to(DEVICE)
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
            yt = torch.from_numpy(Ttr[idx]).float().to(DEVICE); tm = torch.from_numpy(mtr[idx]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            actx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with actx:
                out = model.forward_multitask(xr, xf)
                loss = qcrit(out["q"], yq) + 0.3 * F.cross_entropy(out["binary"], ybin)
                if tm.any():
                    loss = loss + args.type_weight * F.binary_cross_entropy_with_logits(
                        out["type"][tm], yt[tm], pos_weight=tpw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

    # calibrate grade + usable on store val
    model.eval()
    with torch.no_grad():
        ql, bl = [], []
        for i in range(0, len(Xvr), 256):
            o = model.forward_multitask(torch.from_numpy(Xvr[i:i+256]).to(DEVICE),
                                        torch.from_numpy(Vva_s[i:i+256]).to(DEVICE))
            ql.append(o["q"].float().cpu().numpy()); bl.append(o["binary"].float().cpu().numpy())
    ql = np.concatenate(ql); bl = np.concatenate(bl); yb = (yva >= 2).astype(int)
    Tq, q0, q1 = guarded_T(ql, yva); Tb, b0, b1 = guarded_T(bl, yb)
    print(f"[eeg-multihead] grade ECE {q0:.3f}->{q1:.3f} (T={Tq:.2f}) | usable ECE {b0:.3f}->{b1:.3f} (T={Tb:.2f})", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / "eeg_multihead.onnx"
    exp = FeatureFusionExport(model, fmean, fstd, temperature={"q": Tq, "binary": Tb}).to(DEVICE).eval()
    dr = torch.randn(2, 1, L, device=DEVICE); dfeat = torch.randn(2, D, device=DEVICE)
    torch.onnx.export(exp, (dr, dfeat), str(onnx_path), input_names=["x_raw", "x_feat"],
                      output_names=["q_logits", "bin_logits", "type_logits"],
                      dynamic_axes={k: {0: "batch"} for k in ["x_raw", "x_feat", "q_logits", "bin_logits", "type_logits"]},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    with torch.no_grad():
        tq, tb, tt = exp(dr, dfeat)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    oq, ob, ot = sess.run(None, {"x_raw": dr.cpu().numpy(), "x_feat": dfeat.cpu().numpy()})

    def _sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    prob_diff = float(np.abs(_sm(tq.cpu().numpy()) - _sm(oq)).max())
    type_diff = float(np.abs(tt.cpu().numpy() - ot).max())
    argmax_ok = bool((_sm(tq.cpu().numpy()).argmax(1) == _sm(oq).argmax(1)).all())
    parity = prob_diff < 5e-3 and argmax_ok
    print(f"[eeg-multihead] parity: grade-prob diff={prob_diff:.2e} type-logit diff={type_diff:.2e} -> {parity}", flush=True)

    QN = [QUALITY_NAMES[i] for i in range(4)]
    card = {
        "modality": "eeg", "L_m": L, "fs_hz": float(fs), "class_order": QN,
        "normalization": {"method": "none"},
        "training_data_hash": "eeg-multihead-store_v5", "model_version": "v4-multihead-grade-usable-type",
        "inputs": ["x_raw", "x_feat"],
        "feature_preprocessing": {"fn": "combined_vector", "modality": "eeg", "feat_names": names, "n_features": D,
                                  "note": "sqa_features.combined_vector (pack ++ advanced); graph z-scores x_raw and standardizes x_feat."},
        "heads": [{"name": "grade", "output_name": "q_logits", "kind": "ordinal", "activation": "softmax", "class_order": QN},
                  {"name": "usable", "output_name": "bin_logits", "kind": "binary", "activation": "softmax", "class_order": ["unusable", "usable"]},
                  {"name": "artifact", "output_name": "type_logits", "kind": "multilabel", "activation": "sigmoid", "class_order": list(ARTIFACT_TYPES), "threshold": 0.5}],
        "calibration": {"temperatures": {"grade": round(Tq, 4), "usable": round(Tb, 4)},
                        "grade_ece": [round(q0, 4), round(q1, 4)], "usable_ece": [round(b0, 4), round(b1, 4)]},
        "routing": "grade+usable+type <- raw trunk fused with host combined vector; grade = ordinal ordered-logit; type = multilabel",
    }
    (out_dir / "eeg_multihead.model_card.json").write_text(json.dumps(card, indent=2))
    print(f"[eeg-multihead] exported {onnx_path} parity={parity} heads=grade/usable/type", flush=True)
    if args.copy_to_app:
        import shutil
        appdir = Path("app/models")
        for suf in (".onnx", ".model_card.json"):
            cur = appdir / f"eeg{suf}"; bak = appdir / f"eeg_gu{suf}"  # grade+usable-only backup
            if cur.exists() and not bak.exists():
                shutil.copy(cur, bak)
        shutil.copy(onnx_path, appdir / "eeg.onnx")
        shutil.copy(out_dir / "eeg_multihead.model_card.json", appdir / "eeg.model_card.json")
        print("  copied to app/models/eeg.* (grade+usable model backed up as eeg_gu.*)", flush=True)


if __name__ == "__main__":
    main()
