# ---------------------------------------------------------------------------
# GENERATED FILE — do not edit here.
# Verbatim copy of <monorepo>/scripts/<this name>, with one transform: the
# sys.path bootstrap points at <root> (this package's layout puts biosqa/ at
# the root) instead of <root>/src. Regenerate: python scripts/sync_from_src.py
# ---------------------------------------------------------------------------
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosqa.data.artifact_labels import to_multihot  # noqa: E402
from biosqa.data.harmonize import ARTIFACT_TYPES, QUALITY_NAMES  # noqa: E402
from biosqa.data.sqa_features import combined_vector  # noqa: E402
from biosqa.data.store import CANONICAL_FS, WINDOW_S, SegmentStore  # noqa: E402
from biosqa.eval.metrics import evaluate  # noqa: E402
from biosqa.export.to_onnx import sha256_file  # noqa: E402
from biosqa.models.fusion import FeatureFusionExport, FeatureFusionMultiHead  # noqa: E402
from biosqa.train.losses import SORDLoss  # noqa: E402
from biosqa.train.loop import _cosine_warmup  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402
from biosqa.xdomain import calibration as cal  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 32
K = len(ARTIFACT_TYPES)
# Grade + usable are learnable only on cohorts whose labels describe the signal per
# window (expert artifact annotation); proxy cohorts (condition-derived) are kept for
# the artifact-type head only. See manuscript Section "EEG ordinal grade is learnable
# on signal-descriptive labels".
EXPERT_GRADE = ["TUAR", "PhysioMotion"]


def clean_flag(V_s, y, groups):
    """Confident-learning: flag likely-mislabeled windows (cross-validated GBM whose
    confident prediction contradicts the label). Returns a boolean mask of flagged."""
    from sklearn.ensemble import HistGradientBoostingClassifier as GBM
    from sklearn.model_selection import GroupKFold
    P = np.zeros((len(y), 4))
    n_splits = int(min(5, np.unique(groups).size))
    if n_splits < 2:
        return np.zeros(len(y), bool)
    for tri, vai in GroupKFold(n_splits).split(V_s, y, groups):
        c = GBM(max_depth=4, max_iter=300, l2_regularization=1.0, random_state=0).fit(V_s[tri], y[tri])
        for j, cl in enumerate(c.classes_):
            P[vai, int(cl)] = c.predict_proba(V_s[vai])[:, j]
    th = np.array([P[y == k, k].mean() if (y == k).any() else 1.0 for k in range(4)])
    am = P.argmax(1)
    return (am != y) & (P[np.arange(len(y)), am] >= th[am])


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
    ap.add_argument("--store", default="data/store_v8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--type-weight", type=float, default=1.0)
    ap.add_argument("--copy-to-app", action="store_true")
    args = ap.parse_args()
    mod = "eeg"
    seed_everything(args.seed); rng = np.random.default_rng(args.seed)
    fs = CANONICAL_FS[mod]; L = int(round(WINDOW_S[mod] * fs))
    store = SegmentStore(args.store)

    Xtr, ytr, gtr, dtr = store.load_modality(mod, "train"); ytr = ytr.astype(np.int64)
    Xva, yva, _, dva = store.load_modality(mod, "val"); yva = yva.astype(np.int64)
    Xte, yte, _, dte = store.load_modality(mod, "test"); yte = yte.astype(np.int64)
    dtr, dva, dte, gtr = np.asarray(dtr), np.asarray(dva), np.asarray(dte), np.asarray(gtr)
    em_tr = np.isin(dtr, EXPERT_GRADE); em_va = np.isin(dva, EXPERT_GRADE); em_te = np.isin(dte, EXPERT_GRADE)
    Ttr, mtr = to_multihot(store.load_column(mod, "artifact_type", split="train"))
    Vtr, names = combined_vector(Xtr, fs, mod)
    Vva, _ = combined_vector(Xva, fs, mod)
    Vte, _ = combined_vector(Xte, fs, mod)
    fmean, fstd = Vtr.mean(0), Vtr.std(0) + 1e-6
    Vtr_s = ((Vtr - fmean) / fstd).astype(np.float32)
    Vva_s = ((Vva - fmean) / fstd).astype(np.float32)
    Vte_s = ((Vte - fmean) / fstd).astype(np.float32)
    Xr = _inorm(Xtr.astype(np.float32)); Xvr = _inorm(Xva.astype(np.float32))
    Xter = _inorm(Xte.astype(np.float32))
    D = Vtr.shape[1]
    # grade + usable are trained on the expert cohorts only, with confident-learning cleaning;
    # the type head still trains on all typed windows.
    flag = np.zeros(len(ytr), bool)
    if em_tr.any():
        f_e = clean_flag(Vtr_s[em_tr], ytr[em_tr], gtr[em_tr])
        flag[np.where(em_tr)[0][f_e]] = True
    gm_tr = em_tr & ~flag                      # grade/usable loss mask on the training set
    tpw = torch.tensor(np.clip(len(Xtr) - Ttr.sum(0), 1, None) / np.clip(Ttr.sum(0), 1, None),
                       dtype=torch.float32, device=DEVICE)
    print(f"[eeg-multihead] train={len(Xr)} expert-grade={int(gm_tr.sum())}/{int(em_tr.sum())} "
          f"(cleaned {int(flag.sum())}) feat_dim={D} typed={int(mtr.sum())}/{len(mtr)} type_classes={K}", flush=True)

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
            gmb = torch.from_numpy(gm_tr[idx]).to(DEVICE); emb = torch.from_numpy(em_tr[idx]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            actx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with actx:
                out = model.forward_multitask(xr, xf)
                loss = out["q"].sum() * 0.0                       # graph-connected zero seed
                if gmb.any():                                     # grade: expert cohorts, cleaned
                    loss = loss + qcrit(out["q"][gmb], yq[gmb])
                if emb.any():                                     # usable: expert cohorts
                    loss = loss + 0.3 * F.cross_entropy(out["binary"][emb], ybin[emb])
                if tm.any():                                      # type: all typed windows
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
    ql = np.concatenate(ql)[em_va]; bl = np.concatenate(bl)[em_va]
    yva_e = yva[em_va]; yb = (yva_e >= 2).astype(int)
    Tq, q0, q1 = guarded_T(ql, yva_e); Tb, b0, b1 = guarded_T(bl, yb)
    with torch.no_grad():
        qlt, blt = [], []
        for i in range(0, len(Xter), 256):
            o = model.forward_multitask(
                torch.from_numpy(Xter[i:i+256]).to(DEVICE),
                torch.from_numpy(Vte_s[i:i+256]).to(DEVICE),
            )
            qlt.append(o["q"].float().cpu().numpy())
            blt.append(o["binary"].float().cpu().numpy())
    qlt, blt = np.concatenate(qlt)[em_te], np.concatenate(blt)[em_te]
    yte_e = yte[em_te]; ybt = (yte_e >= 2).astype(int)
    qwk = evaluate(yte_e, qlt.argmax(1), labels=[0, 1, 2, 3])["cohen_kappa_quadratic"]
    qt0 = cal.expected_calibration_error(cal.apply_temperature(qlt, 1.0), yte_e)
    qt1 = cal.expected_calibration_error(cal.apply_temperature(qlt, Tq), yte_e)
    bt0 = cal.expected_calibration_error(cal.apply_temperature(blt, 1.0), ybt)
    bt1 = cal.expected_calibration_error(cal.apply_temperature(blt, Tb), ybt)
    print(f"[eeg-multihead] EXPERT-test grade QWK={qwk:.3f} (n={len(yte_e)}); calibration-fit(val) grade "
          f"{q0:.3f}->{q1:.3f}, usable {b0:.3f}->{b1:.3f}; independent-test grade {qt0:.3f}->{qt1:.3f}, "
          f"usable {bt0:.3f}->{bt1:.3f}", flush=True)

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
        "training_data_hash": "eeg-multihead-store_v8-expert-grade", "model_version": "v5-multihead-expert-grade-cleaned",
        # Hashed AFTER the export+parity run above, so this pins the exact graph shipped.
        # model_version is free text and cannot separate two same-shape graphs: app/dist's
        # eeg.onnx is a v4 model of identical size (2,298,439) and identical L_m/head contract
        # to this v5 — every structural check the app runs passes; only the digest differs.
        "onnx_sha256": sha256_file(onnx_path),
        "inputs": ["x_raw", "x_feat"],
        "feature_preprocessing": {"fn": "combined_vector", "modality": "eeg", "feat_names": names, "n_features": D,
                                  "note": "sqa_features.combined_vector (pack ++ advanced); graph z-scores x_raw and standardizes x_feat."},
        "heads": [{"name": "grade", "output_name": "q_logits", "kind": "ordinal", "activation": "softmax", "class_order": QN},
                  {"name": "usable", "output_name": "bin_logits", "kind": "binary", "activation": "softmax", "class_order": ["unusable", "usable"]},
                  {"name": "artifact", "output_name": "type_logits", "kind": "multilabel", "activation": "sigmoid", "class_order": list(ARTIFACT_TYPES), "threshold": 0.5}],
        "calibration": {"location": "onnx_graph",
                        "temperatures": {"grade": round(Tq, 4), "usable": round(Tb, 4)},
                        "fit_partition": "validation", "evaluation_partition": "test",
                        "grade_ece": [round(qt0, 4), round(qt1, 4)],
                        "usable_ece": [round(bt0, 4), round(bt1, 4)],
                        "optimization_ece": {"grade": [round(q0, 4), round(q1, 4)],
                                             "usable": [round(b0, 4), round(b1, 4)]}},
        "routing": "grade+usable+type <- raw trunk fused with host combined vector; grade = ordinal ordered-logit; type = multilabel",
        "grade_training": {"cohorts": EXPERT_GRADE, "expert_test_qwk": round(float(qwk), 4),
                           "note": ("grade+usable trained only on signal-descriptive expert cohorts "
                                    "(TUAR, PhysioMotion) with confident-learning label cleaning; the type "
                                    "head trains on all cohorts. Proxy/condition-derived cohorts lack a "
                                    "native ordinal quality scale.")},
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
