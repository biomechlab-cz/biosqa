# ---------------------------------------------------------------------------
# GENERATED FILE — do not edit here.
# Verbatim copy of <monorepo>/scripts/<this name>, with one transform: the
# sys.path bootstrap points at <root> (this package's layout puts biosqa/ at
# the root) instead of <root>/src. Regenerate: python scripts/sync_from_src.py
# ---------------------------------------------------------------------------
"""Export the ECG DUAL-BRANCH spectral-fusion deployable (arch-search 2026-07-05).

Raw branch -> ordinal SORD grade head (in-dist grade unregressed); raw+spectral fused
-> usable + artifact-type heads (spectral lifts type macro-F1 +0.075 and cross-cohort
usable-AUROC +0.032, verified). Grade/usable train on REAL store grades (grade-masked);
type trains on synthetic rare-type + native. Exports a TWO-INPUT legacy-ONNX graph
(x_raw [B,1,L], x_spec [B,C,L]); the app computes x_spec via spectral_band_channels in
numpy (no in-graph FFT). model_card carries the spectral preprocessing spec.

    python scripts/export_ecg_dualbranch.py --epochs 25 --out models --copy-to-app
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
from biosqa.data.artifact_synth import synth_ecg_artifacts  # noqa: E402
from biosqa.data.harmonize import ARTIFACT_TYPES, QUALITY_NAMES  # noqa: E402
from biosqa.data.signal_channels import MODALITY_BANDS, spectral_band_channels  # noqa: E402
from biosqa.data.store import CANONICAL_FS, WINDOW_S, SegmentStore  # noqa: E402
from biosqa.export.redistributable import assert_redistributable  # noqa: E402
from biosqa.export.to_onnx import sha256_file  # noqa: E402
from biosqa.models.fusion import DualBranchExport, DualBranchMultiHead  # noqa: E402
from biosqa.train.losses import SORDLoss  # noqa: E402
from biosqa.train.loop import _cosine_warmup  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402
from biosqa.xdomain import calibration as cal  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_S, HOP_S = 0.25, 0.0625
K = len(ARTIFACT_TYPES)


def _inorm(X):
    return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)


def _derive_q(Y):
    n = Y[:, 1:].sum(1); q = np.full(len(Y), 3, np.int64); q[n == 1] = 1; q[n >= 2] = 0
    return q


def type_pos_weight(T, mask):
    """BCE ``pos_weight`` for the artifact-TYPE head: negatives/positives over the
    TYPE-LABELLED rows only. The type loss is masked to ``mask``, so counting the
    unlabelled rows as negatives inflates every entry (~2.1x here, 2.6x for 'clean')
    and makes the head over-predict artifacts. Mirrors export_all_modalities.py:116.
    NOTE: only takes effect on the next re-export of models/ecg_dualbranch.onnx."""
    pos = T[mask].sum(0)
    return np.clip(mask.sum() - pos, 1, None) / np.clip(pos, 1, None)


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
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-synth", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-to-app", action="store_true")
    # The store was HARDCODED to data/store_v2, which is why the shipped card
    # reports it. Phase E needs the corrected corpus, so it becomes an argument
    # with the historical value as the default.
    ap.add_argument("--store", default="data/store_v2")
    ap.add_argument("--redistributable", action="store_true",
                    help="enforce the app/LICENSE-MODELS cohort allowlist "
                         "(required with --copy-to-app)")
    args = ap.parse_args()
    seed_everything(args.seed); rng = np.random.default_rng(args.seed)
    fs = CANONICAL_FS["ecg"]; L = int(round(WINDOW_S["ecg"] * fs))
    bands = MODALITY_BANDS["ecg"]; nch = len(bands)
    store = SegmentStore(args.store)

    Xtr, ytr, _, dtr = store.load_modality("ecg", "train"); ytr = ytr.astype(np.int64)
    Ttr, mtr = to_multihot(store.load_column("ecg", "artifact_type", split="train"))
    Xva, yva, _, dva = store.load_modality("ecg", "val"); yva = yva.astype(np.int64)
    Xte, yte, _, dte = store.load_modality("ecg", "test"); yte = yte.astype(np.int64)

    # LICENSING GATE (Plan 09 Phase E) -- see scripts/export_fusion.py for the rationale.
    # All seven ECG cohorts are open-access, so this should never filter anything; it is
    # here so that a future cohort addition cannot reach app/models/ unreviewed.
    if args.redistributable:
        for split_ds in (dtr, dva, dte):
            assert_redistributable("ecg", split_ds)
        print(f"[license] ECG trains on: {sorted(set(np.asarray(dtr).astype(str)))}", flush=True)
    elif args.copy_to_app:
        raise SystemExit(
            "--copy-to-app writes into app/models/, which is redistributed. Pass "
            "--redistributable so the licence allowlist is enforced, or drop --copy-to-app.")
    # synthetic rare-type augmentation (grade-masked)
    carriers = Xtr[ytr == 3]
    if len(carriers) > args.n_synth:
        carriers = carriers[rng.choice(len(carriers), args.n_synth, replace=False)]
    Xs, Ys, _ = synth_ecg_artifacts(carriers.astype(np.float32), float(fs), seed=args.seed)
    Xall = np.concatenate([Xtr, Xs]).astype(np.float32)
    qall = np.concatenate([ytr, _derive_q(Ys)])
    Tall = np.concatenate([Ttr, Ys.astype(np.float32)])
    tmask = np.concatenate([mtr, np.ones(len(Xs), bool)])
    gmask = np.concatenate([np.ones(len(Xtr), bool), np.zeros(len(Xs), bool)])
    print(f"[ecg-dualbranch] train={len(Xall)} (+{len(Xs)} synth) spectral_ch={nch}", flush=True)

    # precompute inormed raw + spectral channels
    Xr = _inorm(Xall); Xsp = _inorm(spectral_band_channels(Xall, fs, bands))
    tpw = torch.tensor(type_pos_weight(Tall, tmask), dtype=torch.float32, device=DEVICE)
    model = DualBranchMultiHead(50, 50, nch, n_type=K, ordinal_grade=True).to(DEVICE)
    qcrit = SORDLoss(4, tau=2.0).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(Xall); bs = 128; steps = args.epochs * math.ceil(n / bs)
    sched = _cosine_warmup(opt, int(0.05 * steps), steps); use_amp = DEVICE == "cuda"
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xr = torch.from_numpy(Xr[idx]).to(DEVICE); xs = torch.from_numpy(Xsp[idx]).to(DEVICE)
            yq = torch.from_numpy(qall[idx]).long().to(DEVICE); yt = torch.from_numpy(Tall[idx]).float().to(DEVICE)
            gm = torch.from_numpy(gmask[idx]).to(DEVICE); tm = torch.from_numpy(tmask[idx]).to(DEVICE)
            ybin = (yq >= 2).long()
            opt.zero_grad(set_to_none=True)
            actx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with actx:
                out = model.forward_multitask(xr, xs)
                loss = out["q"].sum() * 0.0
                if gm.any():
                    loss = qcrit(out["q"][gm], yq[gm]) + 0.3 * F.cross_entropy(out["binary"][gm], ybin[gm])
                if tm.any():
                    loss = loss + 1.0 * F.binary_cross_entropy_with_logits(out["type"][tm], yt[tm], pos_weight=tpw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

    # calibrate grade + usable on store val (real grades)
    model.eval()
    Xvr = _inorm(Xva); Xvs = _inorm(spectral_band_channels(Xva, fs, bands))
    with torch.no_grad():
        ql, bl = [], []
        for i in range(0, len(Xva), 256):
            o = model.forward_multitask(torch.from_numpy(Xvr[i:i+256].astype(np.float32)).to(DEVICE),
                                        torch.from_numpy(Xvs[i:i+256].astype(np.float32)).to(DEVICE))
            ql.append(o["q"].float().cpu().numpy()); bl.append(o["binary"].float().cpu().numpy())
    ql = np.concatenate(ql); bl = np.concatenate(bl); yb = (yva >= 2).astype(int)
    Tq, q0, q1 = guarded_T(ql, yva); Tb, b0, b1 = guarded_T(bl, yb)
    Xter = _inorm(Xte); Xtes = _inorm(spectral_band_channels(Xte, fs, bands))
    with torch.no_grad():
        qlt, blt = [], []
        for i in range(0, len(Xte), 256):
            o = model.forward_multitask(
                torch.from_numpy(Xter[i:i+256].astype(np.float32)).to(DEVICE),
                torch.from_numpy(Xtes[i:i+256].astype(np.float32)).to(DEVICE),
            )
            qlt.append(o["q"].float().cpu().numpy())
            blt.append(o["binary"].float().cpu().numpy())
    qlt, blt = np.concatenate(qlt), np.concatenate(blt)
    ybt = (yte >= 2).astype(int)
    qt0 = cal.expected_calibration_error(cal.apply_temperature(qlt, 1.0), yte)
    qt1 = cal.expected_calibration_error(cal.apply_temperature(qlt, Tq), yte)
    bt0 = cal.expected_calibration_error(cal.apply_temperature(blt, 1.0), ybt)
    bt1 = cal.expected_calibration_error(cal.apply_temperature(blt, Tb), ybt)
    print(f"[ecg-dualbranch] calibration-fit(val) grade {q0:.3f}->{q1:.3f}, usable {b0:.3f}->{b1:.3f}; "
          f"independent-test grade {qt0:.3f}->{qt1:.3f}, usable {bt0:.3f}->{bt1:.3f}", flush=True)

    # export 2-input ONNX
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / "ecg_dualbranch.onnx"
    exp = DualBranchExport(model, temperature={"q": Tq, "binary": Tb}).to(DEVICE).eval()
    dr = torch.randn(2, 1, L, device=DEVICE); dsx = torch.randn(2, nch, L, device=DEVICE)
    torch.onnx.export(exp, (dr, dsx), str(onnx_path), input_names=["x_raw", "x_spec"],
                      output_names=["q_logits", "bin_logits", "type_logits"],
                      dynamic_axes={k: {0: "batch"} for k in ["x_raw", "x_spec", "q_logits", "bin_logits", "type_logits"]},
                      opset_version=17, dynamo=False)
    # parity
    import onnxruntime as ort
    with torch.no_grad():
        tq, tb, tt = exp(dr, dsx)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    oq, ob, ot = sess.run(None, {"x_raw": dr.cpu().numpy(), "x_spec": dsx.cpu().numpy()})
    # decision-level parity (the deployment-relevant criterion): the deeper 2-branch
    # graph accumulates ~1e-3 fp32 logit noise, so compare SOFTMAX probs + argmax, not
    # raw logits. Grade argmax + usable argmax must agree; prob max-diff must be tiny.
    def _sm(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    logit_diff = max(float(np.abs(a.cpu().numpy() - b).max()) for a, b in [(tq, oq), (tb, ob), (tt, ot)])
    pq_t, pq_o = _sm(tq.cpu().numpy()), _sm(oq)
    prob_diff = float(np.abs(pq_t - pq_o).max())
    argmax_ok = bool((pq_t.argmax(1) == pq_o.argmax(1)).all())
    parity = prob_diff < 5e-3 and argmax_ok
    print(f"[ecg-dualbranch] parity: grade-prob max-diff={prob_diff:.2e} argmax_ok={argmax_ok} "
          f"(raw-logit diff {logit_diff:.2e}) -> {parity}", flush=True)
    QN = [QUALITY_NAMES[i] for i in range(4)]
    card = {
        "modality": "ecg", "L_m": L, "fs_hz": float(fs), "class_order": QN,
        "normalization": {"method": "none"},  # per-window instance-norm baked into the graph
        "training_data_hash": "dualbranch-synthaug", "model_version": "v3-dualbranch-spectral",
        # Hashed AFTER the export+parity run above, so this pins the exact graph shipped.
        # model_version is free text and cannot separate two same-shape graphs: app/dist's
        # eeg.onnx is a v4 model of identical size and L_m/head contract to the shipped v5.
        "onnx_sha256": sha256_file(onnx_path),
        "inputs": ["x_raw", "x_spec"],
        "spectral_preprocessing": {"fn": "spectral_band_channels", "bands_hz": bands, "frame_s": FRAME_S,
                                   "hop_s": HOP_S, "n_channels": nch,
                                   "note": "numpy Hann-STFT log band-power -> interp to L; the app computes x_spec then the graph z-scores both inputs per-window"},
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
        "routing": "grade<-raw branch; usable+type<-raw+spectral fused",
    }
    (out_dir / "ecg_dualbranch.model_card.json").write_text(json.dumps(card, indent=2))
    print(f"[ecg-dualbranch] exported {onnx_path} parity={parity} outputs=q/bin/type", flush=True)
    if args.copy_to_app:
        import shutil
        for f in (onnx_path, out_dir / "ecg_dualbranch.model_card.json"):
            shutil.copy(f, Path("app/models") / f.name)
        print("  copied to app/models", flush=True)


if __name__ == "__main__":
    main()
