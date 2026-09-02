# ---------------------------------------------------------------------------
# GENERATED FILE — do not edit here.
# Verbatim copy of <monorepo>/scripts/<this name>, with one transform: the
# sys.path bootstrap points at <root> (this package's layout puts biosqa/ at
# the root) instead of <root>/src. Regenerate: python scripts/sync_from_src.py
# ---------------------------------------------------------------------------
"""Train + CALIBRATE + export a deployable multi-head model for every modality.

For each of {ecg, ppg, eeg, eda}: train the 3-level multi-task model on the store,
fit temperature scaling on the val split (Guo 2017) for the grade + usable heads
(baked into the export graph so the app's softmax is calibrated), compute the
metrology conformal abstention threshold (Vovk), then export multi-head ONNX +
INT8 + v2 model card (with calibration/ood metadata) and verify parity + latency.

    python scripts/export_all_modalities.py --store data/store_v2 --epochs 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosqa.data.artifact_labels import to_multihot  # noqa: E402
from biosqa.data.harmonize import ARTIFACT_TYPES, QUALITY_NAMES  # noqa: E402
from biosqa.data.store import CANONICAL_FS, WINDOW_S, SegmentStore  # noqa: E402
from biosqa.export.to_onnx import export_and_verify_multihead  # noqa: E402
from biosqa.models.model import BioSQAModel  # noqa: E402
from biosqa.train.multitask import fit_multitask, make_multitask_dataset  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402
from biosqa.xdomain import calibration as cal  # noqa: E402
from biosqa.xdomain import conformal_sqa as conf  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = {"ecg": 50, "ppg": 32, "eeg": 32, "eda": 8}


def load_split(store, mod, split):
    X, y, _, _ = store.load_modality(mod, split=split)
    Yt, mask = to_multihot(store.load_column(mod, "artifact_type", split=split))
    return X.astype(np.float32), y.astype(np.int64), Yt, mask


def class_weights(y, n=4):
    c = np.bincount(y, minlength=n).astype(float)
    return torch.tensor(c.sum() / (n * np.clip(c, 1, None)), dtype=torch.float32)


@torch.no_grad()
def head_logits(model, X, mod, bs=256):
    """Raw q + binary logits on X, with the export's instance-norm applied."""
    model.eval(); qs, bs_ = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
        out = model.forward_multitask(x, mod)
        qs.append(out["q"].float().cpu().numpy())
        if out["binary"] is not None:
            bs_.append(out["binary"].float().cpu().numpy())
    return np.concatenate(qs), (np.concatenate(bs_) if bs_ else None)


# Grade-head loss per modality (arch-search campaign 2026-07-05): SORD soft ordinal
# targets (tau=2) beat class-weighted CE on ECG/EEG/EDA (QWK Δ+0.03..+0.06, AUROC
# +0.11 on ECG) at ZERO export cost. PPG's Q2 is near-absent (effectively 3-level)
# so SORD is neutral-negative there -> keep CE.
Q_LOSS = {"ecg": "sord", "ppg": "ce", "eeg": "sord", "eda": "sord"}

# Ordered-logit (proportional-odds) grade head per modality (arch-search campaign
# 2026-07-05): deployed on ECG/EEG where it beats nominal softmax cross-cohort
# (LODO QWK +0.05..+0.08, usable-AUROC +0.02..+0.03) AND passes the in-distribution
# no-regression gate. NOT on EDA (proxy labels, few cohorts -> regressed in-dist) or
# PPG (effectively 3-level). Emits log-probs as q_logits so the ONNX/app contract is
# unchanged.
ORDINAL_GRADE = {"ecg": True, "ppg": False, "eeg": True, "eda": False}
# Synthetic artifact-TYPE augmentation per modality (arch-search 2026-07-05): the
# export previously trained the type head on SPARSE native labels only -> weak. ECG has
# a calibrated-SNR + procedural synth pipeline covering all 9 types (incl. the rare
# burst/clipping/dropout/motion) -> type macro-F1 0.333->0.509. Synthetic windows are
# GRADE-MASKED (proxy grades never touch the real grade head).
SYNTH_TYPE_AUG = {"ecg": True, "ppg": False, "eeg": False, "eda": False}


def _derive_q_proxy(Y):
    n = Y[:, 1:].sum(1); q = np.full(len(Y), 3, np.int64); q[n == 1] = 1; q[n >= 2] = 0
    return q


def export_modality(store, mod, epochs, out_dir, seed, q_loss=None, n_synth=6000):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    q_loss = q_loss or Q_LOSS.get(mod, "ce")
    ordinal_grade = ORDINAL_GRADE.get(mod, False)
    fs = CANONICAL_FS[mod]; L = int(round(WINDOW_S[mod] * fs)); K = len(ARTIFACT_TYPES)
    Xtr, ytr, Ttr, mtr = load_split(store, mod, "train")
    Xva, yva, Tva, mva = load_split(store, mod, "val")
    if len(Xtr) < 50 or len(Xva) < 20:
        print(f"[{mod}] SKIP — too few windows (train={len(Xtr)} val={len(Xva)})"); return None

    # grade_mask: real store windows train grade+usable; synthetic windows train type only.
    gm_tr = np.ones(len(Xtr), dtype=bool)
    if SYNTH_TYPE_AUG.get(mod):
        from biosqa.data.artifact_synth import synth_ecg_artifacts
        carriers = Xtr[ytr == 3]
        if len(carriers) > n_synth:
            carriers = carriers[rng.choice(len(carriers), n_synth, replace=False)]
        if len(carriers) >= 50:
            Xs, Ys, _ = synth_ecg_artifacts(carriers.astype(np.float32), float(fs), seed=seed)
            Xtr = np.concatenate([Xtr, Xs]); ytr = np.concatenate([ytr, _derive_q_proxy(Ys)])
            Ttr = np.concatenate([Ttr, Ys.astype(np.float32)])
            mtr = np.concatenate([mtr, np.ones(len(Xs), bool)])
            gm_tr = np.concatenate([gm_tr, np.zeros(len(Xs), bool)])
            print(f"[{mod}] +{len(Xs)} synthetic type-aug windows (grade-masked); "
                  f"type positives/type: {dict((ARTIFACT_TYPES[i], int(Ttr[mtr][:, i].sum())) for i in range(K) if Ttr[mtr][:, i].sum() > 0)}", flush=True)

    pos = Ttr[mtr].sum(0) if mtr.any() else np.ones(K)
    tpw = torch.tensor(np.clip(mtr.sum() - pos, 1, None) / np.clip(pos, 1, None), dtype=torch.float32)
    model = BioSQAModel(modalities={mod: {"patch_len": PATCH[mod], "stride": PATCH[mod], "c_in": 1,
                                          "max_tokens": max(16, L // PATCH[mod])}},
                        backbone="cnn1d", d_model=128, n_classes=4, n_glitch=K, n_binary=2,
                        ordinal_grade=ordinal_grade, backbone_cfg={"n_blocks": 4}, pool="mean").to(DEVICE)
    tr = make_multitask_dataset(Xtr, ytr, Ttr, mtr, grade_mask=gm_tr); va = make_multitask_dataset(Xva, yva, Tva, mva)
    fit_multitask(model, DataLoader(tr, batch_size=128, shuffle=True), DataLoader(va, batch_size=256),
                  modality=mod, n_classes=4, artifact_types=list(ARTIFACT_TYPES), device=DEVICE,
                  # weights over the GRADE-SUPERVISED rows only: the synthetic windows
                  # appended above carry proxy grades and are masked out of the grade
                  # loss, so folding them into the statistic mis-weights the real grades
                  # (ECG: Q1 under-weighted 3.6x). Only takes effect on re-export.
                  epochs=epochs, q_loss=q_loss, q_loss_tau=2.0, class_weights=class_weights(ytr[gm_tr]),
                  type_pos_weight=tpw, bin_weight=0.5, type_weight=0.5, monitor="macro_f1",
                  patience=8, amp=(DEVICE == "cuda"))
    print(f"[{mod}] grade-head loss = {q_loss} | ordinal-cutpoint head = {ordinal_grade}", flush=True)

    # --- calibrate on val (temperature scaling, DO-NO-HARM: keep T only if it
    # lowers val ECE; NLL-optimal T isn't always ECE-optimal) ---
    ql, bl = head_logits(model, Xva, mod)
    yb = (yva >= 2).astype(int)

    def guarded_T(logits, labels):
        # DO-NO-HARM ECE minimisation over T. SORD soft targets make the grade head
        # UNDER-confident (soft targets bleed mass to neighbours), so the ECE-optimal T
        # is often <1 (sharpening) which the NLL fit alone can miss -> grid-search ECE
        # over candidates {1, NLL-fit, 0.5..3.0} and keep the ECE-min (never worse than T=1).
        e0 = cal.expected_calibration_error(cal.apply_temperature(logits, 1.0), labels)
        cands = [1.0, cal.fit_temperature(logits, labels)] + list(np.linspace(0.2, 3.0, 57))
        best_T, best_e = 1.0, e0
        for T in cands:
            e = cal.expected_calibration_error(cal.apply_temperature(logits, T), labels)
            if e < best_e - 1e-6:
                best_T, best_e = float(T), e
        return best_T, e0, best_e, ("applied" if best_T != 1.0 else "rejected")

    T_q, ece0, ece1, q_status = guarded_T(ql, yva)
    T_b, be0, be1, b_status = guarded_T(bl, yb) if bl is not None else (1.0, 0.0, 0.0, "none")
    q_thr = conf.calibrate_threshold(cal.apply_temperature(ql, T_q), yva, alpha=0.1)

    # T is a 59-candidate ECE grid minimised on val — the SAME split used for early
    # stopping — so the val-fit ECE is a fit-set minimum (~1.7x optimistic on ECG).
    # Report the untouched TEST partition in the card, labelled, and keep the val
    # numbers under `optimization_ece`. Matches export_ecg_dualbranch/export_fusion.
    Xte, yte, _, _ = load_split(store, mod, "test")
    eval_part = "test" if len(Xte) else "validation"
    if len(Xte):
        qlt, blt = head_logits(model, Xte, mod)
        ybt = (yte >= 2).astype(int)
        qt0 = cal.expected_calibration_error(cal.apply_temperature(qlt, 1.0), yte)
        qt1 = cal.expected_calibration_error(cal.apply_temperature(qlt, T_q), yte)
        bt0, bt1 = (cal.expected_calibration_error(cal.apply_temperature(blt, 1.0), ybt),
                    cal.expected_calibration_error(cal.apply_temperature(blt, T_b), ybt)) if blt is not None else (0.0, 0.0)
    else:  # no test partition for this modality -> be explicit that the card is fit-set
        qt0, qt1, bt0, bt1 = ece0, ece1, be0, be1
    calib = {"temperatures": {"grade": round(T_q, 4), "usable": round(T_b, 4)},
             "fit_partition": "validation", "evaluation_partition": eval_part,
             "grade_ece": [round(qt0, 4), round(qt1, 4)],
             "usable_ece": [round(bt0, 4), round(bt1, 4)],
             "optimization_ece": {"grade": [round(ece0, 4), round(ece1, 4)],
                                  "usable": [round(be0, 4), round(be1, 4)]},
             "grade_temp": q_status, "usable_temp": b_status,
             "method": "temperature_scaling(do-no-harm)"}
    ood = {"method": "conformal_aps", "grade_nonconformity_threshold": round(float(q_thr), 4),
           "alpha": 0.1, "fit_partition": "validation"}

    v = export_and_verify_multihead(
        model, mod, L, float(fs), out_dir, class_order=[QUALITY_NAMES[i] for i in range(4)],
        artifact_class_order=list(ARTIFACT_TYPES), temperature={"q": T_q, "binary": T_b},
        calibration=calib, ood=ood, model_version="v2-calibrated", quantize=False)
    print(f"[{mod}] outputs={v['outputs']} parity={v['passes_parity']} fp32={v.get('fp32_latency_ms', float('nan')):.2f}ms | "
          f"grade T={T_q:.2f}({q_status}) fit(val) ECE {ece0:.3f}->{ece1:.3f}, {eval_part} {qt0:.3f}->{qt1:.3f} | "
          f"usable T={T_b:.2f}({b_status}) fit(val) ECE {be0:.3f}->{be1:.3f}, {eval_part} {bt0:.3f}->{bt1:.3f} | conf_thr={q_thr:.3f}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/store_v2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out", default="models")
    ap.add_argument("--modalities", nargs="+", default=["ecg", "ppg", "eeg", "eda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-to-app", action="store_true")
    args = ap.parse_args()
    store = SegmentStore(args.store)
    results = {}
    for mod in args.modalities:
        results[mod] = export_modality(store, mod, args.epochs, args.out, args.seed)
    print("\n=== SUMMARY ===")
    for mod, v in results.items():
        if v is None:
            print(f"  {mod}: skipped"); continue
        c = v["model_card"]["calibration"]
        print(f"  {mod}: parity={v['passes_parity']} outputs={len(v['outputs'])} "
              f"grade ECE({c['evaluation_partition']}) {c['grade_ece'][0]:.3f}->{c['grade_ece'][1]:.3f}")
    if args.copy_to_app:
        import shutil
        app = Path("app/models")
        for mod in args.modalities:
            for ext in (".onnx", ".model_card.json", ".int8.onnx"):
                src = Path(args.out) / f"{mod}{ext}"
                if src.exists():
                    shutil.copy(src, app / src.name)
        print(f"  copied models to {app}")


if __name__ == "__main__":
    main()
