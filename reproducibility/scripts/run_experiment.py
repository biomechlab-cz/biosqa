"""Single experiment entrypoint (Plan 1 §4, §3.2).

    python scripts/run_experiment.py --experiment dummy_smoke
    python scripts/run_experiment.py --experiment dummy_smoke --set train.lr=5e-4 seed=1

Reads configs/base.yaml + configs/experiment/<name>.yaml (+ CLI dotlist), trains
via the single loop, evaluates with the FROZEN harness, optionally exports+parity-
checks ONNX, logs everything to MLflow, and appends a 5-line research_log entry.
Nothing is "done" until it is logged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# make `biosqa` importable whether or not the package is pip-installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosqa.eval.metrics import evaluate  # noqa: E402
from biosqa.models.model import build_model  # noqa: E402
from biosqa.train.loop import fit, predict  # noqa: E402
from biosqa.utils import paths  # noqa: E402
from biosqa.utils.config import config_hash, load_config, to_container  # noqa: E402
from biosqa.utils.seed import seed_everything  # noqa: E402


def build_loaders(cfg):
    """Return (train, val, test) DataLoaders + labels for the configured source."""
    modality = cfg.data.modality
    n_classes = cfg.model.n_classes
    if cfg.data.source == "dummy":
        from biosqa.data.synthetic import synthetic_datasets

        tr, va, te = synthetic_datasets(
            length=cfg.data.window_length, n_classes=n_classes, seed=cfg.seed
        )
    elif cfg.data.source == "cinc2011":
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset

        from biosqa.data.loaders import load_cinc2011

        X, y, _ = load_cinc2011(label_scheme=cfg.data.get("label_scheme", "native"))
        print(f"[data] CinC-2011 loaded X={X.shape} class_counts={np.bincount(y).tolist()}")
        # record-level stratified 70/15/15 (CinC has no subject ids; challenge convention)
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=cfg.seed)
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=cfg.seed)
        to = lambda A, b: TensorDataset(torch.from_numpy(A), torch.from_numpy(b))
        tr, va, te = to(Xtr, ytr), to(Xva, yva), to(Xte, yte)
    elif cfg.data.source == "store":
        from torch.utils.data import TensorDataset

        from biosqa.data.augment import AugmentedDataset, RandomAugment
        from biosqa.data.store import SegmentStore

        store = SegmentStore(cfg.data.store_dir)
        aug = RandomAugment() if cfg.data.get("augment", False) else None
        Xtr, ytr, _, _ = store.load_modality(modality, split="train")
        Xva, yva, _, _ = store.load_modality(modality, split="val")
        Xte, yte, _, _ = store.load_modality(modality, split="test")
        print(f"[data] store={cfg.data.store_dir} modality={modality} "
              f"train={Xtr.shape} val={Xva.shape} test={Xte.shape} "
              f"train_labels={np.bincount(ytr, minlength=n_classes).tolist()}")
        tr = AugmentedDataset(Xtr, ytr, aug)
        va = AugmentedDataset(Xva, yva, None)
        te = AugmentedDataset(Xte, yte, None)
    else:
        raise ValueError(f"unknown data.source '{cfg.data.source}'")

    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    mk = lambda ds, shuf: DataLoader(ds, batch_size=bs, shuffle=shuf, num_workers=nw, drop_last=False)
    return mk(tr, True), mk(va, False), mk(te, False), list(range(n_classes)), modality


def compute_class_weights(train_loader, n_classes, mode) -> torch.Tensor | None:
    if mode in (None, "null", False):
        return None
    if isinstance(mode, (list, tuple)):
        return torch.tensor(list(mode), dtype=torch.float32)
    if mode == "balanced":
        counts = np.zeros(n_classes)
        for batch in train_loader:
            y = np.asarray(batch[1])
            for c in range(n_classes):
                counts[c] += (y == c).sum()
        counts = np.clip(counts, 1, None)
        w = counts.sum() / (n_classes * counts)
        return torch.tensor(w, dtype=torch.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", "-e", default=None, help="configs/experiment/<name>.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. train.lr=1e-4 seed=1")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.experiment, overrides=args.set)
    seed_everything(cfg.seed)
    chash = config_hash(cfg)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, labels, modality = build_loaders(cfg)
    class_weights = compute_class_weights(train_loader, cfg.model.n_classes, cfg.train.get("class_weights"))
    model = build_model(cfg)

    def log_fn(epoch, m):
        print(f"  epoch {epoch:3d}  loss={m['train_loss']:.4f}  "
              f"val_macroF1={m['macro_f1']:.4f}  val_kappaQ={m['cohen_kappa_quadratic']:.4f}")

    print(f"[run] experiment={cfg.experiment_name} hash={chash} device={device} "
          f"backbone={cfg.model.backbone} modality={modality}")
    result = fit(
        model, train_loader, val_loader,
        modality=modality, n_classes=cfg.model.n_classes, labels=labels, device=device,
        epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        warmup_frac=cfg.train.warmup_frac, loss=cfg.train.loss, class_weights=class_weights,
        monitor=cfg.train.monitor, patience=cfg.train.patience, amp=cfg.train.amp, log_fn=log_fn,
    )

    # final held-out test with the frozen evaluator (decode with the TRAINED loss —
    # a CORN head outputs rank thresholds, so softmax-argmax here would be wrong)
    y_true, y_pred, y_prob = predict(model, test_loader, modality, device,
                                     loss=cfg.train.loss, n_classes=cfg.model.n_classes)
    test_metrics = evaluate(y_true, y_pred, y_prob, labels=labels)
    print(f"[test] macro_f1={test_metrics['macro_f1']:.4f} "
          f"kappa={test_metrics['cohen_kappa']:.4f} kappaQ={test_metrics['cohen_kappa_quadratic']:.4f} "
          f"bal_acc={test_metrics['balanced_accuracy']:.4f} auroc={test_metrics['auroc_ovr_macro']:.4f}")

    # ONNX export + parity gate
    export_verdict = None
    if cfg.export.enabled:
        from biosqa.export.to_onnx import export_and_verify

        paths.ensure_dirs()
        export_verdict = export_and_verify(
            model, modality, cfg.data.window_length, paths.MODELS,
            c_in=cfg.modalities[modality].c_in, quantize=cfg.export.quantize,
        )
        print(f"[onnx] passes_parity={export_verdict['passes_parity']} "
              f"rel_drift={export_verdict['fp32_parity']['max_rel_logit_diff']:.2e} "
              f"cpu_ms={export_verdict['fp32_latency_ms']:.3f}")

    # ---- tracking (non-fatal: a completed run must not be lost to a logging hiccup) ----
    if not args.no_mlflow and cfg.tracking.backend == "mlflow":
        try:
            import mlflow

            paths.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            # MLflow 3.x deprecated the file backend -> use a local SQLite DB.
            uri = cfg.tracking.uri or f"sqlite:///{(paths.RUNS_DIR / 'mlflow.db').as_posix()}"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(cfg.experiment_name)
            with mlflow.start_run(run_name=f"{cfg.experiment_name}-{chash}"):
                flat = _flatten(to_container(cfg))
                mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})
                mlflow.log_param("config_hash", chash)
                scalar = {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
                scalar.update({f"val_{k}": v for k, v in result.best_metrics.items() if isinstance(v, (int, float))})
                mlflow.log_metrics(scalar)
                if export_verdict is not None:
                    mlflow.log_param("onnx_exportable", export_verdict["passes_parity"])
                    mlflow.log_metric("onnx_rel_drift", export_verdict["fp32_parity"]["max_rel_logit_diff"])
                    mlflow.log_metric("onnx_cpu_latency_ms", export_verdict["fp32_latency_ms"])
                try:  # artifact store may be unconfigured; metrics/params already logged
                    mlflow.log_dict(to_container(cfg), "config.yaml")
                    mlflow.log_dict(test_metrics, "test_metrics.json")
                except Exception:
                    pass
            print(f"[mlflow] logged run -> {uri}")
        except Exception as e:
            print(f"[mlflow] logging skipped ({e!r})")

    _append_research_log(cfg, chash, test_metrics, export_verdict)
    return test_metrics


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _append_research_log(cfg, chash, test_metrics, export_verdict):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    exp = "n/a" if export_verdict is None else export_verdict["passes_parity"]
    entry = (
        f"\n- **{ts}** `{cfg.experiment_name}` ({chash}) — "
        f"backbone={cfg.model.backbone}, modality={cfg.data.modality}, source={cfg.data.source}. "
        f"test macro-F1={test_metrics['macro_f1']:.3f}, kappaQ={test_metrics['cohen_kappa_quadratic']:.3f}, "
        f"bal-acc={test_metrics['balanced_accuracy']:.3f}. onnx_parity={exp}.\n"
    )
    try:
        with open(paths.RESEARCH_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        pass


if __name__ == "__main__":
    main()
