"""Generate the repo `dummy_data/` folder: small synthetic, quality-VARYING WFDB recordings,
one per modality, whose channel is named with a canonical token (II / PLETH / Fp1 / EDA) so the
app's channel-name modality detection routes each to the right model.

    python scripts/make_dummy_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biosqa.io.loaders import detect_modality, open_recording  # noqa: E402
from biosqa.io.synth import write_test_recording  # noqa: E402

OUT = ROOT / "dummy_data"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    ok = True
    for modality, minutes in (("ecg", 3.0), ("ppg", 3.0), ("eeg", 3.0), ("eda", 3.0)):
        hea = write_test_recording(modality, dirpath=OUT, minutes=minutes)
        detected = detect_modality(open_recording(hea))
        match = detected == modality
        ok = ok and match
        print(f"{modality:>4}: {Path(hea).name:<22} detected={detected:<4} {'OK' if match else 'MISMATCH!'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
