# dummy_data

Small **synthetic** biosignal recordings for trying the app out, open any of them with **Open recording…**.

Each is a single-channel WFDB file (`.hea` + `.dat`) with a few minutes of deliberately quality-varying
signal (clean segments interleaved with motion, baseline wander, muscle/ocular artifact, clipping, dropout).
The channel is named with a canonical token (`II`, `PLETH`, `Fp1`, `EDA`) so the app auto-detects the modality
and loads the matching model:

| file | modality | channel | model |
|---|---|---|---|
| `test_ecg_3min` | ECG | `II` | `ecg.onnx` |
| `test_ppg_3min` | PPG | `PLETH` | `ppg.onnx` |
| `test_eeg_3min` | EEG | `Fp1` | `eeg.onnx` |
| `test_eda_3min` | EDA | `EDA` | `eda.onnx` |

Not real physiological data, do not use for anything but exercising the UI. Regenerate with
`python scripts/make_dummy_data.py`.
