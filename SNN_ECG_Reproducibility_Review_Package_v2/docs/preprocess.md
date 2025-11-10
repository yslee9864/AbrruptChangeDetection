# Preprocessing Pipeline (v2, concrete and deterministic)

1. **Resampling**: resample all ECG leads to **1000 Hz**.
2. **Filtering**: zero‑phase 4th‑order Butterworth **0.5–45 Hz** band‑pass; apply **50 Hz** notch (use 60 Hz if local mains is 60 Hz).
3. **Lead Selection**: use **I, V2, V5** leads.
4. **Windowing**: fixed windows of **2.5 s** with **50 % overlap** (stride = **1.25 s**).
5. **Normalization**: per‑lead **z‑score** using statistics computed **on the training set only**; apply the same mean/std to val/test (no leakage).
6. **Spike Encoding**: event threshold = **μ + kσ** with **k ∈ {1.5, 2.0, 2.5}**; choose **k** by validation AUROC and fix for the test run.
7. **Artifacts & Missing Samples**: drop windows with **> 20 %** missing/flatlined samples; interpolate shorter gaps linearly before filtering.
8. **Reproducibility**: the above parameters are fixed across all seeds and models; subject‑wise splits follow `config/splits.json` (deterministic hash policy).