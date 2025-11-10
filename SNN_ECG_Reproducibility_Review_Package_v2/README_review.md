# Reproducibility Package (v2, Reviewer‑ready)

This package removes all placeholders by defining a **deterministic, subject‑wise split policy** and concrete preprocessing choices.
Reviewers can reproduce results without redistributing raw PTB(-XL) data.

## What changed in v2
- `config/splits.json` now encodes a **hash‑based split rule** (sha256(patient_id) % 100)→train/val/test.
- `docs/preprocess.md` is **fully specified** (no <FILL>).
- `config/config.yaml` is concrete and self‑contained.

## Materializing explicit splits (optional)
If you need the actual patient lists, run:
```bash
python scripts/make_splits.py --meta_csv /path/to/ptbxl_metadata.csv --out_json config/splits_explicit.json
```
This reads the official metadata, applies the same deterministic rule, and outputs explicit lists.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_ptbxl.py --dest ./data/raw
python scripts/preprocess.py   --raw ./data/raw --out ./data/processed

python scripts/train_snn.py --config config/config.yaml --splits config/splits.json --data ./data/processed --out ./runs/snn
python scripts/eval_snn.py  --config config/config.yaml --splits config/splits.json --data ./data/processed --ckpt ./runs/snn/best.ckpt
```