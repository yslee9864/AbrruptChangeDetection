import argparse, pathlib, json, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True, help="Path to raw PTB(-XL) data")
    ap.add_argument("--out", type=str, required=True, help="Output directory for processed NumPy arrays")
    args = ap.parse_args()
    raw = pathlib.Path(args.raw); out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[INFO] Placeholder preprocessing stub.")
    print("[TODO] Implement steps as specified in docs/preprocess.md:")
    print("  - resample, band-pass, notch, leads, windowing, z-score(train stats), spike encoding (mu+k*sigma).")
    print("[OUT ] Save arrays: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")

    # Example shapes (dummy): (N, C, T)
    # np.save(out / "X_train.npy", np.random.randn(10, 3, 2500).astype(np.float32))
    # np.save(out / "y_train.npy", np.random.randint(0, 2, size=(10,)).astype(np.int64))

if __name__ == "__main__":
    main()