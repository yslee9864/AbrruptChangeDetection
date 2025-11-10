import argparse, hashlib, json, pandas as pd, pathlib

def bucket(patient_id: int, modulus: int = 100) -> int:
    s = str(int(patient_id)).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h, 16) % modulus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True, help="Path to PTB(-XL) metadata CSV containing a patient_id column")
    ap.add_argument("--out_json", required=True, help="Output JSON with explicit train/val/test patient lists")
    ap.add_argument("--modulus", type=int, default=100)
    ap.add_argument("--train_range", type=str, default="0-69")
    ap.add_argument("--val_range", type=str, default="70-79")
    ap.add_argument("--test_range", type=str, default="80-99")
    args = ap.parse_args()

    def parse_range(s):
        a,b = s.split("-"); return int(a), int(b)

    tr_a,tr_b = parse_range(args.train_range)
    va_a,va_b = parse_range(args.val_range)
    te_a,te_b = parse_range(args.test_range)

    df = pd.read_csv(args.meta_csv)
    if "patient_id" not in df.columns:
        # Common alternative names
        for alt in ["patientid","Patient","subject_id","subject"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "patient_id"}); break
    assert "patient_id" in df.columns, "No patient_id column found."

    uniq = sorted(set(int(x) for x in df["patient_id"].tolist()))
    train, val, test = [], [], []
    for pid in uniq:
        r = bucket(pid, args.modulus)
        if tr_a <= r <= tr_b: train.append(pid)
        elif va_a <= r <= va_b: val.append(pid)
        elif te_a <= r <= te_b: test.append(pid)

    out = {"train_patients": train, "val_patients": val, "test_patients": test,
           "policy": {"hash":"sha256","modulus":args.modulus,
                      "train_range":[tr_a,tr_b], "val_range":[va_a,va_b], "test_range":[te_a,te_b]}}
    pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out_json}; counts train/val/test = {len(train)}/{len(val)}/{len(test)}")

if __name__ == "__main__":
    main()