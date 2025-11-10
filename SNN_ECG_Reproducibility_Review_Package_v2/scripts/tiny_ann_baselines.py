# Text-only baselines for two lightweight ANNs: TinyCNN and LiteTransformer
# Evaluate on same subject-wise splits and report mean ± SD over seeds.
import argparse, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class ECGDataset(Dataset):
    def __init__(self, x_np, y_np): self.x, self.y = x_np, y_np
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], int(self.y[i])

def make_loader(d, split, bs, nw=0):
    X = np.load(os.path.join(d, f"X_{split}.npy")); y = np.load(os.path.join(d, f"y_{split}.npy"))
    return DataLoader(ECGDataset(X,y), batch_size=bs, shuffle=(split=='train'), num_workers=nw, pin_memory=False,
                      collate_fn=lambda b: (torch.stack([torch.tensor(x) for x,_ in b]).float(),
                                            torch.tensor([y for _,y in b]).long()))

class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super().__init__()
        self.pw1 = nn.Conv1d(in_ch, 16, 1)
        self.dw1 = nn.Conv1d(16, 16, 9, stride=2, padding=4, groups=16, bias=False)
        self.pw2 = nn.Conv1d(16, 32, 1)
        self.dw2 = nn.Conv1d(32, 32, 9, stride=2, padding=4, groups=32, bias=False)
        self.pw3 = nn.Conv1d(32, 64, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(64, 2)
    def forward(self, x):
        x = F.silu(self.pw1(x)); x = F.silu(self.dw1(x)); x = F.silu(self.pw2(x)); x = F.silu(self.dw2(x)); x = F.silu(self.pw3(x))
        return self.fc(self.gap(x).squeeze(-1))

class LiteTransformer(nn.Module):
    def __init__(self, in_ch=3, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Conv1d(in_ch, d_model, kernel_size=10, stride=10, padding=0, bias=True)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)
    def forward(self, x):
        x = self.embed(x).transpose(1,2); x = self.encoder(x); return self.head(x.mean(1))

@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval(); P=[]; Y=[]
    for xb,yb in loader:
        xb=xb.to(device); logits=model(xb); p=torch.softmax(logits,dim=-1)[:,1].cpu().numpy()
        P.append(p); Y.append(yb.numpy())
    p=np.concatenate(P); y=np.concatenate(Y); yhat=(p>=0.5).astype(int)
    acc=accuracy_score(y,yhat); prec,rec,f1,_=precision_recall_fscore_support(y,yhat,average='binary',zero_division=0)
    try: auroc=roc_auc_score(y,p)
    except: auroc=float('nan')
    try: auprc=average_precision_score(y,p)
    except: auprc=float('nan')
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auroc=auroc, auprc=auprc)

def train_one_epoch(model, loader, opt, device, class_weight=None):
    model.train(); ce=nn.CrossEntropyLoss(weight=class_weight.to(device) if class_weight is not None else None)
    for xb,yb in loader:
        xb, yb = xb.to(device), yb.to(device); opt.zero_grad(); loss=ce(model(xb), yb); loss.backward(); opt.step()

def run(model_name, loaders, device, lr=3e-3, max_epochs=50, patience=5, class_weight=None):
    model = TinyCNN() if model_name=='TinyCNN' else LiteTransformer()
    model.to(device); opt=torch.optim.Adam(model.parameters(), lr=lr)
    best, best_val, wait = {k:None for k in ['state','val']}, -1e9, 0
    for _ in range(max_epochs):
        train_one_epoch(model, loaders['train'], opt, device, class_weight)
        val = eval_metrics(model, loaders['val'], device)['auroc']; val = -1e9 if (val!=val) else val
        if val>best_val: best_val=val; best['state']=model.state_dict(); wait=0
        else:
            wait+=1; if wait>=patience: break
    model.load_state_dict(best['state']); return eval_metrics(model, loaders['test'], device)

def main():
    import argparse, numpy as np, torch, json, pandas as pd, os
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", default="ann_baselines_textonly_summary.csv")
    ap.add_argument("--class_weight_pos", type=float, default=1.0)
    args=ap.parse_args()
    device=torch.device(args.device); seeds=[int(s) for s in args.seeds.split(",")]
    def loader(split, bs): return make_loader(args.data, split, bs)
    y_tr = np.load(os.path.join(args.data, "y_train.npy"))
    pos_ratio = (y_tr==1).mean(); pos_w = args.class_weight_pos if args.class_weight_pos>0 else (1-pos_ratio)/max(pos_ratio,1e-6)
    class_weight = torch.tensor([1.0, float(pos_w)], dtype=torch.float32)
    loaders={"train":loader("train",32),"val":loader("val",64),"test":loader("test",64)}
    agg=[]; per=[]
    for name in ["TinyCNN","LiteTransformer"]:
        seed_metrics=[]
        for s in seeds:
            set_seed(s); m=run(name, loaders, device, class_weight=class_weight); m["seed"]=s; m["model"]=name; seed_metrics.append(m); per.append(m)
        row={"model":name}
        for k in ["accuracy","precision","recall","f1","auroc","auprc"]:
            import numpy as np
            vals=np.array([mm[k] for mm in seed_metrics], float); row[f"{k}_mean"]=float(np.nanmean(vals)); row[f"{k}_sd"]=float(np.nanstd(vals, ddof=1))
        agg.append(row)
    import pandas as pd
    pd.DataFrame(agg).to_csv(args.out_csv, index=False)
    with open(args.out_csv.replace(".csv","_per_seed.json"),"w") as f: json.dump(per, f, indent=2)
    print("Saved:", args.out_csv)
if __name__ == "__main__":
    main()