"""
Does AI4Bharat's self-supervised ST-GCN beat our model? A controlled answer.

    .venv-tf/bin/python train/eval_dpc.py

Reads  data/dataset_stgcn27.npz  +  data/openhands/x/raw_dpc/*.ckpt
Writes run/dpc_eval.json

The question
-----------
train_production.py ships a 1-D CNN warm-started from MS-ASL + WLASL and scores
64.8% on a held-out signer. AI4Bharat publish raw_dpc: an ST-GCN pretrained by
Dense Predictive Coding across six sign languages, with no labels involved. It is
the only downloadable thing resembling a sign-language foundation model, and the
honest way to settle whether it is better is to run it on our protocol.

Two arms per fold, identical in every respect but initialisation:

    scratch      the same ST-GCN, randomly initialised
    pretrained   encoder weights from raw_dpc

The control is not optional. Without it a win could not be attributed: ST-GCN is
a different architecture from our CNN, so a bare "ST-GCN beats CNN" result would
confound the architecture with the pretraining. This repo has made exactly that
mistake before -- ARCHITECTURE.md 9, where SL-GCN was first run at a different
parameter count than its baseline.

Leave-one-group-out over the three INCLUDE signer groups, matching the protocol
behind the 64.8%, so the numbers are directly comparable. Validation is carved
out of TRAIN; the test group is scored once.

Handicap worth stating: raw_dpc takes 2 channels, so this arm never sees depth,
while our 64.8% model does. That is a property of the released checkpoint, not a
choice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
# lstub lives next to this file; see its docstring

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_stgcn27.npz"
CKPT = ROOT / "data" / "openhands" / "x" / "raw_dpc"
OUT = ROOT / "run" / "dpc_eval.json"

SEED = 0
EPOCHS, BATCH, VAL_FRACTION = 40, 32, 0.15
# (in, out, stride) per block, read off the checkpoint's own shapes.
BLOCKS = [(2, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1),
          (64, 128, 2), (128, 128, 1), (128, 128, 1),
          (128, 256, 2), (256, 256, 1), (256, 256, 1)]


class GraphConv(nn.Module):
    """ConvTemporalGraphical: 1x1 conv into 3 partitions, then mix along the graph."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, cout * k, kernel_size=1)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k, kc // self.k, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, stride, dropout=0.0, residual=True):
        super().__init__()
        self.gcn = GraphConv(cin, cout)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif cin == cout and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x, A)) + res)


class STGCN(nn.Module):
    """The encoder raw_dpc was pretrained as, plus a fresh linear head."""
    def __init__(self, n_classes, n_joints=27, cin=2):
        super().__init__()
        self.register_buffer("A", torch.zeros(3, n_joints, n_joints))
        self.data_bn = nn.BatchNorm1d(n_joints * cin)
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(a, b, s, residual=(i != 0))
            for i, (a, b, s) in enumerate(BLOCKS)])
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(3, n_joints, n_joints)) for _ in BLOCKS])
        self.head = nn.Linear(BLOCKS[-1][1], n_classes)

    def forward(self, x):                      # x: (N, C, T, V)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for blk, imp in zip(self.st_gcn_networks, self.edge_importance):
            x = blk(x, self.A * imp)
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(n, -1)
        return self.head(x)


def load_encoder(model: STGCN) -> int:
    """Copy raw_dpc's conv_encoder into `model`. Returns tensors transferred."""
    import lstub
    lstub.install()
    p = sorted(CKPT.rglob("*.ckpt"))[0]
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    enc = {k[len("model.conv_encoder."):]: v
           for k, v in sd.items() if k.startswith("model.conv_encoder.")}
    own = model.state_dict()
    moved = {k: v for k, v in enc.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in enc if k not in moved]
    if missing:
        print(f"    ! {len(missing)} checkpoint tensors did not match, e.g. {missing[:3]}")
    own.update(moved)
    model.load_state_dict(own)
    return len(moved)


def run_fold(X, y, tr_m, te_m, n_classes, pretrained, rng, A):
    torch.manual_seed(SEED)
    model = STGCN(n_classes)
    model.A.copy_(A)
    moved = load_encoder(model) if pretrained else 0
    if pretrained:
        model.A.copy_(A)                       # load_state_dict overwrote the buffer

    tr_idx = np.flatnonzero(tr_m)
    rng.shuffle(tr_idx)
    cut = max(int((1 - VAL_FRACTION) * len(tr_idx)), 1)
    core, va = tr_idx[:cut], tr_idx[cut:]

    dev = torch.device("cpu")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.int64))
    best_acc, best_state, patience = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(core))
        for i in range(0, len(core), BATCH):
            b = core[perm[i:i + BATCH].numpy()]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xt[va[i:i + 128]]) for i in range(0, len(va), 128)])
            acc = (pv.argmax(1) == yt[va]).float().mean().item()
        sched.step(acc)
        if acc > best_acc:
            best_acc, best_state, patience = acc, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)

    model.eval()
    te = np.flatnonzero(te_m)
    with torch.no_grad():
        p = torch.cat([model(Xt[te[i:i + 128]]) for i in range(0, len(te), 128)])
    yte = yt[te]
    top1 = (p.argmax(1) == yte).float().mean().item()
    top5 = (p.topk(5, dim=1).indices == yte[:, None]).any(1).float().mean().item()
    return top1, top5, moved, ep + 1


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA.relative_to(ROOT)} — run train/preprocess_stgcn27.py")
        return 1
    d = np.load(DATA, allow_pickle=True)
    X, y, signer, corpus = d["X"], d["y"], d["signer"], d["corpus"]
    n_classes = len(d["labels"])
    rng = np.random.default_rng(SEED)

    import lstub; lstub.install()
    ck = torch.load(sorted(CKPT.rglob("*.ckpt"))[0], map_location="cpu",
                    weights_only=False)["state_dict"]
    A = ck["model.conv_encoder.A"].clone()
    print(f"{len(X)} clips | {n_classes} classes | X {X.shape}")
    print(f"graph A from checkpoint: {tuple(A.shape)}\n")

    results = []
    for g in (0, 1, 2):                        # INCLUDE groups; matches the 64.8%
        te = (signer == g) & (corpus == 0)
        tr = ~te
        for pre in (False, True):
            t1, t5, moved, ep = run_fold(X, y, tr, te, n_classes, pre, rng, A)
            tag = "pretrained" if pre else "scratch   "
            print(f"  group {g}  {tag}  top-1 {t1*100:5.1f}%  top-5 {t5*100:5.1f}%"
                  f"   ({ep} epochs, {moved} tensors)", flush=True)
            results.append({"group": int(g), "pretrained": pre,
                            "top1": t1, "top5": t5, "epochs": ep})

    sc = [r for r in results if not r["pretrained"]]
    pt = [r for r in results if r["pretrained"]]
    m_sc = float(np.mean([r["top1"] for r in sc]))
    m_pt = float(np.mean([r["top1"] for r in pt]))
    print("\n" + "=" * 60)
    print(f"  ST-GCN scratch      mean top-1 {m_sc*100:5.1f}%")
    print(f"  ST-GCN + raw_dpc    mean top-1 {m_pt*100:5.1f}%   ({(m_pt-m_sc)*100:+.1f})")
    print(f"  our shipped CNN     mean top-1  64.8%   (train_production.py)")
    print("=" * 60)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"folds": results, "scratch_mean": m_sc,
                               "pretrained_mean": m_pt, "cnn_baseline": 0.648},
                              indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
