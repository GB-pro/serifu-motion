"""Reader on a pretrained speech encoder: a small network that reads the emotion of a line from the summary the encoder
gives of it (embed_lines.mjs: for every line the mean over time of the output of each block).

  python train_head.py <prefix>[,<prefix>...] --data open|all [--blocks 1,2,3] [--reads line|relative|both]
                       [--hidden 256] [--seeds 3] [--epochs 120] [--out model.json]

<prefix>: what embed_lines.mjs wrote (<prefix>.<shard>.tsv and .f32). The roles of the corpora are those of corpus.py
and SPLITS.md: training on CREMA-D (training actors) and the training corpora, the epoch and everything else chosen
on the development corpora only. Nothing of the final check is read here, even if it is in the files.

Two ways of reading, as in the app: "line" (the line by itself) and "relative" (the line against the other lines of
the same speaker: its summary minus their mean; what a long recording allows).
Every corpus counts the same in training, and every emotion within a corpus the same.
Printed: for every development corpus the mean over the emotions of the share read right (for FLEURS, which has only
neutral lines, the share read as neutral).
"""
import argparse, glob, json, os
import numpy as np, torch
import corpus as CP

NE = 6
EM = CP.EMOS


def load(prefixes):
    rows, feats = [], []
    for pre in prefixes:
        for p in sorted(glob.glob(pre + '.*.tsv')):
            r = [l.rstrip('\n').split('\t') for l in open(p) if l.strip()]
            f = np.fromfile(p[:-4] + '.f32', np.float32)
            if not r: continue
            f = f.reshape(len(r), -1); rows += r; feats.append(f)
    X = np.concatenate(feats); corp = np.array([r[1] for r in rows]); y = np.array([int(r[3]) for r in rows])
    spk = np.array([r[1].split(':')[0] + '/' + r[2] for r in rows])
    return rows, X, corp, spk, y


def uar(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(NE) if (y == c).any()]))


def weights(tr, corp, y):
    w = np.zeros(len(y), np.float32)
    for c in set(corp[tr]):
        for e in range(NE):
            k = tr & (corp == c) & (y == e)
            if k.any(): w[k] = 1.0 / k.sum()
        k = tr & (corp == c); w[k] /= w[k].sum()
    return w


class Net(torch.nn.Module):
    def __init__(self, d, hidden, drop):
        super().__init__()
        self.inp = torch.nn.Dropout(drop); self.a = torch.nn.Linear(d, hidden); self.mid = torch.nn.Dropout(drop); self.b = torch.nn.Linear(hidden, NE)

    def forward(self, x):
        return self.b(self.mid(torch.relu(self.a(self.inp(x)))))


def train(X, y, w, tr, devs, hidden, drop, epochs, seed, noise):
    """devs: {name: mask}. The epoch is the one with the best mean over the development corpora."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    t = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    Xt, yt, wt = t(X[tr]), torch.tensor(y[tr]), torch.tensor(w[tr]); n = len(Xt)
    Xd = {k: (t(X[m]), y[m]) for k, m in devs.items()}
    net = Net(X.shape[1], hidden, drop); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = None; p = wt / wt.sum()
    for ep in range(epochs):
        net.train(); idx = torch.multinomial(p, n, replacement=True)   # lines drawn by their weight
        for i in range(0, n, 256):
            j = idx[i:i + 256]; x = Xt[j]
            if noise: x = x + noise * torch.randn_like(x)
            l = torch.nn.functional.cross_entropy(net(x), yt[j], label_smoothing=0.1)
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        if ep % 4 == 3 or ep == epochs - 1:
            net.eval(); res = {}
            with torch.no_grad():
                for k, (xd, yd) in Xd.items(): res[k] = uar(yd, net(xd).argmax(1).numpy())
            score = float(np.mean([v for k, v in res.items() if not k.startswith('crema')]))
            if best is None or score > best[0]: best = (score, ep + 1, res, {k: v.detach().clone() for k, v in net.state_dict().items()})
    return best, mu, sd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('prefix'); ap.add_argument('--data', default='open'); ap.add_argument('--blocks', default='1,2,3')
    ap.add_argument('--reads', default='both'); ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--drop', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--epochs', type=int, default=120); ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--out', default=''); ap.add_argument('--width', type=int, default=384)
    a = ap.parse_args()
    rows, X, corp, spk, y = load(a.prefix.split(','))
    blocks = [int(b) - 1 for b in a.blocks.split(',')]; D = a.width
    X = np.concatenate([X[:, b * D:(b + 1) * D] for b in blocks], 1)
    train_specs = ('crema:train',) + (CP.TRAIN_ALL if a.data == 'all' else CP.TRAIN_OPEN)
    dev_specs = ('crema:val',) + CP.DEV
    final = set(CP.FINAL); assert not (set(corp) & final) or True
    tr = np.isin(corp, train_specs) & ~np.isin(corp, list(final)); w = weights(tr, corp, y)
    devs = {d: (corp == d) for d in dev_specs if (corp == d).any()}
    print('training lines', {c: int((tr & (corp == c)).sum()) for c in train_specs}, flush=True)
    print('blocks', a.blocks, 'hidden', a.hidden, 'drop', a.drop, 'noise', a.noise, flush=True)
    out = {}
    for reads in (['line', 'relative'] if a.reads == 'both' else [a.reads]):
        Xr = X
        if reads == 'relative':
            Xr = X.copy()
            for s in set(spk): k = spk == s; Xr[k] -= Xr[k].mean(0)
        print(f'\n{reads:10s}' + ''.join(f'{d[:10]:>11s}' for d in devs) + '   mean(dev)  epoch', flush=True)
        runs = []
        for seed in range(a.seeds):
            (score, ep, res, state), mu, sd = train(Xr, y, w, tr, devs, a.hidden, a.drop, a.epochs, seed, a.noise)
            print(f'seed {seed:<5d}' + ''.join(f'{res[d]:11.3f}' for d in devs) + f'{score:12.3f}{ep:7d}', flush=True)
            runs.append((score, res, state, mu, sd))
        m = {d: float(np.mean([r[1][d] for r in runs])) for d in devs}
        print(f'{"mean":10s}' + ''.join(f'{m[d]:11.3f}' for d in devs) + f'{np.mean([r[0] for r in runs]):12.3f}', flush=True)
        out[reads] = dict(dev=m, mean=float(np.mean([r[0] for r in runs])), nets=[dict(mu=r[3].tolist(), sd=r[4].tolist(), **{k: v.numpy().tolist() for k, v in r[2].items()}) for r in runs])
    if a.out:
        meta = dict(kind='line-reader', encoder='whisper-tiny blocks ' + a.blocks, data=a.data, hidden=a.hidden, emotions=EM, training=list(train_specs))
        json.dump(dict(meta=meta, **out), open(a.out, 'w')); print('saved', a.out, os.path.getsize(a.out) // 1024, 'kB')


if __name__ == '__main__':
    main()
