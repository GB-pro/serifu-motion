"""Train the Serifu Motion speech->emo temporal conv net and export it for the browser.
Usage: python train.py <feat_dir> <out_json> --mode offline|realtime [--epochs N]

Outputs per frame: [K emo channels (z-scored)] [6 emotion logits] [aux: accHead, accBrow, intensity]
The regression channels drive lip sync and the "natural" style; emotion logits and aux signals are the
"what / when" reading that the animation interpretation layer (app/core/animate.js) turns into acting.
"""
import argparse, json, base64, random, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
from dataset import load_all, actor_baselines, apply_baseline, add_aux, NAMES, EMOS, AUX, FEAT_DIM, CHANNELS

SIL = np.zeros(FEAT_DIM, np.float32); SIL[0] = -4.0
TEST_ACTORS = set(map(int, os.environ['TEST_ACTORS'].split(','))) if os.environ.get('TEST_ACTORS') else set(range(1083, 1092))
VAL_ACTORS = set(range(1076, 1083)) if not os.environ.get('QUICK') else {1003}
K, NE, NA = len(NAMES), len(EMOS), len(AUX)


class TCN(nn.Module):
    def __init__(self, din, dout, C=64, dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16), causal=False, drop=0.0):
        super().__init__()
        self.causal, self.dil = causal, dil
        self.inp = nn.Conv1d(din, C, 1)
        self.convs = nn.ModuleList([nn.Conv1d(C, C, 3, dilation=d) for d in dil])
        self.mix = nn.ModuleList([nn.Conv1d(C, C, 1) for _ in dil])
        self.out = nn.Conv1d(C, dout, 1)
        self.drop = nn.Dropout1d(drop)

    def forward(self, x):  # x: B,D,T
        h = self.inp(x)
        for conv, mix, d in zip(self.convs, self.mix, self.dil):
            p = (2 * d, 0) if self.causal else (d, d)
            z = conv(Fn.pad(h, p))
            h = h + mix(self.drop(Fn.gelu(z, approximate='tanh')))
        return self.out(Fn.gelu(h, approximate='tanh'))


def make_seq(clips, rng, n_clips=6, aug=True):
    X, Y, M, E, A, AM, S = [], [], [], [], [], [], []

    def gap():
        g = rng.randint(8, 30)
        X.append(np.repeat(SIL[None], g, 0)); Y.append(np.zeros((g, K), np.float32)); M.append(np.zeros(g, np.float32))
        E.append(np.full(g, -1)); A.append(np.zeros((g, NA), np.float32)); AM.append(np.zeros((g, NA), np.float32)); S.append(np.full(g, -1))
    for si, c in enumerate(rng.sample(clips, min(n_clips, len(clips)))):
        gap()
        x = c['X'].copy()
        if aug:
            x[:, 0] += rng.uniform(-0.4, 0.4)
            x[:, 2] = np.where(x[:, 1] > 0, x[:, 2] + rng.uniform(-0.4, 0.4), 0)
            x[:, 8:] *= rng.uniform(0.8, 1.2)
            if rng.random() < 0.5:  # drop one or two cepstral bands for the whole clip
                for _ in range(rng.randint(1, 2)): x[:, 8 + rng.randrange(12)] = 0
            x += np.random.normal(0, 0.06, x.shape).astype(np.float32)
        n = len(x)
        X.append(x); Y.append(c['Yz']); M.append(np.ones(n, np.float32)); E.append(np.full(n, c['emo']))
        A.append(c['A']); AM.append(c['AM']); S.append(np.full(n, si))
    gap()
    return [np.concatenate(v) for v in (X, Y, M, E, A, AM, S)]


def batch(by_actor, rng, B, aug=True, n_clips=6):
    actors = list(by_actor)
    seqs = [make_seq(by_actor[rng.choice(actors)], rng, n_clips=n_clips, aug=aug) for _ in range(B)]
    T = max(len(s[0]) for s in seqs)
    X = np.repeat(SIL[None, None], B, 0).repeat(T, 1)
    Y = np.zeros((B, T, K), np.float32); M = np.zeros((B, T), np.float32); E = np.full((B, T), -1)
    A = np.zeros((B, T, NA), np.float32); AM = np.zeros((B, T, NA), np.float32); S = np.full((B, T), -1)
    for i, (x, y, m, e, a, am, s) in enumerate(seqs):
        n = len(x); X[i, :n] = x; Y[i, :n] = y; M[i, :n] = m; E[i, :n] = e; A[i, :n] = a; AM[i, :n] = am
        S[i, :n] = np.where(s >= 0, s + i * n_clips, -1)
    t = torch.tensor
    return dict(X=t(X).permute(0, 2, 1), Y=t(Y).permute(0, 2, 1), M=t(M), E=t(E), A=t(A).permute(0, 2, 1), AM=t(AM).permute(0, 2, 1), S=t(S), nseg=B * n_clips)


def shift_for_causal(b, la):
    """Realtime model: output at frame t describes frame t-la (uses la frames of lookahead)."""
    if not la: return b
    b = dict(b)
    for k in ('Y', 'A', 'AM'): b[k] = Fn.pad(b[k], (la, 0))
    b['M'] = Fn.pad(b['M'], (la, 0)); b['E'] = Fn.pad(b['E'], (la, 0), value=-1); b['S'] = Fn.pad(b['S'], (la, 0), value=-1)
    X = Fn.pad(b['X'], (0, la)); X[:, 0, -la:] = -4.0; b['X'] = X
    return b


def loss_fn(out, b, smooth=0.1):
    pred, logit, aux = out[:, :K], out[:, K:K + NE], out[:, K + NE:]
    Y, M, E, A, AM, S = b['Y'], b['M'], b['E'], b['A'], b['AM'], b['S']
    m = M[:, None]
    mse = (((pred - Y) ** 2) * m).sum() / (m.sum() * K + 1e-6)
    dp, dy = pred[:, :, 1:] - pred[:, :, :-1], Y[:, :, 1:] - Y[:, :, :-1]
    mv = m[:, :, 1:] * m[:, :, :-1]
    vel = (((dp - dy) ** 2) * mv).sum() / (mv.sum() * K + 1e-6)
    mu_p = (pred * m).sum((0, 2)) / m.sum(); mu_y = (Y * m).sum((0, 2)) / m.sum()
    sd_p = (((pred - mu_p[None, :, None]) ** 2) * m).sum((0, 2)) / m.sum()
    sd_y = (((Y - mu_y[None, :, None]) ** 2) * m).sum((0, 2)) / m.sum()
    stdl = ((sd_p.sqrt() - sd_y.sqrt()) ** 2).mean()
    lf = logit.permute(0, 2, 1).reshape(-1, NE); ef = E.reshape(-1).long(); sf = S.reshape(-1).long()
    ce = Fn.cross_entropy(lf, ef, ignore_index=-1, label_smoothing=smooth)
    # clip-level emotion: mean logits over each clip
    ok = sf >= 0
    seg_sum = torch.zeros(b['nseg'], NE).index_add_(0, sf[ok], lf[ok]); seg_n = torch.zeros(b['nseg']).index_add_(0, sf[ok], torch.ones(int(ok.sum())))
    seg_lab = torch.full((b['nseg'],), -1, dtype=torch.long); seg_lab[sf[ok]] = ef[ok]
    have = seg_n > 0
    cec = Fn.cross_entropy(seg_sum[have] / seg_n[have, None], seg_lab[have], label_smoothing=smooth)
    acc = (seg_sum[have].argmax(1) == seg_lab[have]).float().mean()
    auxl = (((aux - A) ** 2) * AM).sum() / (AM.sum() + 1e-6)
    total = mse + 0.5 * vel + 0.3 * stdl + 0.15 * ce + 0.25 * cec + 0.3 * auxl
    return total, dict(mse=mse.item(), vel=vel.item(), std=stdl.item(), ce=ce.item(), cec=cec.item(), acc=acc.item(), aux=auxl.item())


def predict_clip(model, c, la):
    x = np.concatenate([np.repeat(SIL[None], 20, 0), c['X'], np.repeat(SIL[None], 20 + la, 0)])
    with torch.no_grad():
        o = model(torch.tensor(x).T[None])[0].T.numpy()
    o = o[20 + la: 20 + la + len(c['X'])]
    return o[:, :K], o[:, K:K + NE], o[:, K + NE:]


def compute_stats(model, train, val, mu, sd, la):
    """Data statistics used by the app's natural-style stochastic layer and by the animation layer."""
    n_au = len(CHANNELS)
    model.eval()
    prof = []
    allY = {e: np.concatenate([c['Yn'] for c in train if c['emo'] == e]) for e in range(NE)}
    neu = allY[0].mean(0)
    for e in range(NE):
        prof.append((allY[e].mean(0) - neu).tolist())
    P, G, AX, AG = [], [], [], []
    for c in val:
        p, _, ax = predict_clip(model, c, la)
        P.append(p * sd + mu); G.append(c['Yn']); AX.append(ax); AG.append(c['A'])
    P, G, AX, AG = np.concatenate(P), np.concatenate(G), np.concatenate(AX), np.concatenate(AG)
    gain = np.clip(G.std(0) / (P.std(0) + 1e-6), 1.0, 3.0)
    Pg = P.mean(0) + (P - P.mean(0)) * gain
    resid = G - Pg
    rstd = resid.std(0); r1 = np.array([np.corrcoef(resid[1:, k], resid[:-1, k])[0, 1] if resid[:, k].std() > 1e-6 else 0 for k in range(K)])
    bi = NAMES.index('AU45'); blinks = 0; frames = 0; dur = []
    for c in train:
        b = c['Y'][:, bi]; frames += len(b)
        on = np.flatnonzero((b[1:] > 0.45) & (b[:-1] <= 0.45)); off = np.flatnonzero((b[1:] <= 0.45) & (b[:-1] > 0.45))
        blinks += len(on)
        for o in on:
            f = off[off > o]
            if len(f): dur.append(f[0] - o)
    gx, gy = NAMES.index('gazeX'), NAMES.index('gazeY'); amps = []; nsac = 0; gfr = 0
    for c in train:
        g = c['Yn'][:, [gx, gy]]; v = np.linalg.norm(np.diff(g, axis=0), axis=1); gfr += len(v)
        pk = np.flatnonzero((v[1:-1] > 0.08) & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))
        nsac += len(pk); amps += list(v[pk + 1])
    aux_r = [float(np.corrcoef(AX[:, j], AG[:, j])[0, 1]) for j in range(2)]
    return dict(emo_profiles=prof, gain=gain.tolist(), resid_std=rstd.tolist(), resid_ar1=np.nan_to_num(r1).tolist(),
                blink_per_min=float(blinks / (frames / 30) * 60), blink_frames=float(np.median(dur)) if dur else 6.0,
                saccade_per_sec=float(nsac / (gfr / 30)), saccade_amp=float(np.median(amps)) if amps else 0.1,
                gt_std=G.std(0).tolist(), gt_mean=G.mean(0).tolist(), aux_val_r=aux_r)


def export(model, path, meta):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    W = {k: dict(shape=list(v.shape), b64=base64.b64encode(v.tobytes()).decode()) for k, v in sd.items()}
    json.dump(dict(meta=meta, weights=W), open(path, 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('feat_dir'); ap.add_argument('out')
    ap.add_argument('--mode', default='offline'); ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--steps', type=int, default=150); ap.add_argument('--C', type=int, default=64)
    ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', '4'))); torch.manual_seed(a.seed); rng = random.Random(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    clips = load_all(a.feat_dir)
    base, gneu = actor_baselines(clips)
    apply_baseline(clips, base, gneu); add_aux(clips)
    train = [c for c in clips if c['actor'] not in TEST_ACTORS | VAL_ACTORS]
    val = [c for c in clips if c['actor'] in VAL_ACTORS]
    Yall = np.concatenate([c['Yn'] for c in train])
    mu, sd = Yall.mean(0), Yall.std(0) + 1e-4
    for c in clips:
        c['Yz'] = ((c['Yn'] - mu) / sd).astype(np.float32)
    print(f'clips {len(clips)} train {len(train)} val {len(val)} load {time.time()-t0:.0f}s', flush=True)
    by_tr, by_va = {}, {}
    for c in train: by_tr.setdefault(c['actor'], []).append(c)
    for c in val: by_va.setdefault(c['actor'], []).append(c)
    causal = a.mode == 'realtime'
    la = 4 if causal else 0
    model = TCN(FEAT_DIM, K + NE + NA, C=a.C, causal=causal, drop=a.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * a.steps, pct_start=0.15)
    best, best_state = 1e9, None
    vrng = random.Random(1)
    vb = [shift_for_causal(batch(by_va, vrng, 16, aug=False), la) for _ in range(8)]
    for ep in range(a.epochs):
        model.train()
        for st in range(a.steps):
            b = shift_for_causal(batch(by_tr, rng, 16), la)
            l, _ = loss_fn(model(b['X']), b)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval(); vl = []; parts = []
        with torch.no_grad():
            for b in vb:
                l, p = loss_fn(model(b['X']), b, smooth=0.0); vl.append(l.item()); parts.append(p)
        v = float(np.mean(vl))
        if v < best:
            best = v; best_state = {k: t.clone() for k, t in model.state_dict().items()}
        pm = {k: round(float(np.mean([p[k] for p in parts])), 3) for k in parts[0]}
        print(f'ep {ep} val {v:.4f} {pm} {time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(best_state)
    extra = compute_stats(model, train, val, mu, sd, la)
    meta = dict(mode=a.mode, lookahead=la, names=NAMES, emos=EMOS, aux=AUX, feat_dim=FEAT_DIM, C=a.C,
                dil=list(model.dil), causal=causal, y_mean=mu.tolist(), y_std=sd.tolist(),
                neutral=gneu.tolist(), n_au=len(CHANNELS), trained_clips=len(train), val_loss=best,
                dataset='CREMA-D (ODbL) — 91 actors, numeric features only', **extra)
    export(model, a.out, meta)
    print('saved', a.out, 'best', best, 'aux r', extra['aux_val_r'])


if __name__ == '__main__':
    main()
