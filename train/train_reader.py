"""Readers: networks that only read the emotion of a line, trained on CREMA-D and on other corpora (corpus.py, SPLITS.md).

The motion network (train.py) is trained on CREMA-D alone, because only CREMA-D has faces. Its reading turned out to be
tied to the 12 sentences and to the rooms of that corpus. A reader has the same shape and the same input, so the app can
put it on the committee (app/core/generator.js), but it is trained on many corpora and gets no motion targets.

  python train_reader.py <crema feat> <out.json> --data open|all [--mix a,b,c,d] [--seed N] [--epochs N] [--steps N]

Every training sequence is a simulated recording, normalised the way the app would normalise it (as in train.py):
long (24 to 72 lines of a speaker), a few lines, a single line, lines in one emotion. Two things are varied on top,
so that the reading cannot lean on them: the level between the words (lowered by up to 50 dB, or filled with room
noise taken from CREMA-D) and the length of a line (long lines are cut).
The epoch is chosen on the development corpora (never on those of the final check).
"""
import argparse, glob, json, os, random, time
import numpy as np, torch
import torch.nn.functional as Fn
import train as T, simnorm as SN, corpus as CP

NE, FEAT_DIM = T.NE, T.FEAT_DIM
MAX_LINE = 240          # frames; longer lines are cut at a random place (8 s)
DEV_LIMIT = 240         # lines per development corpus used for choosing the epoch


def crema_speakers(feat_dir, actors):
    by = {}
    for a in sorted(actors):
        clips = []
        for p in sorted(glob.glob(os.path.join(feat_dir, f'{a}_*.raw.f32'))):
            nm = os.path.basename(p)[:-8]; clips.append(dict(name=nm, emo=CP.EMOS.index(nm.split('_')[2]), R=SN.load_raw(feat_dir, nm), speaker=str(a), corpus='CREMA-D'))
        if clips: by[('CREMA-D', str(a))] = clips
    return by


def make_speakers(by):
    """{(corpus, speaker): lines} -> {key: dict(clips, by_emo, alone)}; lines without speaker identity are pooled (alone)."""
    out = {}; pool = {}
    for key, clips in by.items():
        if clips[0].get('alone'): pool.setdefault(key[0], []).extend(clips); continue
        A = dict(clips=clips, by_emo={}, alone=False)
        for c in clips: A['by_emo'].setdefault(c['emo'], []).append(c)
        out[key] = A
    for corpus, clips in pool.items():
        A = dict(clips=clips, by_emo={}, alone=True)
        for c in clips: A['by_emo'].setdefault(c['emo'], []).append(c)
        out[(corpus, '*')] = A
    return out


def noise_bank(speakers, n=40):
    """Room-noise frames of CREMA-D (level relative to the speech level in column 0), in their original order."""
    out = []
    for key in sorted(k for k in speakers if k[0] == 'CREMA-D')[:n]:
        clips = speakers[key]['clips']; st = SN.speaker_stats([c['R'] for c in clips])
        for c in clips:
            R = c['R'].astype(np.float32).copy(); rel = R[:, 0] - st['eSpeech']; ns = (R[:, 1] <= 0) & (rel < -15); R[:, 0] = rel; out.append(R[ns])
    return np.concatenate(out)


def vary_floor(raws, rng, bank):
    """One recording: change what is between the words. Returns new raw frames (the same change for all its lines)."""
    r = rng.random()
    if r < 0.45: return raws
    es = SN.speaker_stats(raws)['eSpeech']
    if r < 0.8:   # quieter between the words: a studio, a gate, noise suppression
        d = rng.uniform(5, 50); out = []
        for R in raws:
            R = R.copy(); ns = (R[:, 1] <= 0) & (R[:, 0] - es < -15); R[ns, 0] -= d; out.append(R)
        return out
    lev = rng.uniform(-28, -14); B = bank; out = []   # room noise at `lev` dB under the speech level wherever the line is quieter
    med = -19.3
    for R in raws:
        R = R.copy(); mask = R[:, 0] < es + lev; i = 0; n = len(R)
        while i < n:
            if not mask[i]: i += 1; continue
            j = i
            while j < n and mask[j]: j += 1
            k = rng.randrange(0, len(B) - (j - i)); seg = B[k:k + j - i].copy(); seg[:, 0] += es + lev - med; seg[:, 1] = 0; R[i:j] = seg; i = j
        out.append(R)
    return out


def pick_balanced(A, rng, n):
    emos = sorted(A['by_emo']); return [rng.choice(A['by_emo'][rng.choice(emos)]) for _ in range(n)]


def recording(A, rng, n_clips, mix):
    """-> [(line, lines of the recording it is normalised with)]: the simulated recordings of train.py."""
    r = rng.random()
    if A['alone'] or (mix[0] + mix[1] <= r < mix[0] + mix[1] + mix[2]):
        return [(c, [c]) for c in pick_balanced(A, rng, n_clips)]
    if r < mix[0]: rec = pick_balanced(A, rng, rng.randint(24, 72))
    elif r < mix[0] + mix[1]: rec = pick_balanced(A, rng, rng.randint(2, 12))
    else: pool = A['by_emo'][rng.choice(sorted(A['by_emo']))]; rec = [rng.choice(pool) for _ in range(rng.randint(2, 14))]
    return [(c, rec) for c in rec[:n_clips]]


def make_seq(A, rng, mix, bank, n_clips=6, aug=True):
    X, E, S = [], [], []
    rec = recording(A, rng, n_clips, mix); cache = {}

    def gap():
        n = rng.randint(8, 30); X.append(np.repeat(T.SIL[None], n, 0)); E.append(np.full(n, -1)); S.append(np.full(n, -1))
    for si, (c, group_) in enumerate(rec):
        gid = id(group_)
        if gid not in cache:
            raws = [g['R'] for g in group_]
            if aug: raws = vary_floor(raws, rng, bank)
            cache[gid] = (SN.speaker_stats(raws), {id(g): r for g, r in zip(group_, raws)})
        st, rmap = cache[gid]; R = rmap[id(c)]
        x = SN.normalize(R, st)
        if len(x) > MAX_LINE: o = rng.randrange(0, len(x) - MAX_LINE + 1); x = x[o:o + MAX_LINE]
        gap()
        if aug:
            x = x.copy(); x[:, 0] += rng.uniform(-0.4, 0.4)
            x[:, 2] = np.where(x[:, 1] > 0, x[:, 2] + rng.uniform(-0.4, 0.4), 0)
            x[:, 8:] *= rng.uniform(0.8, 1.2)
            if rng.random() < 0.5:
                for _ in range(rng.randint(1, 2)): x[:, 8 + rng.randrange(12)] = 0
            x += np.random.normal(0, 0.06, x.shape).astype(np.float32)
        n = len(x); X.append(x); E.append(np.full(n, c['emo'])); S.append(np.full(n, si))
    gap()
    return np.concatenate(X), np.concatenate(E), np.concatenate(S)


def batch(speakers, keys, probs, rng, nrng, B, mix, bank, n_clips=6):
    seqs = [make_seq(speakers[keys[nrng.choice(len(keys), p=probs)]], rng, mix, bank, n_clips) for _ in range(B)]
    Tm = max(len(s[0]) for s in seqs)
    X = np.zeros((B, Tm, FEAT_DIM), np.float32); X[:, :, 0] = -4.0; E = np.full((B, Tm), -1); S = np.full((B, Tm), -1)
    for i, (x, e, s) in enumerate(seqs):
        n = len(x); X[i, :n] = x; E[i, :n] = e; S[i, :n] = np.where(s >= 0, s + i * n_clips, -1)
    t = torch.tensor
    return t(X).permute(0, 2, 1), t(E), t(S), B * n_clips


def loss_fn(out, E, S, nseg, smooth=0.1, cw=None):
    logit = out[:, T.K:T.K + NE]; lf = logit.permute(0, 2, 1).reshape(-1, NE); ef = E.reshape(-1).long(); sf = S.reshape(-1).long()
    ce = Fn.cross_entropy(lf, ef, ignore_index=-1, label_smoothing=smooth, weight=cw)
    ok = sf >= 0
    seg_sum = torch.zeros(nseg, NE).index_add_(0, sf[ok], lf[ok]); seg_n = torch.zeros(nseg).index_add_(0, sf[ok], torch.ones(int(ok.sum())))
    seg_lab = torch.full((nseg,), -1, dtype=torch.long); seg_lab[sf[ok]] = ef[ok]; have = seg_n > 0
    cec = Fn.cross_entropy(seg_sum[have] / seg_n[have, None], seg_lab[have], label_smoothing=smooth, weight=cw)
    acc = (seg_sum[have].argmax(1) == seg_lab[have]).float().mean()
    return 0.4 * ce + 0.6 * cec, acc.item()


def dev_sets(feat_dir):
    import eval_corpora  # noqa: F401  (same reading as the development reports)
    out = {}
    for spec in CP.DEV:
        lines = CP.load(spec, limit=DEV_LIMIT)
        if lines: out[spec] = lines
    cl = [c for clips in crema_speakers(feat_dir, T.VAL_ACTORS).values() for c in clips]
    out['CREMA-D validation'] = [cl[i] for i in np.linspace(0, len(cl) - 1, DEV_LIMIT).round().astype(int)]
    return out


def dev_score(model, dev):
    import eval_corpora as EC
    res = {}
    for spec, lines in dev.items():
        C1, C2 = EC.evaluate([model], [1.0], [1.0], lines)
        uar = lambda C: float(np.mean([C[i, i] / C[i].sum() for i in range(NE) if C[i].sum()])) if C.sum() else float('nan')
        res[spec] = (uar(C1), uar(C2) if C2.sum() else uar(C1))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('feat_dir'); ap.add_argument('out')
    ap.add_argument('--data', default='open', help='open | all | crema, or a list of corpus specs (corpus.py)')
    ap.add_argument('--mix', default='.3,.2,.25,.25'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=16); ap.add_argument('--steps', type=int, default=150); ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--C', type=int, default=64); ap.add_argument('--drop', type=float, default=0.1); ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--crema-share', type=float, default=0.3, help='share of the training sequences that come from CREMA-D')
    ap.add_argument('--shares', default='', help='fixed shares for named corpora, e.g. fleurs_ja:train=0.05 (the others share the rest)')
    ap.add_argument('--no-floor', action='store_true', help='do not vary the level between the words')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', '4'))); torch.manual_seed(a.seed); rng = random.Random(a.seed); np.random.seed(a.seed); nrng = np.random.default_rng(a.seed)
    t0 = time.time(); mix = [float(v) for v in a.mix.split(',')]; mix = [v / sum(mix) for v in mix]
    specs = dict(open=CP.TRAIN_OPEN, all=CP.TRAIN_ALL, crema=()).get(a.data) if a.data in ('open', 'all', 'crema') else tuple(a.data.split(','))
    by = crema_speakers(a.feat_dir, [x for x in range(1001, 1092) if x not in T.TEST_ACTORS | T.VAL_ACTORS]); n_lines = {'CREMA-D': sum(map(len, by.values()))}
    for spec in specs:
        lines = CP.load(spec); n_lines[spec] = len(lines)
        for c in lines: by.setdefault((c['corpus'], c['speaker']), []).append(c)
    speakers = make_speakers(by); bank = noise_bank(speakers)
    if a.no_floor: globals()['vary_floor'] = lambda raws, rng, bank: raws
    # sampling: CREMA-D a fixed share, the other corpora by the square root of their size; speakers evenly within a corpus
    fixed = {k: float(v) for k, v in (s.split('=') for s in a.shares.split(',') if s)}
    corp = sorted(n_lines); w = {c: np.sqrt(n_lines[c]) for c in corp if c != 'CREMA-D' and n_lines[c] and c not in fixed}
    tot = sum(w.values()); share = {c: (1 - a.crema_share - sum(fixed.values())) * v / tot for c, v in w.items()} if tot else {}
    share.update(fixed); share['CREMA-D'] = a.crema_share if share else 1.0
    keys = sorted(speakers); per = {c: sum(1 for k in keys if k[0] == c) for c in share}
    probs = np.array([share[k[0]] / per[k[0]] for k in keys]); probs /= probs.sum()
    print('lines:', {c: n_lines[c] for c in corp}, flush=True)
    print('share of the training sequences:', {c: round(float(v), 3) for c, v in share.items()}, f'speakers {len(keys)}; mix {mix}; load {time.time() - t0:.0f}s', flush=True)
    dev = dev_sets(a.feat_dir); print('development lines for choosing the epoch:', {k: len(v) for k, v in dev.items()}, flush=True)
    model = T.TCN(FEAT_DIM, T.K + NE + T.NA, C=a.C, causal=False, drop=a.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * a.steps, pct_start=0.15)
    best, best_res = -1, None

    def save(score, res):
        meta = dict(mode='offline', lookahead=0, names=T.NAMES, emos=T.EMOS, aux=T.AUX, feat_dim=FEAT_DIM, globals=[], train_mix=mix, C=a.C, dil=list(model.dil), causal=False,
                    reads='line' if mix[0] != 1 else 'relative', role='reader', trained_on=a.data, lines=n_lines, dev_score=score,
                    dev={k: [round(x, 4) for x in v] for k, v in res.items()},
                    y_mean=[0.0] * T.K, y_std=[1.0] * T.K, neutral=[0.0] * T.K, n_au=len(T.CHANNELS), dataset='see train/SPLITS.md: numeric features only')
        T.export(model, a.out, meta); torch.save(dict(state=model.state_dict(), G=[], la=0, C=a.C, causal=False), a.out[:-5] + '.pt')

    for ep in range(a.epochs):
        model.train(); accs = []
        for _ in range(a.steps):
            X, E, S, nseg = batch(speakers, keys, probs, rng, nrng, a.batch, mix, bank)
            l, acc = loss_fn(model(X), E, S, nseg); accs.append(acc)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval(); res = dev_score(model, dev)
        out_of = [np.mean(v) for k, v in res.items() if k != 'CREMA-D validation' and not k.startswith('fleurs')]
        score = float(np.mean(out_of + [np.mean(res['CREMA-D validation'])]))
        if score > best: best, best_res = score, res; save(best, best_res)   # the files always hold the best epoch so far
        print(f'ep {ep} train {np.mean(accs):.3f} dev {score:.3f} | ' + '  '.join(f'{k} {v[0]:.2f}/{v[1]:.2f}' for k, v in res.items()) + f' {time.time() - t0:.0f}s', flush=True)
    print('saved', os.path.basename(a.out), 'dev', round(best, 4), flush=True)


if __name__ == '__main__':
    main()
