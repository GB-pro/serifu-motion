"""Train the Serifu Motion speech->emo temporal conv net and export it for the browser.
Usage: python train.py <feat_dir> <out_json> --mode offline|realtime [--epochs N] [--mix a,b,c,d] [--globals names]

The app knows a speaker only from the recording it is given, so every training sequence is normalised the way the app
would normalise a recording: a long one (all lines of the actor), a few lines, a single line, or lines that are all in
one emotion (--mix gives the shares). --globals adds recording-level inputs (absolute pitch, amount of speech, ...).

Outputs per frame: [K emo channels (z-scored)] [6 emotion logits] [aux: accHead, accBrow, intensity]
The regression channels drive lip sync and the "natural" style; emotion logits and aux signals are the
"what / when" reading that the animation interpretation layer (app/core/animate.js) turns into acting.

The five networks of the app (iteration 5; speakers as in SPLITS.md, logs in train5_*.log), run from this folder:
  app/models/emo_offline.json     python train.py <feat> <out> --mix .3,.2,.25,.25              'line' reader
  app/models/emo_offline_b.json   python train.py <feat> <out> --mix .3,.2,.25,.25 --seed 1     'line' reader
  app/models/emo_offline_c.json   python train.py <feat> <out> --mix 1,0,0,0                    'relative' reader
  app/models/emo_realtime.json    python train.py <feat> <out> --mode realtime --live <live/train> --live-eval <live/eval_val> --late 0.6
  app/models/emo_realtime_b.json  python train.py <feat> <out> --mode realtime --mix 1,0,0,0 --live-eval <live/eval_val>
The live sessions come from sim_live.mjs:  node train/sim_live.mjs <feat> <live/train> <training and validation actors> train
and  node train/sim_live.mjs <feat> <live/eval_val> <validation actors> eval.  How the two kinds of reader share the
reading is in runNet and LiveEmo (app/core/generator.js).
"""
import argparse, json, base64, random, time, os, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
from dataset import load_all, actor_baselines, apply_baseline, add_aux, NAMES, EMOS, AUX, FEAT_DIM, CHANNELS
import simnorm as SN

SIL = np.zeros(FEAT_DIM, np.float32); SIL[0] = -4.0
GAP_EVAL = 24
FRESH_ACTORS = set(a for a in range(1001, 1076) if a % 8 == 0)  # second held-out set, see SPLITS.md
TEST_ACTORS = set(map(int, os.environ['TEST_ACTORS'].split(','))) if os.environ.get('TEST_ACTORS') else set(range(1083, 1092)) | FRESH_ACTORS
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


GLOBAL_SETS = dict(f0=[SN.G_F0], spread=[SN.G_SPREAD], amount=[SN.G_AMOUNT], cent=[SN.G_CENT, SN.G_CENTSD],
                   mfmean=list(range(SN.G_MFMEAN, SN.G_MFMEAN + 12)), mfstd=list(range(SN.G_MFSTD, SN.G_MFSTD + 12)), fluxsd=[SN.G_FLUXSD])
FRAGILE = set([SN.G_CENT, SN.G_CENTSD] + GLOBAL_SETS['mfmean'])  # change with the microphone


def parse_globals(spec):
    return [] if spec in ('', 'none') else [i for n in spec.split(',') for i in GLOBAL_SETS[n]]


def prepare_actor(clips, feat_dir):
    """Per actor: the clips, the long-recording statistics (all lines) and the clips by emotion."""
    by = {}
    for c in clips:
        r = SN.load_raw(feat_dir, c['name']); c['R'] = r; c['n'] = len(c['X'])
        by.setdefault(c['actor'], dict(clips=[], by_emo={}))
        by[c['actor']]['clips'].append(c); by[c['actor']]['by_emo'].setdefault(c['emo'], []).append(c)
    for a, A in by.items():  # the long recording is every line of the actor, as in feats.mjs (also lines whose video could not be used)
        names = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(feat_dir, f'{a}_*.raw.f32')))
        A['stats'] = SN.speaker_stats([SN.load_raw(feat_dir, nm) for nm in names]); A['glob'] = SN.global_features(A['stats'])
    return by


def choose_recording(A, rng, n_clips, mix):
    """A simulated recording of one actor -> [(clip, statistics of that recording or None for the long recording)]."""
    r = rng.random(); clips = A['clips']
    if r < mix[0]:
        return [(c, None) for c in rng.sample(clips, min(n_clips, len(clips)))]
    if r < mix[0] + mix[1]:      # a few lines, any emotions
        rec = rng.sample(clips, min(rng.randint(2, 12), len(clips)))
    elif r < mix[0] + mix[1] + mix[2]:  # single lines: every clip is a recording of its own
        return [(c, SN.speaker_stats([c['R']])) for c in rng.sample(clips, min(n_clips, len(clips)))]
    else:                        # lines that are all in one emotion
        pool = A['by_emo'][rng.choice(sorted(A['by_emo']))]; rec = rng.sample(pool, min(rng.randint(2, 14), len(pool)))
    st = SN.speaker_stats([c['R'] for c in rec])
    return [(c, st) for c in rec[:n_clips]]


def clip_input(c, st, A, G):
    """Features of a clip inside a recording with statistics st, and the recording-level inputs."""
    if st is None: return c['X'], A['glob'][G]
    return SN.normalize(c['R'], st)[:c['n']], SN.global_features(st)[G]


def aug_globals(g, G, rng):
    g = g + np.random.normal(0, 0.1, g.shape).astype(np.float32)
    for j, gi in enumerate(G):
        if gi == SN.G_F0: g[j] += rng.uniform(-0.4, 0.4)       # the same performance by a higher or lower voice
        elif gi in FRAGILE: g[j] += rng.gauss(0, AUG_CHANNEL)   # another microphone
    return g


AUG_CHANNEL = float(os.environ.get('AUG_CHANNEL', '0.5'))


def make_seq(A, rng, G, mix, n_clips=6, aug=True):
    X, Y, M, E, Am, AM, S = [], [], [], [], [], [], []
    rec = choose_recording(A, rng, n_clips, mix); g = None

    def gap(g):
        n = rng.randint(8, 30)
        X.append(np.concatenate([np.repeat(SIL[None], n, 0), np.repeat(g[None], n, 0)], 1)); Y.append(np.zeros((n, K), np.float32)); M.append(np.zeros(n, np.float32))
        E.append(np.full(n, -1)); Am.append(np.zeros((n, NA), np.float32)); AM.append(np.zeros((n, NA), np.float32)); S.append(np.full(n, -1))
    for si, (c, st) in enumerate(rec):
        x, g0 = clip_input(c, st, A, G)
        if g is None or st is not rec[si - 1][1]:  # a new recording begins (single lines) or the first clip
            g = aug_globals(g0, G, rng) if aug else g0
        gap(g)
        x = x.copy()
        if aug:
            x[:, 0] += rng.uniform(-0.4, 0.4)
            x[:, 2] = np.where(x[:, 1] > 0, x[:, 2] + rng.uniform(-0.4, 0.4), 0)
            x[:, 8:] *= rng.uniform(0.8, 1.2)
            if rng.random() < 0.5:  # drop one or two cepstral bands for the whole clip
                for _ in range(rng.randint(1, 2)): x[:, 8 + rng.randrange(12)] = 0
            x += np.random.normal(0, 0.06, x.shape).astype(np.float32)
        n = len(x)
        X.append(np.concatenate([x, np.repeat(g[None], n, 0)], 1)); Y.append(c['Yz']); M.append(np.ones(n, np.float32)); E.append(np.full(n, c['emo']))
        Am.append(c['A']); AM.append(c['AM']); S.append(np.full(n, si))
    gap(g)
    return [np.concatenate(v) for v in (X, Y, M, E, Am, AM, S)]


def batch(by_actor, rng, B, G, mix, aug=True, n_clips=6):
    actors = list(by_actor); D = FEAT_DIM + len(G)
    seqs = [make_seq(by_actor[rng.choice(actors)], rng, G, mix, n_clips=n_clips, aug=aug) for _ in range(B)]
    T = max(len(s[0]) for s in seqs)
    X = np.zeros((B, T, D), np.float32); X[:, :, 0] = -4.0
    Y = np.zeros((B, T, K), np.float32); M = np.zeros((B, T), np.float32); E = np.full((B, T), -1)
    A = np.zeros((B, T, NA), np.float32); AM = np.zeros((B, T, NA), np.float32); S = np.full((B, T), -1)
    for i, (x, y, m, e, a, am, s) in enumerate(seqs):
        n = len(x); X[i, :n] = x; X[i, n:, FEAT_DIM:] = x[-1, FEAT_DIM:]; Y[i, :n] = y; M[i, :n] = m; E[i, :n] = e; A[i, :n] = a; AM[i, :n] = am
        S[i, :n] = np.where(s >= 0, s + i * n_clips, -1)
    t = torch.tensor
    return dict(X=t(X).permute(0, 2, 1), Y=t(Y).permute(0, 2, 1), M=t(M), E=t(E), A=t(A).permute(0, 2, 1), AM=t(AM).permute(0, 2, 1), S=t(S), nseg=B * n_clips)


def load_sessions(live_dir, by_all):
    """Simulated live sessions (sim_live.mjs): per actor a list of streams with the position of every line."""
    byname = {c['name']: c for A in by_all.values() for c in A['clips']}; out = {}
    for jf in sorted(glob.glob(os.path.join(live_dir, '*.json'))):
        m = json.load(open(jf)); X = np.fromfile(jf[:-5] + '.f32', np.float32).reshape(-1, m['dim'])
        lines = [(byname.get(l['name']), l['t0'], l['n'], EMOS.index(l['name'].split('_')[2])) for l in m['lines']]
        out.setdefault(m['actor'], []).append(dict(X=X, lines=lines))
    return out


def make_seq_live(sess, rng, G, n_clips=6, aug=True):
    """A stretch of a live session: up to n_clips consecutive lines with the pauses between them, as live mode saw them."""
    s = rng.choice(sess); L = len(s['lines'])
    i = 0 if rng.random() < 0.25 else rng.randrange(L); j = min(L, i + n_clips)
    ta = max(0, s['lines'][i][1] - rng.randint(8, 30))
    end = s['lines'][j - 1][1] + s['lines'][j - 1][2]; nxt = s['lines'][j][1] if j < L else len(s['X'])
    tb = min(nxt, end + rng.randint(8, 30))
    x = s['X'][ta:tb][:, list(range(FEAT_DIM)) + [FEAT_DIM + g for g in G]].copy(); T = len(x)
    Y = np.zeros((T, K), np.float32); M = np.zeros(T, np.float32); E = np.full(T, -1); A = np.zeros((T, NA), np.float32); AM = np.zeros((T, NA), np.float32); S = np.full(T, -1)
    R = np.zeros(T, np.float32)  # how far into its line a frame is (0..1)
    for si, (c, t0, n, emo) in enumerate(s['lines'][i:j]):
        a0 = t0 - ta; E[a0:a0 + n] = emo; S[a0:a0 + n] = si; R[a0:a0 + n] = np.arange(n) / max(1, n - 1)
        if c is None: continue  # a line without usable video: the feeling is known, the motion is not
        m = min(n, c['n']); Y[a0:a0 + m] = c['Yz'][:m]; M[a0:a0 + m] = 1; A[a0:a0 + m] = c['A'][:m]; AM[a0:a0 + m] = c['AM'][:m]
    if aug:
        sp = x[:, 0] > -3.9
        x[sp, 0] += rng.uniform(-0.3, 0.3)
        x[:, 2] = np.where(x[:, 1] > 0, x[:, 2] + rng.uniform(-0.4, 0.4), 0)
        x[:, 8:FEAT_DIM] *= rng.uniform(0.8, 1.2)
        if rng.random() < 0.5:
            for _ in range(rng.randint(1, 2)): x[:, 8 + rng.randrange(12)] = 0
        x[:, :FEAT_DIM] += np.random.normal(0, 0.06, (T, FEAT_DIM)).astype(np.float32)
        if G: x[:, FEAT_DIM:] += aug_globals(np.zeros(len(G), np.float32), G, rng)[None]
    return [x, Y, M, E, A, AM, S, R]


def batch_live(sessions, rng, B, G, aug=True, n_clips=6):
    actors = list(sessions); D = FEAT_DIM + len(G)
    seqs = [make_seq_live(sessions[rng.choice(actors)], rng, G, n_clips=n_clips, aug=aug) for _ in range(B)]
    T = max(len(q[0]) for q in seqs)
    X = np.zeros((B, T, D), np.float32); X[:, :, 0] = -4.0
    Y = np.zeros((B, T, K), np.float32); M = np.zeros((B, T), np.float32); E = np.full((B, T), -1)
    A = np.zeros((B, T, NA), np.float32); AM = np.zeros((B, T, NA), np.float32); S = np.full((B, T), -1); R = np.zeros((B, T), np.float32)
    for i, (x, y, m, e, a, am, sg, r) in enumerate(seqs):
        n = len(x); X[i, :n] = x; X[i, n:] = x[-1]; Y[i, :n] = y; M[i, :n] = m; E[i, :n] = e; A[i, :n] = a; AM[i, :n] = am; R[i, :n] = r
        S[i, :n] = np.where(sg >= 0, sg + i * n_clips, -1)
    t = torch.tensor
    return dict(X=t(X).permute(0, 2, 1), Y=t(Y).permute(0, 2, 1), M=t(M), E=t(E), A=t(A).permute(0, 2, 1), AM=t(AM).permute(0, 2, 1), S=t(S), R=t(R), nseg=B * n_clips)


def live_posteriors(models, sess, G, la):
    """Run the realtime network(s) over a whole session -> per frame: emotion probabilities (aligned with the frame), speech flag."""
    x = sess['X'][:, list(range(FEAT_DIM)) + [FEAT_DIM + g for g in G]]
    xx = np.concatenate([x, np.repeat(x[-1:], la, 0)]); P = 0
    with torch.no_grad():
        for m in models:
            lg = m(torch.tensor(xx).T[None])[0].T.numpy()[la:, K:K + NE]; e = np.exp(lg - lg.max(1, keepdims=True)); P = P + e / e.sum(1, keepdims=True)
    return P / len(models), (x[:, 1] > 0.5) | (x[:, 0] > -1.5)


def live_director(P, sp, la, margins=(0.03, 0.08), ema=0.15, reread=(8, 16, 12), new_line=15, cont=True):
    """What the live director (animate.js) hears and plays at every frame, from the network's answers.
    new_line: a pause of at least this many frames starts a new line (first pose from the context, evidence from
    scratch); after a shorter pause the same line goes on (cont=False, new_line=8: the rule of the first versions,
    where every pause of 0.27 s started from scratch and a pause of 6 or 7 frames left the rest of the line unheard)."""
    T = len(P); ctx = np.full(NE, 1 / NE); recent = np.full(NE, 1 / NE); acc_n = 0; in_ph = False; sil = 99; emo = 0
    heard = np.zeros(T, int); played = np.zeros(T, int)
    for t in range(T):
        p = P[t]; ctx = 0.995 * ctx + 0.005 * p; v = sp[t]; coming = sp[t + 1:t + 1 + la].any()
        if not in_ph and (v or coming):
            if sil >= new_line: in_ph = True; acc_n = 0; emo = int(ctx.argmax())
            elif cont: in_ph = True
        sil = 0 if v else sil + 1
        if in_ph:
            if v:
                recent = recent + (ema if acc_n else 1.0) * (p - recent); acc_n += 1
                if acc_n == reread[0] or acc_n == reread[1] or (acc_n > reread[1] and (acc_n - reread[1]) % reread[2] == 0):
                    q = 0.85 * recent + 0.15 * ctx; q = q / q.sum(); b = int(q.argmax())
                    if b != emo and q[b] - q[emo] > margins[0 if acc_n == reread[0] else 1]: emo = b
            if sil == 6: in_ph = False
        elif sil == 80: emo = 0
        heard[t] = int((recent if acc_n > 3 else ctx).argmax()); played[t] = emo
    return heard, played


def eval_live(models, sessions, G, la, **kw):
    """Live reading on eval-order sessions: share of lines where the feeling heard / played is the acted one."""
    tot = {k: np.zeros(2) for k in ('heard_0.5s', 'heard_1s', 'heard_end', 'played_0.5s', 'played_1s', 'played_end')}; per = np.zeros((NE, 2))
    for a, ss in sorted(sessions.items()):
        for sess in ss:
            P, sp = live_posteriors(models, sess, G, la); heard, played = live_director(P, sp, la, **kw); T = len(P)
            for (c, t0, n, emo) in sess['lines']:
                idx = np.flatnonzero(sp[t0:min(T, t0 + n + 6)])
                if not len(idx): continue
                s0, e0 = t0 + idx[0], t0 + idx[-1]
                for key, i in (('0.5s', s0 + 15), ('1s', s0 + 30), ('end', e0)):
                    if i <= e0 and i < T:
                        tot['heard_' + key] += (heard[i] == emo, 1); tot['played_' + key] += (played[i] == emo, 1)
                per[emo] += (played[e0] == emo, 1)
    return {k: round(float(v[0] / max(1, v[1])), 4) for k, v in tot.items()}, (per[:, 0] / np.maximum(1, per[:, 1])).round(2).tolist()


def shift_for_causal(b, la):
    """Realtime model: output at frame t describes frame t-la (uses la frames of lookahead)."""
    if not la: return b
    b = dict(b)
    for k in ('Y', 'A', 'AM'): b[k] = Fn.pad(b[k], (la, 0))
    b['M'] = Fn.pad(b['M'], (la, 0)); b['E'] = Fn.pad(b['E'], (la, 0), value=-1); b['S'] = Fn.pad(b['S'], (la, 0), value=-1)
    if 'R' in b: b['R'] = Fn.pad(b['R'], (la, 0))
    X = Fn.pad(b['X'], (0, la)); X[:, 0, -la:] = -4.0; X[:, FEAT_DIM:, -la:] = X[:, FEAT_DIM:, -la - 1:-la]; b['X'] = X
    return b


def loss_fn(out, b, smooth=0.1, wce=1.0, late=0.0):
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
    if late and 'R' in b:  # realtime: what the network says late in a line is what gets played, so it counts more
        wf = ((1 - late) + 2 * late * b['R'].reshape(-1)) * (ef >= 0)
        ce = (Fn.cross_entropy(lf, ef.clamp(min=0), label_smoothing=smooth, reduction='none') * wf).sum() / (wf.sum() + 1e-6)
    else:
        ce = Fn.cross_entropy(lf, ef, ignore_index=-1, label_smoothing=smooth)
    # clip-level emotion: mean logits over each clip
    ok = sf >= 0
    seg_sum = torch.zeros(b['nseg'], NE).index_add_(0, sf[ok], lf[ok]); seg_n = torch.zeros(b['nseg']).index_add_(0, sf[ok], torch.ones(int(ok.sum())))
    seg_lab = torch.full((b['nseg'],), -1, dtype=torch.long); seg_lab[sf[ok]] = ef[ok]
    have = seg_n > 0
    cec = Fn.cross_entropy(seg_sum[have] / seg_n[have, None], seg_lab[have], label_smoothing=smooth)
    acc = (seg_sum[have].argmax(1) == seg_lab[have]).float().mean()
    auxl = (((aux - A) ** 2) * AM).sum() / (AM.sum() + 1e-6)
    total = mse + 0.5 * vel + 0.3 * stdl + wce * (0.15 * ce + 0.25 * cec) + 0.3 * auxl
    return total, dict(mse=mse.item(), vel=vel.item(), std=stdl.item(), ce=ce.item(), cec=cec.item(), acc=acc.item(), aux=auxl.item())


def run_padded(model, x, la, pad=20):
    """x (n, D) with silence around it -> network output for the n frames."""
    sil = np.repeat(np.concatenate([SIL, x[0, FEAT_DIM:]])[None], pad + la, 0)
    xx = np.concatenate([sil[:pad], x, sil])
    with torch.no_grad():
        o = model(torch.tensor(xx).T[None])[0].T.numpy()
    return o[pad + la: pad + la + len(x)]


def predict_clip(model, c, la):
    o = run_padded(model, c['XG'], la)
    return o[:, :K], o[:, K:K + NE], o[:, K + NE:]


def read_recording(models, rec, st, A, G, la):
    """The clips of one recording joined with pauses, as the app would get them -> emotion read for each clip
    (several models: their probabilities are averaged, as the app's committee does)."""
    xs = [clip_input(c, st, A, G) for c in rec]; g = xs[0][1]
    gapx = np.repeat(np.concatenate([SIL, g])[None], GAP_EVAL, 0); parts = []; spans = []; t = 0
    for (x, _), c in zip(xs, rec):
        parts += [gapx, np.concatenate([x, np.repeat(g[None], len(x), 0)], 1)]; spans.append((t + GAP_EVAL, t + GAP_EVAL + len(x))); t += GAP_EVAL + len(x)
    xx = np.concatenate(parts + [gapx]); P = 0
    for m in models:
        lg = run_padded(m, xx, la, pad=30)[:, K:K + NE]; e = np.exp(lg - lg.max(1, keepdims=True)); P = P + e / e.sum(1, keepdims=True)
    out = []
    for (a, b), (x, _) in zip(spans, xs):
        p = P[a:b]; sp = (x[:, 1] > 0.5) | (x[:, 0] > -1.5)
        out.append(int(p[sp].mean(0).argmax()) if sp.any() else int(p.mean(0).argmax()))
    return out


def eval_conditions(models, by_actor, G, la):
    """Share of lines whose emotion is read correctly, by the kind of recording the line is part of."""
    res = {}

    def add(key, rec, st, A):
        r = read_recording(models, rec, st, A, G, la); t = res.setdefault(key, [0, 0, np.zeros((NE, 2))])
        for c, e in zip(rec, r): t[0] += e == c['emo']; t[1] += 1; t[2][c['emo']] += (e == c['emo'], 1)
    for a, A in sorted(by_actor.items()):
        clips = sorted(A['clips'], key=lambda c: c['name'])
        add('all', clips, None, A)
        by_s = {}
        for c in clips: by_s.setdefault(c['sent'], []).append(c)
        for g in by_s.values():
            add('six', g, SN.speaker_stats([c['R'] for c in g]), A)
            for i in range(0, len(g), 3): add('three', g[i:i + 3], SN.speaker_stats([c['R'] for c in g[i:i + 3]]), A)
        for c in clips: add('single', [c], SN.speaker_stats([c['R']]), A)
        for e, g in sorted(A['by_emo'].items()):
            g = sorted(g, key=lambda c: c['name']); st = SN.speaker_stats([c['R'] for c in g]); add('same', g, st, A)
            add('same_long', g, dict(st, nVoiced=st['nVoiced'] * 6), A)  # probe: the same recording, but as if it were six times as long
    return {k: (v[0] / v[1], (v[2][:, 0] / np.maximum(1, v[2][:, 1])).round(2).tolist()) for k, v in res.items()}


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
    ap.add_argument('--mix', default='1,0,0,0', help='shares of training recordings: long, a few lines, single line, one emotion')
    ap.add_argument('--globals', default='none', help='recording-level inputs: ' + ','.join(GLOBAL_SETS))
    ap.add_argument('--live', default='', help='realtime: folder of simulated live sessions of the training and validation actors (sim_live.mjs train)')
    ap.add_argument('--live-eval', default='', help='realtime: folder of eval-order sessions of the validation actors (sim_live.mjs eval)')
    ap.add_argument('--late', type=float, default=0.0, help='realtime: 0..1, how much more the late frames of a line count in the emotion loss')
    ap.add_argument('--wce', type=float, default=1.0, help='weight of the emotion reading in the loss, relative to the first models')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', '4'))); torch.manual_seed(a.seed); rng = random.Random(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    clips = load_all(a.feat_dir)
    base, gneu = actor_baselines(clips)
    apply_baseline(clips, base, gneu); add_aux(clips)
    mix = [float(v) for v in a.mix.split(',')]; mix = [v / sum(mix) for v in mix]; G = parse_globals(a.globals)
    by_all = prepare_actor(clips, a.feat_dir)
    for c in clips: c['XG'] = np.concatenate([c['X'], np.repeat(by_all[c['actor']]['glob'][G][None], c['n'], 0)], 1)
    train = [c for c in clips if c['actor'] not in TEST_ACTORS | VAL_ACTORS]
    val = [c for c in clips if c['actor'] in VAL_ACTORS]
    Yall = np.concatenate([c['Yn'] for c in train])
    mu, sd = Yall.mean(0), Yall.std(0) + 1e-4
    for c in clips:
        c['Yz'] = ((c['Yn'] - mu) / sd).astype(np.float32)
    print(f'clips {len(clips)} train {len(train)} val {len(val)} load {time.time()-t0:.0f}s', flush=True)
    by_tr = {k: v for k, v in by_all.items() if k not in TEST_ACTORS | VAL_ACTORS}; by_va = {k: v for k, v in by_all.items() if k in VAL_ACTORS}
    print(f'actors: train {len(by_tr)} val {len(by_va)} excluded {len(by_all) - len(by_tr) - len(by_va)}; mix {mix}; globals {a.globals} ({len(G)})', flush=True)
    causal = a.mode == 'realtime'
    la = 4 if causal else 0
    model = TCN(FEAT_DIM + len(G), K + NE + NA, C=a.C, causal=causal, drop=a.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * a.steps, pct_start=0.15)
    best, best_state = 1e9, None
    vrng = random.Random(1)
    if causal and a.live:
        sess = load_sessions(a.live, by_all); sess_tr = {k: v for k, v in sess.items() if k in by_tr}; sess_va = {k: v for k, v in sess.items() if k in by_va}
        print(f'live sessions: train actors {len(sess_tr)} val actors {len(sess_va)}', flush=True)
        get = lambda r, aug, src: batch_live(sess_tr if src == 'tr' else sess_va, r, 16, G, aug=aug)
    else:
        get = lambda r, aug, src: batch(by_tr if src == 'tr' else by_va, r, 16, G, mix, aug=aug)
    vb = [shift_for_causal(get(vrng, False, 'va'), la) for _ in range(8)]
    for ep in range(a.epochs):
        model.train()
        for st in range(a.steps):
            b = shift_for_causal(get(rng, True, 'tr'), la)
            l, _ = loss_fn(model(b['X']), b, wce=a.wce, late=a.late)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval(); vl = []; parts = []
        with torch.no_grad():
            for b in vb:
                l, p = loss_fn(model(b['X']), b, smooth=0.0, wce=a.wce, late=a.late); vl.append(l.item()); parts.append(p)
        v = float(np.mean(vl))
        if v < best:
            best = v; best_state = {k: t.clone() for k, t in model.state_dict().items()}
        pm = {k: round(float(np.mean([p[k] for p in parts])), 3) for k in parts[0]}
        print(f'ep {ep} val {v:.4f} {pm} {time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(best_state); model.eval()
    cond = eval_conditions([model], by_va, G, la)
    print('validation reading by recording:', '  '.join(f'{k} {v[0]:.3f}' for k, v in cond.items()), flush=True)
    print('  single line by emotion', dict(zip(EMOS, cond['single'][1])), ' one emotion', dict(zip(EMOS, cond['same'][1])), flush=True)
    live_val = None
    if causal and a.live_eval:
        ev = {k: v for k, v in load_sessions(a.live_eval, by_all).items() if k in by_va}
        live_val, per = eval_live([model], ev, G, la)
        print('validation live:', live_val, ' played at the end by emotion', dict(zip(EMOS, per)), flush=True)
    extra = compute_stats(model, train, val, mu, sd, la)
    meta = dict(mode=a.mode, lookahead=la, names=NAMES, emos=EMOS, aux=AUX, feat_dim=FEAT_DIM + len(G), globals=G, train_mix=None if (causal and a.live) else mix, trained_on='simulated live sessions (train/sim_live.mjs)' if (causal and a.live) else 'simulated recordings', C=a.C,
                reads='line' if (causal and a.live) or mix[0] != 1 else 'relative',  # see runNet / LiveEmo in app/core/generator.js
                val_reading={k: round(v[0], 4) for k, v in cond.items()}, val_live=live_val,
                dil=list(model.dil), causal=causal, y_mean=mu.tolist(), y_std=sd.tolist(),
                neutral=gneu.tolist(), n_au=len(CHANNELS), trained_clips=len(train), val_loss=best,
                dataset='CREMA-D (ODbL) — 91 actors, numeric features only', **extra)
    export(model, a.out, meta); torch.save(dict(state=model.state_dict(), G=G, la=la, C=a.C, causal=causal), a.out[:-5] + '.pt')
    print('saved', a.out, 'best', best, 'aux r', extra['aux_val_r'])


if __name__ == '__main__':
    main()
