"""How predictable is the actor's own emphasis (strong peaks of head / brow motion energy) from the voice?
python accent_ceiling.py <feat_dir> <eval_dir>"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'train'))
from dataset import load_all, actor_baselines, apply_baseline, add_aux, NAMES
from train import TEST_ACTORS
feat, ev = sys.argv[1], sys.argv[2]
clips = load_all(feat); base, g = actor_baselines(clips); apply_baseline(clips, base, g); add_aux(clips)
index = {i['name']: i for i in json.load(open(os.path.join(ev, 'index.json')))}
te = [c for c in clips if c['actor'] in TEST_ACTORS and c['name'] in index]
THR, TOL = 1.0, 3
def peaks(v, thr): 
    pk = np.zeros(len(v), bool); pk[1:-1] = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thr); return pk
def near_mask(c, chans=(0, 1), thr=THR, tol=TOL):
    n = len(c['A']); pk = np.zeros(n, bool)
    for ch in chans: pk |= peaks(c['A'][:, ch], thr)
    near = np.zeros(n, bool)
    for t in np.flatnonzero(pk): near[max(0, t - tol):t + tol + 1] = True
    return near, pk
def score(name, pick, chans=(0, 1), shift=0):
    hits = tot = cn = cd = 0
    for c in te:
        near, _ = near_mask(c, chans); n = len(near); info = index[c['name']]
        for ph in info['auto']['phrases']:
            a0, a1 = max(0, ph['t0']), min(n, ph['t1'])
            if a1 > a0: cn += near[a0:a1].sum(); cd += a1 - a0
        for t in pick(c, info):
            t = int(t) + shift
            if 0 <= t < n: tot += 1; hits += bool(near[t])
    hit, ch = hits / max(1, tot), cn / max(1, cd)
    print('%-58s n=%5d hit %.3f chance %.3f lift %.2f' % (name, tot, hit, ch, hit / ch))
cur = lambda c, i: [a['t'] for ph in i['auto']['phrases'] for a in ph['accents']]
allnuc = lambda c, i: [u['t'] for u in i['nuclei']]
def topnuc(k):
    return lambda c, i: [u['t'] for u in sorted(i['nuclei'], key=lambda u: -u['score'])[:k]]
def auxpk(ch, thr):
    def f(c, i):
        a = np.fromfile(os.path.join(ev, c['name'] + '.aux.f32'), np.float32).reshape(-1, 3); return np.flatnonzero(peaks(a[:, ch], thr))
    return f
def auxtop(ch, k):
    def f(c, i):
        a = np.fromfile(os.path.join(ev, c['name'] + '.aux.f32'), np.float32).reshape(-1, 3)[:, ch]; pk = np.flatnonzero(peaks(a, -9)); return pk[np.argsort(-a[pk])][:k]
    return f
def enpk(k):
    def f(c, i):
        e = c['X'][:, 0]; pk = np.flatnonzero(peaks(e, -1.3)); return pk[np.argsort(-e[pk])][:k]
    return f
for sh in (-4, -2, 0, 2): score('current accents, shifted %+d' % sh, cur, shift=sh)
score('current accents vs head peaks only', cur, chans=(0,)); score('current accents vs brow peaks only', cur, chans=(1,))
score('all syllable nuclei', allnuc)
for k in (1, 2, 3): score('top-%d nuclei by current score' % k, topnuc(k))
for thr in (0.0, 0.5, 1.0): score('predicted head-accent peaks > %.1f' % thr, auxpk(0, thr)); score('predicted brow-accent peaks > %.1f' % thr, auxpk(1, thr))
for k in (1, 2): score('top-%d predicted head-accent peaks' % k, auxtop(0, k)); score('top-%d predicted brow-accent peaks' % k, auxtop(1, k)); score('top-%d energy peaks' % k, enpk(k))
for k in (1, 2):
    for sh in (-3, -2, -1, 1, 2, 3): score('top-%d predicted head-accent peaks shifted %+d' % (k, sh), auxtop(0, k), shift=sh)
# correlation of predicted and actual envelopes
rs = [[], []]
for c in te:
    a = np.fromfile(os.path.join(ev, c['name'] + '.aux.f32'), np.float32).reshape(-1, 3); n = min(len(a), len(c['A']))
    for ch in range(2):
        if a[:n, ch].std() > 1e-5 and c['A'][:n, ch].std() > 1e-5: rs[ch].append(np.corrcoef(a[:n, ch], c['A'][:n, ch])[0, 1])
print('correlation predicted vs actual envelope: head %.3f brow %.3f' % (np.mean(rs[0]), np.mean(rs[1])))
# where do the actor's strong peaks fall relative to the phrase (share by tenth of the phrase)
pos = []
for c in te:
    near, pk = near_mask(c); n = len(pk)
    for ph in index[c['name']]['auto']['phrases']:
        a0, a1 = max(0, ph['t0']), min(n, ph['t1'])
        for t in np.flatnonzero(pk[a0:a1]): pos.append(t / max(1, a1 - a0))
    
print('actor strong peaks by position in phrase (deciles):', np.round(np.histogram(pos, bins=10, range=(0, 1))[0] / max(1, len(pos)), 3))
out = []
for c in te:
    near, pk = near_mask(c); n = len(pk); sp = (c['X'][:, 1] > 0.5) | (c['X'][:, 0] > -1.5)
    out.append([pk[sp[:n]].sum(), pk[~sp[:n]].sum(), sp.sum(), (~sp).sum()])
o = np.array(out).sum(0); print('strong peaks per second: in speech %.2f, in silence %.2f' % (o[0] / o[2] * 30, o[1] / o[3] * 30))
