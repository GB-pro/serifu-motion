"""Tune the accent picker on the VALIDATION actors (never on the held-out test actors).
python accent_tune.py <feat_dir> <eval_dir> <actors: val|test>"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'train'))
from dataset import load_all, actor_baselines, apply_baseline, add_aux
from train import TEST_ACTORS, VAL_ACTORS
feat, ev, which = sys.argv[1], sys.argv[2], sys.argv[3]
clips = load_all(feat); base, g = actor_baselines(clips); apply_baseline(clips, base, g); add_aux(clips)
index = {i['name']: i for i in json.load(open(os.path.join(ev, 'index.json')))}
S = TEST_ACTORS if which == 'test' else VAL_ACTORS
te = [c for c in clips if c['actor'] in S and c['name'] in index]
for c in te: c['aux'] = np.fromfile(os.path.join(ev, c['name'] + '.aux.f32'), np.float32).reshape(-1, 3)
def peaks(v, thr):
    pk = np.zeros(len(v), bool); pk[1:-1] = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > thr); return pk
def near_mask(c, thr=1.0, tol=3):
    n = len(c['A']); pk = peaks(c['A'][:, 0], thr) | peaks(c['A'][:, 1], thr); near = np.zeros(n, bool)
    for t in np.flatnonzero(pk): near[max(0, t - tol):t + tol + 1] = True
    return near
for c in te: c['near'] = near_mask(c)
def evaluate(name, pick):
    hits = tot = cn = cd = nph = 0
    for c in te:
        near = c['near']; n = len(near); info = index[c['name']]
        for ph in info['auto']['phrases']:
            a0, a1 = max(0, ph['t0']), min(n, ph['t1'])
            if a1 <= a0: continue
            cn += near[a0:a1].sum(); cd += a1 - a0; nph += 1
            for t in pick(c, info, a0, a1):
                if 0 <= t < n: tot += 1; hits += bool(near[int(t)])
    hit, ch = hits / max(1, tot), cn / max(1, cd)
    print('%-64s n=%5d per phrase %.2f hit %.3f chance %.3f lift %.2f' % (name, tot, tot / max(1, nph), hit, ch, hit / ch)); return hit / ch
def cur(c, info, a0, a1): return [a['t'] for ph in info['auto']['phrases'] for a in ph['accents'] if a0 <= a['t'] < a1]
def ruleA(theta, wB=0.9, lo=-4, hi=2, always=True, space=10, per=0.55, wS=0.0):
    def f(c, info, a0, a1):
        aux = c['aux']; n = len(aux); nuc = [u for u in info['nuclei'] if a0 <= u['t'] < a1]
        if not nuc: return []
        val = []
        for u in nuc:
            s = slice(max(0, u['t'] + lo), min(n, u['t'] + hi + 1)); val.append(max(aux[s, 0].max(), wB * aux[s, 1].max()) + wS * u['score'])
        order = np.argsort(-np.array(val)); out = []; mx = max(1, round((a1 - a0) / 30 / per))
        for i in order:
            if len(out) >= mx: break
            t = nuc[i]['t']
            if any(abs(t - o) < space for o in out): continue
            if val[i] < theta and not (always and not out): continue
            out.append(t)
        return out
    return f
def ruleB(theta, snap=3, wB=0.9, space=10, per=0.55, always=True, shift=0):
    def f(c, info, a0, a1):
        aux = c['aux']; n = len(aux); m = np.maximum(aux[:, 0], wB * aux[:, 1]); pk = np.flatnonzero(peaks(m, -9)); pk = pk[(pk >= a0 - 2) & (pk < a1)]
        if not len(pk): return []
        nuc = np.array([u['t'] for u in info['nuclei'] if a0 <= u['t'] < a1]); order = pk[np.argsort(-m[pk])]; out = []; mx = max(1, round((a1 - a0) / 30 / per))
        for t in order:
            if len(out) >= mx: break
            if m[t] < theta and not (always and not out): continue
            tt = t
            if snap and len(nuc):
                j = np.argmin(np.abs(nuc - t))
                if abs(nuc[j] - t) <= snap: tt = int(nuc[j])
            if any(abs(tt - o) < space for o in out): continue
            out.append(tt + shift)
        return out
    return f
print(which, 'clips', len(te))
evaluate('current', cur)
for th in (0.3, 0.6, 0.8, 1.0, 1.2): evaluate('A nuclei ranked by predicted accent, theta %.1f' % th, ruleA(th))
for th in (0.6, 1.0): evaluate('A theta %.1f, window -2..+2' % th, ruleA(th, lo=-2, hi=2)); evaluate('A theta %.1f, window -6..+1' % th, ruleA(th, lo=-6, hi=1)); evaluate('A theta %.1f, + 0.3 syllable score' % th, ruleA(th, wS=0.3)); evaluate('A theta %.1f, no forced accent' % th, ruleA(th, always=False))
for th in (0.3, 0.6, 0.8, 1.0, 1.2): evaluate('B predicted peaks snapped to a syllable, theta %.1f' % th, ruleB(th))
for th in (0.6, 1.0): evaluate('B theta %.1f, not snapped' % th, ruleB(th, snap=0)); evaluate('B theta %.1f, no forced accent' % th, ruleB(th, always=False)); evaluate('B theta %.1f, snapped within 5' % th, ruleB(th, snap=5))
for sh in (-2, -1, 1, 2): evaluate('B theta 0.8, not snapped, shifted %+d' % sh, ruleB(0.8, snap=0, shift=sh))
def ruleC(ts, tw, wB=0.9, lo=-4, hi=2, space=12, per=0.7):
    def f(c, info, a0, a1):
        aux = c['aux']; n = len(aux); nuc = [u for u in info['nuclei'] if a0 <= u['t'] < a1]
        if not nuc: return []
        val = []
        for u in nuc:
            s = slice(max(0, u['t'] + lo), min(n, u['t'] + hi + 1)); val.append(max(aux[s, 0].max(), wB * aux[s, 1].max()))
        order = np.argsort(-np.array(val)); out = []; mx = max(1, round((a1 - a0) / 30 / per))
        for i in order:
            if len(out) >= mx: break
            t = nuc[i]['t']
            if any(abs(t - o) < space for o in out): continue
            if val[i] >= ts or (not out and val[i] >= tw): out.append(t)
        return out
    return f
print('--- rule C: strong accents where emphasis is predicted, one soft key accent otherwise')
for ts in (0.6, 0.8, 1.0):
    for tw in (0.0, 0.3, 0.5, 9):
        evaluate('C strong %.1f weak %.1f' % (ts, tw), ruleC(ts, tw))
evaluate('C strong 0.8 weak 0.3, spacing 10, per 0.55', ruleC(0.8, 0.3, space=10, per=0.55))
evaluate('C strong 0.8 weak 0.3, window -3..+1', ruleC(0.8, 0.3, lo=-3, hi=1))
evaluate('C strong 0.8 weak 0.3, window -2..+2', ruleC(0.8, 0.3, lo=-2, hi=2))
evaluate('C strong 0.8 weak 0.3, window 0..0', ruleC(0.8, 0.3, lo=0, hi=0))
evaluate('C strong 0.8 weak 0.3, brows weight 1.0', ruleC(0.8, 0.3, wB=1.0))
evaluate('C strong 0.8 weak 0.3, brows weight 0.6', ruleC(0.8, 0.3, wB=0.6))
vals = []
for c in te:
    aux = c['aux']; n = len(aux)
    for u in index[c['name']]['nuclei']:
        s = slice(max(0, u['t'] - 4), min(n, u['t'] + 3)); vals.append(max(aux[s, 0].max(), 0.9 * aux[s, 1].max()))
print('predicted emphasis at syllable nuclei: percentiles 10/25/50/75/90 =', np.round(np.percentile(vals, [10, 25, 50, 75, 90]), 2))
print('--- rule C grid')
for (lo, hi) in ((0, 0), (-1, 1), (-2, 1)):
    for ts in (0.7, 0.8, 0.9):
        for tw in (0.4, 0.5):
            evaluate('C window %+d..%+d strong %.1f weak %.1f' % (lo, hi, ts, tw), ruleC(ts, tw, lo=lo, hi=hi))
