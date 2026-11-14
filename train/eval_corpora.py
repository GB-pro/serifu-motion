"""Development-time reading of other corpora (corpus.py) by exported networks, in Python.
The numbers that count are made with the app's own code; this follows it closely (see floor_check.py) and is fast.

  python eval_corpora.py <a.json,b.json,...> <spec,spec,...> [--limit N] [--gate dB]

A single line is read by the 'line' readers only, a long recording (corpus.sessions) by all, with the 'relative'
readers at 0.4 together, as in app/core/generator.js. Reported per corpus: share of lines read correctly, the mean
of the per-emotion shares (which does not depend on how many lines each emotion has), and the answers given.
"""
import argparse, json
import numpy as np
import train as T, evalpy as EP, simnorm as SN, corpus as CP


def committee(paths):
    models = [EP.load_one(p)[0] for p in paths]; kinds = [json.load(open(p))['meta'].get('reads', 'line') for p in paths]
    nl, nr = kinds.count('line'), kinds.count('relative')
    w_one = [1 / nl if k == 'line' else 0.0 for k in kinds] if nl else [1 / len(kinds)] * len(kinds)
    w_long = [(0.6 if nr else 1.0) / nl if k == 'line' else 0.4 / nr for k in kinds] if nl else w_one
    return models, w_one, w_long


def read(models, w, xs):
    """Lines of one recording joined with pauses -> emotion probabilities of every line."""
    gapx = np.repeat(T.SIL[None], T.GAP_EVAL, 0); parts = []; spans = []; t = 0
    for x in xs: parts += [gapx, x]; spans.append((t + T.GAP_EVAL, t + T.GAP_EVAL + len(x))); t += T.GAP_EVAL + len(x)
    xx = np.concatenate(parts + [gapx]).astype(np.float32); P = 0
    for m, wm in zip(models, w):
        if not wm: continue
        lg = T.run_padded(m, xx, 0, pad=30)[:, T.K:T.K + T.NE]; e = np.exp(lg - lg.max(1, keepdims=True)); P = P + wm * e / e.sum(1, keepdims=True)
    out = []
    for (a, b), x in zip(spans, xs):
        p = P[a:b]; sp = (x[:, 1] > 0.5) | (x[:, 0] > -1.5); out.append((p[sp] if sp.any() else p).mean(0))
    return np.array(out)


def evaluate(models, w_one, w_long, lines, prep=None):
    """-> confusion matrices (rows acted, columns read) for single lines and for long recordings."""
    C1 = np.zeros((T.NE, T.NE), int); C2 = np.zeros((T.NE, T.NE), int)
    for rec in CP.sessions(lines):
        raws = [prep(c['R']) if prep else c['R'] for c in rec]
        for c, r in zip(rec, raws):
            C1[c['emo'], read(models, w_one, [SN.normalize(r, SN.speaker_stats([r]))])[0].argmax()] += 1
        if len(rec) > 1:
            st = SN.speaker_stats(raws)
            for c, p in zip(rec, read(models, w_long, [SN.normalize(r, st) for r in raws])): C2[c['emo'], p.argmax()] += 1
    return C1, C2


def summary(C):
    n = C.sum(); rows = [i for i in range(T.NE) if C[i].sum()]
    if not n: return 'no lines'
    rec = [C[i, i] / C[i].sum() for i in rows]
    return (f'{np.trace(C) / n:.3f} (mean over emotions {np.mean(rec):.3f}, n {n})  ' + ' '.join(f'{CP.EMOS[i]} {r:.2f}' for i, r in zip(rows, rec))
            + '  | answers ' + ' '.join(f'{CP.EMOS[j]} {C[:, j].sum() / n:.2f}' for j in range(T.NE)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('models'); ap.add_argument('specs'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    models, w_one, w_long = committee(a.models.split(','))
    for spec in a.specs.split(','):
        lines = CP.load(spec, limit=a.limit or None)
        C1, C2 = evaluate(models, w_one, w_long, lines)
        print(f'{spec}: speakers {len(set(c["speaker"] for c in lines))}', flush=True)
        print(f'  single line     {summary(C1)}', flush=True)
        if C2.sum(): print(f'  long recording  {summary(C2)}', flush=True)
