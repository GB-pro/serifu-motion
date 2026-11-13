"""The app's speaker normalisation, ported to numpy, so that training can normalise every sequence the way the app
would normalise the recording it comes from (a single line, a few lines, lines in one emotion, a long recording).

The app measures a speaker's pitch, level and timbre from the recording it is given (app/core/features.js:
speakerStats, normalize, globalFeatures). The first models were trained only with statistics of all lines of an actor,
which is the easy case. `python simnorm.py <feat_dir>` checks this port against the files written by feats.mjs.
"""
import os, glob, json, math
import numpy as np

RAW_DIM, FEAT_DIM, NMFCC = 19, 20, 12

# Recording-level inputs ("globals"): what the recording as a whole sounds like, in absolute terms. Same definition as
# globalFeatures() in app/core/features.js. Centre and scale are rounded population values of the corpus (audio statistics only).
G_F0, G_SPREAD, G_AMOUNT, G_CENT, G_CENTSD = 0, 1, 2, 3, 4
G_MFMEAN, G_MFSTD, G_FLUXSD = 5, 17, 29
NG = 30
POP = dict(
    f0Median=89.7, f0MedianS=5.0, f0Spread=7.5, f0SpreadS=2.0, centMean=520.0, centMeanS=150.0, centStd=620.0, centStdS=230.0,
    mfMean=[35.0, 12.0, 8.5, 0.0, -0.5, 0.0, -1.5, 0.5, -0.5, -0.5, -0.5, -1.5], mfMeanS=[4.0, 5.0, 3.0, 3.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5],
    mfStd=[14.5, 13.5, 8.5, 8.5, 6.5, 5.5, 5.0, 4.0, 4.0, 3.5, 3.5, 3.5], mfStdS=[2.5, 2.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75, 0.75],
    fluxStd=0.95, fluxStdS=0.15)


def _q(sorted_arr, p):
    n = len(sorted_arr)
    return float(sorted_arr[min(n - 1, int(math.floor(p * n)))]) if n else 0.0


def speaker_stats(raws):
    """raws: list of float arrays (n, 19). Returns the same dictionary as speakerStats() in features.js."""
    raw = np.concatenate(raws).astype(np.float64) if len(raws) > 1 else np.asarray(raws[0], np.float64)
    e = raw[:, 0]; f0 = raw[:, 1]
    st_f0 = np.sort(np.log2(f0[f0 > 0]) * 12)
    ev = np.sort(e[f0 > 0])  # speech level from the voiced frames only (see speakerStats in features.js)
    e_hi = _q(ev, 0.7) + 0.5 if len(ev) else max(_q(np.sort(e), 0.9), -40.0)
    sp = e > e_hi - 25
    mf = raw[sp, 6:18]; cent = raw[sp, 4]; flux = raw[sp, 3]
    n = len(cent)

    def mean(a): return a.mean(0) if n else np.zeros(a.shape[1:])

    def std(a, m): return np.sqrt(((a - m) ** 2).mean(0)) + 1e-6 if n > 1 else np.ones(a.shape[1:])
    mm = mean(mf); cm = float(mean(cent)); fm = float(mean(flux))
    return dict(
        eSpeech=e_hi,
        f0Median=_q(st_f0, 0.5) if len(st_f0) else 12 * math.log2(160),
        f0Spread=max(1.5, (_q(st_f0, 0.9) - _q(st_f0, 0.1)) / 2) if len(st_f0) > 10 else 3.0,
        mfMean=mm, mfStd=std(mf, mm), centMean=cm, centStd=float(std(cent, cm)), fluxMean=fm, fluxStd=float(std(flux, fm)),
        nVoiced=int(len(st_f0)))


def normalize(raw, st):
    """raw (n, 19) -> features (n, 20), as normalize() in features.js."""
    raw = np.asarray(raw, np.float64); n = len(raw)
    F = np.zeros((n, FEAT_DIM), np.float32)
    v = raw[:, 1] > 0
    f = np.zeros(n); f[v] = (np.log2(raw[v, 1]) * 12 - st['f0Median']) / st['f0Spread']
    F[:, 0] = np.clip((raw[:, 0] - st['eSpeech']) / 10, -4, 2)
    F[:, 1] = v
    F[:, 2] = np.clip(f, -4, 4)
    iv = np.flatnonzero(v)
    if len(iv) > 1: F[iv[1:], 3] = np.clip(np.diff(f[iv]), -2, 2)
    e32 = F[:, 0].astype(np.float64)
    F[:, 4] = np.clip(e32 - np.concatenate([[0.0], e32[:-1]]), -2, 2)
    F[:, 5] = np.clip((raw[:, 3] - st['fluxMean']) / st['fluxStd'], -3, 3)
    F[:, 6] = np.clip((raw[:, 4] - st['centMean']) / st['centStd'], -3, 3)
    F[:, 7] = np.minimum(3, raw[:, 5] * 10)
    gate = np.where(F[:, 0] > -2.5, 1.0, 0.2)
    F[:, 8:20] = gate[:, None] * np.clip((raw[:, 6:18] - np.asarray(st['mfMean'])) / np.asarray(st['mfStd']), -3, 3)
    return F


def global_features(st):
    """Recording-level inputs (NG values), as globalFeatures() in features.js."""
    g = np.zeros(NG, np.float32); P = POP
    g[G_F0] = (st['f0Median'] - P['f0Median']) / P['f0MedianS']
    g[G_SPREAD] = (st['f0Spread'] - P['f0Spread']) / P['f0SpreadS']
    g[G_AMOUNT] = min(1.0, math.log2(1 + st.get('nVoiced', 0) / 75) / 6)
    g[G_CENT] = (st['centMean'] - P['centMean']) / P['centMeanS']
    g[G_CENTSD] = (st['centStd'] - P['centStd']) / P['centStdS']
    g[G_MFMEAN:G_MFMEAN + NMFCC] = (np.asarray(st['mfMean']) - P['mfMean']) / P['mfMeanS']
    g[G_MFSTD:G_MFSTD + NMFCC] = (np.asarray(st['mfStd']) - P['mfStd']) / P['mfStdS']
    g[G_FLUXSD] = (st['fluxStd'] - P['fluxStd']) / P['fluxStdS']
    return np.clip(g, -4, 4)


def load_raw(feat_dir, name):
    a = np.fromfile(os.path.join(feat_dir, name + '.raw.f32'), np.float32)
    return a[:len(a) // RAW_DIM * RAW_DIM].reshape(-1, RAW_DIM)


if __name__ == '__main__':
    import sys
    d = sys.argv[1]; worst = 0; worst_s = 0; n = 0
    for sj in sorted(glob.glob(os.path.join(d, '*.stats.json')))[::6]:
        a = os.path.basename(sj)[:4]; ref = json.load(open(sj))
        names = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, a + '_*.raw.f32')))
        raws = [load_raw(d, nm) for nm in names]
        st = speaker_stats(raws)
        for k in ('eSpeech', 'f0Median', 'f0Spread', 'centMean', 'centStd', 'fluxMean', 'fluxStd', 'mfMean', 'mfStd'):
            worst_s = max(worst_s, float(np.max(np.abs(np.asarray(st[k]) - np.asarray(ref[k])))))
        for nm, r in zip(names, raws):
            F = np.fromfile(os.path.join(d, nm + '.feat.f32'), np.float32).reshape(-1, FEAT_DIM)
            worst = max(worst, float(np.max(np.abs(normalize(r, st) - F)))); n += 1
    print('speakers checked', len(glob.glob(os.path.join(d, '*.stats.json'))[::6]), 'clips', n, 'largest difference: statistics %.2e, features %.2e' % (worst_s, worst))
