"""Build training arrays: audio features (from feats.mjs) + emo targets (AU/head/torso) per clip."""
import os, glob, json
import numpy as np

# Output channels of the emo model (shared with app/core/au.js)
CHANNELS = [
    ('AU1', ['browInnerUp']),
    ('AU2', ['browOuterUpLeft', 'browOuterUpRight']),
    ('AU4', ['browDownLeft', 'browDownRight']),
    ('AU5', ['eyeWideLeft', 'eyeWideRight']),
    ('AU6', ['cheekSquintLeft', 'cheekSquintRight']),
    ('AU7', ['eyeSquintLeft', 'eyeSquintRight']),
    ('AU9', ['noseSneerLeft', 'noseSneerRight']),
    ('AU10', ['mouthUpperUpLeft', 'mouthUpperUpRight']),
    ('AU12', ['mouthSmileLeft', 'mouthSmileRight']),
    ('AU14', ['mouthDimpleLeft', 'mouthDimpleRight']),
    ('AU15', ['mouthFrownLeft', 'mouthFrownRight']),
    ('AU16', ['mouthLowerDownLeft', 'mouthLowerDownRight']),
    ('AU17', ['mouthShrugLower']),
    ('AU18', ['mouthPucker']),
    ('AU20', ['mouthStretchLeft', 'mouthStretchRight']),
    ('AU22', ['mouthFunnel']),
    ('AU24', ['mouthPressLeft', 'mouthPressRight']),
    ('AU26', ['jawOpen']),
    ('AU28', ['mouthRollLower', 'mouthRollUpper']),
    ('AU45', ['eyeBlinkLeft', 'eyeBlinkRight']),
]
EXTRA = ['gazeX', 'gazeY', 'headPitch', 'headYaw', 'headRoll', 'headZ', 'torsoRoll', 'torsoShift', 'torsoShrug', 'torsoLean']
NAMES = [c[0] for c in CHANNELS] + EXTRA
EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
AUX = ['accHead', 'accBrow', 'intensity']  # learned "when/how strongly" signals for the animation layer
LEVELS = {'LO': 0.0, 'MD': 0.5, 'HI': 1.0}
FEAT_DIM = 20


def interp_nan(a):
    a = a.copy()
    for j in range(a.shape[1]):
        col = a[:, j]
        m = np.isfinite(col)
        if m.sum() == 0:
            col[:] = 0
        elif (~m).any():
            col[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), col[m])
    return a


def smooth(x, k=3):
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    return np.stack([np.convolve(np.pad(x[:, j], (k // 2, k // 2), mode='edge'), ker, 'valid') for j in range(x.shape[1])], 1)


def targets(d):
    names = list(d['bs_names'])
    bs = interp_nan(d['bs'])
    idx = {n: i for i, n in enumerate(names)}
    T = [bs[:, [idx[n] for n in src]].mean(1) for _, src in CHANNELS]
    gx = (bs[:, idx['eyeLookOutLeft']] + bs[:, idx['eyeLookInRight']] - bs[:, idx['eyeLookInLeft']] - bs[:, idx['eyeLookOutRight']]) / 2
    gy = (bs[:, idx['eyeLookUpLeft']] + bs[:, idx['eyeLookUpRight']] - bs[:, idx['eyeLookDownLeft']] - bs[:, idx['eyeLookDownRight']]) / 2
    head = interp_nan(d['head'])
    torso = smooth(interp_nan(d['torso'][:, :4]), 5)
    deg = np.degrees
    E = [gx, gy, deg(head[:, 0]), deg(head[:, 1]), deg(head[:, 2]), head[:, 5],
         deg(torso[:, 0]), torso[:, 1] * 100, torso[:, 2] * 100, torso[:, 3] * 100]
    return np.stack(T + E, 1).astype(np.float32)


def load_all(feat_dir):
    clips = []
    for f in sorted(glob.glob(os.path.join(feat_dir, '*.npz'))):
        name = os.path.basename(f)[:-4]
        ff = os.path.join(feat_dir, name + '.feat.f32')
        if not os.path.exists(ff):
            continue
        d = np.load(f)
        if d['ok'].mean() < 0.8 or len(d['bs_names']) == 0:
            continue
        X = np.fromfile(ff, np.float32)
        if X.size % FEAT_DIM:
            continue
        X = X.reshape(-1, FEAT_DIM)
        Y = targets(d)
        n = min(len(X), len(Y))
        if n < 20:
            continue
        actor, sent, emo, lvl = name.split('_')
        clips.append(dict(name=name, actor=int(actor), sent=sent, emo=EMOS.index(emo), lvl=lvl, X=X[:n], Y=Y[:n]))
    return clips


def actor_baselines(clips):
    """Per-actor neutral baseline: removes identity/camera bias from AUs, head, torso."""
    base = {}
    for a in sorted(set(c['actor'] for c in clips)):
        neu = [c['Y'] for c in clips if c['actor'] == a and c['emo'] == 0]
        allc = [c['Y'] for c in clips if c['actor'] == a]
        ref = np.concatenate(neu if neu else allc)
        base[a] = np.median(ref, 0)
    gneu = np.median(np.stack(list(base.values())), 0)
    return base, gneu


def _hann_smooth(x, k=5):
    w = np.hanning(k + 2)[1:-1]; w /= w.sum()
    return np.convolve(np.pad(x, (k // 2, k // 2), mode='edge'), w, 'valid')


def accent_envelopes(Yn):
    """Motion-energy envelopes of the real performance: where the actor accents with head / brows."""
    hi = [NAMES.index(n) for n in ('headPitch', 'headYaw', 'headRoll')]
    bi = [NAMES.index(n) for n in ('AU1', 'AU2', 'AU4')]
    hs = np.linalg.norm(np.diff(Yn[:, hi], axis=0, prepend=Yn[:1, hi]), axis=1)
    bs = np.abs(np.diff(Yn[:, bi], axis=0, prepend=Yn[:1, bi])).sum(1)
    return np.stack([np.log1p(2 * _hann_smooth(hs)), np.log1p(20 * _hann_smooth(bs))], 1).astype(np.float32)


def add_aux(clips):
    """Aux targets per clip: accent envelopes (z-scored per actor) and acted intensity level where labelled."""
    by = {}
    for c in clips:
        c['acc'] = accent_envelopes(c['Yn']); by.setdefault(c['actor'], []).append(c)
    for a, cs in by.items():
        allv = np.concatenate([c['acc'] for c in cs]); mu, sd = allv.mean(0), allv.std(0) + 1e-6
        for c in cs:
            z = (c['acc'] - mu) / sd
            lvl = LEVELS.get(c['lvl'])
            inten = np.full((len(z), 1), lvl if lvl is not None else 0.0, np.float32)
            c['A'] = np.concatenate([z, inten], 1).astype(np.float32)
            c['AM'] = np.tile(np.array([1, 1, 1 if lvl is not None else 0], np.float32), (len(z), 1))
    return clips


def apply_baseline(clips, base, gneu):
    n_au = len(CHANNELS)
    for c in clips:
        Y = c['Y'].copy()
        b = base[c['actor']]
        # AUs: actor-relative + global neutral; blink untouched
        Y[:, :n_au - 1] = np.clip(Y[:, :n_au - 1] - b[:n_au - 1] + gneu[:n_au - 1], 0, 1)
        # gaze/head/torso: relative to actor neutral (motion + emotion-driven offsets)
        Y[:, n_au:] = Y[:, n_au:] - b[n_au:]
        c['Yn'] = Y
    return clips
