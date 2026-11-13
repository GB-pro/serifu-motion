"""Self-evaluation of generated emo against the held-out actors' real (tracked) motion.
Usage: python evaluate.py <feat_dir> <gen_dir> <model_json> [--report out.json]
Pass criteria are fixed before looking at results (see CRITERIA).
"""
import sys, os, json, argparse
import numpy as np
from dataset import load_all, actor_baselines, apply_baseline, NAMES, EMOS, CHANNELS
from train import TEST_ACTORS

K = len(NAMES)
LIP = ['AU26']
EXPR = ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU9', 'headPitch', 'headYaw', 'headRoll']
GROUPS = {'brow': ['AU1', 'AU2', 'AU4'], 'eye': ['AU5', 'AU6', 'AU7'], 'mouth': ['AU10', 'AU12', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU20', 'AU22', 'AU24', 'AU26', 'AU28'],
          'head': ['headPitch', 'headYaw', 'headRoll'], 'torso': ['torsoRoll', 'torsoShift', 'torsoShrug', 'torsoLean'], 'gaze': ['gazeX', 'gazeY']}
CRITERIA = {
    'lip_r_min': 0.50,              # jaw-open correlation with the real actor (lip sync)
    'expr_r_gain_min': 0.05,        # expression/head correlation must beat a same-sentence/same-emotion shuffled baseline by this much
    'emotion_acc_min': 0.40,        # 6-class emotion from audio (chance 0.167; CREMA-D human audio-only raters ~0.41)
    'profile_cos_min': 0.80,        # emotion-specific AU pattern (generated vs real), mean over 5 non-neutral emotions
    'dyn_ratio': (0.75, 1.33),      # amount of motion (std) and speed (|velocity|) relative to the real actor, per group
    'blink_rate_tol': 0.25,         # blink rate within ±25%
}


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def resample(y, n):
    if len(y) == n: return y
    x = np.linspace(0, len(y) - 1, n)
    return np.stack([np.interp(x, np.arange(len(y)), y[:, k]) for k in range(y.shape[1])], 1)


def blink_rate(Y):
    b = Y[:, NAMES.index('AU45')]
    return ((b[1:] > 0.45) & (b[:-1] <= 0.45)).sum() / (len(b) / 30) * 60


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('feat_dir'); ap.add_argument('gen_dir'); ap.add_argument('model'); ap.add_argument('--report')
    a = ap.parse_args()
    clips = load_all(a.feat_dir)
    base, gneu = actor_baselines(clips); apply_baseline(clips, base, gneu)
    test = [c for c in clips if c['actor'] in TEST_ACTORS]
    G, P, E = [], [], []
    by_key = {}
    for c in test:
        f = os.path.join(a.gen_dir, c['name'] + '.gen.f32')
        if not os.path.exists(f): continue
        p = np.fromfile(f, np.float32).reshape(-1, K)
        e = np.fromfile(os.path.join(a.gen_dir, c['name'] + '.emo.f32'), np.float32).reshape(-1, 6)
        n = min(len(p), len(c['Yn']))
        c['P'], c['E'] = p[:n], e[:n]; c['G'] = c['Yn'][:n]
        by_key.setdefault((c['sent'], c['emo']), []).append(c)
    test = [c for c in test if 'P' in c]
    Gc = np.concatenate([c['G'] for c in test]); Pc = np.concatenate([c['P'] for c in test])
    # shuffled baseline: prediction from ANOTHER actor saying the same sentence with the same emotion (time-normalized)
    Sh = []
    for c in test:
        others = [o for o in by_key[(c['sent'], c['emo'])] if o['actor'] != c['actor']]
        o = others[hash(c['name']) % len(others)] if others else c
        Sh.append(resample(o['P'], len(c['G'])))
    Sc = np.concatenate(Sh)
    r = {n: pearson(Pc[:, k], Gc[:, k]) for k, n in enumerate(NAMES)}
    rs = {n: pearson(Sc[:, k], Gc[:, k]) for k, n in enumerate(NAMES)}
    # within-clip correlation (motion timing inside each line), averaged
    r_in = {}
    for k, n in enumerate(NAMES):
        v = [pearson(c['P'][:, k], c['G'][:, k]) for c in test if c['G'][:, k].std() > 1e-4 and c['P'][:, k].std() > 1e-4]
        r_in[n] = float(np.mean(v)) if v else 0.0
    # emotion recognition (clip level, voiced frames)
    acc = []
    conf = np.zeros((6, 6), int)
    for c in test:
        v = c['X'][:len(c['E']), 1] > 0.5
        m = c['E'][v].mean(0) if v.any() else c['E'].mean(0)
        conf[c['emo'], int(m.argmax())] += 1; acc.append(int(m.argmax()) == c['emo'])
    # emotion profiles
    prof_g, prof_p = [], []
    gneu_m = np.concatenate([c['G'] for c in test if c['emo'] == 0]).mean(0)
    pneu_m = np.concatenate([c['P'] for c in test if c['emo'] == 0]).mean(0)
    sel = [NAMES.index(n) for n in NAMES[:19]] + [NAMES.index('headPitch')]
    cos = {}
    for e in range(1, 6):
        g = np.concatenate([c['G'] for c in test if c['emo'] == e]).mean(0) - gneu_m
        p = np.concatenate([c['P'] for c in test if c['emo'] == e]).mean(0) - pneu_m
        sg = Gc.std(0)[sel] + 1e-6
        gv, pv = g[sel] / sg, p[sel] / sg
        cos[EMOS[e]] = float(gv @ pv / (np.linalg.norm(gv) * np.linalg.norm(pv) + 1e-9))
    # dynamics
    dyn = {}
    for gname, chans in GROUPS.items():
        idx = [NAMES.index(n) for n in chans]
        sr = float(np.mean([np.mean([c['P'][:, k].std() for c in test]) / (np.mean([c['G'][:, k].std() for c in test]) + 1e-9) for k in idx]))
        vr = float(np.mean([np.mean([np.abs(np.diff(c['P'][:, k])).mean() for c in test]) / (np.mean([np.abs(np.diff(c['G'][:, k])).mean() for c in test]) + 1e-9) for k in idx]))
        dyn[gname] = {'std_ratio': round(sr, 3), 'vel_ratio': round(vr, 3)}
    br_g, br_p = blink_rate(Gc), blink_rate(Pc)
    expr_gain = float(np.mean([r[n] - rs[n] for n in EXPR]))
    res = {
        'n_clips': len(test), 'actors': sorted(set(c['actor'] for c in test)),
        'lip_r': {n: round(r[n], 3) for n in ['AU26', 'AU25'] if n in r},
        'r_global': {n: round(v, 3) for n, v in r.items()}, 'r_shuffled': {n: round(v, 3) for n, v in rs.items()},
        'r_within_clip': {n: round(v, 3) for n, v in r_in.items()},
        'expr_r_mean': round(float(np.mean([r[n] for n in EXPR])), 3), 'expr_r_shuffled_mean': round(float(np.mean([rs[n] for n in EXPR])), 3), 'expr_r_gain': round(expr_gain, 3),
        'emotion_acc': round(float(np.mean(acc)), 3), 'emotion_confusion': conf.tolist(),
        'profile_cos': {k: round(v, 3) for k, v in cos.items()}, 'profile_cos_mean': round(float(np.mean(list(cos.values()))), 3),
        'dynamics': dyn, 'blink_rate': {'generated': round(br_p, 1), 'real': round(br_g, 1)},
    }
    lo, hi = CRITERIA['dyn_ratio']
    checks = {
        'lip_sync': res['lip_r']['AU26'] >= CRITERIA['lip_r_min'],
        'expression_timing': expr_gain >= CRITERIA['expr_r_gain_min'],
        'emotion_recognition': res['emotion_acc'] >= CRITERIA['emotion_acc_min'],
        'emotion_profiles': res['profile_cos_mean'] >= CRITERIA['profile_cos_min'],
        'motion_amount': all(lo <= d['std_ratio'] <= hi and lo <= d['vel_ratio'] <= hi for d in dyn.values()),
        'blink_rate': abs(br_p - br_g) <= CRITERIA['blink_rate_tol'] * br_g,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    res['checks'] = checks; res['passed'] = bool(all(checks.values())); res['criteria'] = CRITERIA
    print(json.dumps({k: res[k] for k in ['n_clips', 'lip_r', 'expr_r_mean', 'expr_r_shuffled_mean', 'expr_r_gain', 'emotion_acc', 'profile_cos', 'profile_cos_mean', 'dynamics', 'blink_rate', 'checks', 'passed']}, indent=1, ensure_ascii=False))
    if a.report: json.dump(res, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
