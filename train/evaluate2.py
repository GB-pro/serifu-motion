"""Self-evaluation, second version. The first version rewarded copying the recorded actor; this one asks whether the
speech-driven emo is (R) read correctly, (T) in time with the speech, (L) legible as the intended emotion,
(S) moving the way animated films move rather than the way a seated actor moves, and (A) free of artifacts.

python evaluate2.py <feat_dir> <eval_dir> <model.json> <films_dir> <tracked_dir> [--reading 0.673] [--report out.json]
  eval_dir    : output of gen_eval2.mjs (per clip: .lin.f32 / .auto.f32 / .fix.f32 rig-space tracks, index.json)
  tracked_dir : face-tracker output (extract_film.py) of the RENDERED avatars: <name>.linear.npz, <name>.auto.npz
The pass marks below were fixed before the generated tracks were measured.
"""
import sys, os, json, glob, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_all, actor_baselines, apply_baseline, add_aux, NAMES, EMOS
from train import TEST_ACTORS, VAL_ACTORS
import anim_stats as AS

CRITERIA = {
    'reading_acc_min': 0.60,         # R: 6-class emotion from the voice on held-out actors (chance 0.167, human listeners on this corpus about 0.41)
    'lip_r_min': 0.50,               # T: avatar jaw vs the real actor's jaw, mean correlation over held-out clips
    'accent_lift_min': 1.5,          # T: avatar accents land on the actor's own emphasis this many times more often than chance
    'legible_ratio_min': 1.0,        # L: an observer trained on real faces reads the intended emotion from the avatar at least as often as from the actor
    'style_closer_share_min': 0.70,  # S: share of style measures on which the rendered avatar is closer to animated films than to seated actors
    'style_overact_max': 2.0,        # S: no style measure above twice the film value
    'jaw_closed_silence_min': 0.90,  # A: mouth closed on silent frames
    'max_head_step_deg': 7.0,        # A: largest frame-to-frame head rotation
}
K = 40
KM = len(NAMES)
GAIN = {'brow': 1.7, 'eye': 1.4, 'nose': 1.4, 'mouth': 1.25, 'gaze': 1.2, 'head': 1.6, 'torso': 1.4}  # LINEAR_STYLE of the app
GROUP = {'AU1': 'brow', 'AU2': 'brow', 'AU4': 'brow', 'AU5': 'eye', 'AU6': 'eye', 'AU7': 'eye', 'AU9': 'nose', 'AU45': 'eye', 'gazeX': 'gaze', 'gazeY': 'gaze'}
for n in NAMES:
    if n not in GROUP: GROUP[n] = 'head' if n.startswith('head') else 'torso' if n.startswith('torso') else 'mouth'
OBS = [i for i, n in enumerate(NAMES) if n not in ('AU45', 'gazeX', 'gazeY')]


def rig_to_tracker(R, neutral):
    """Inverse of the app's linear mapping: the tracker-space expression that the linear mapping would turn into this avatar pose."""
    out = np.zeros((len(R), KM), np.float32)
    for k, n in enumerate(NAMES):
        g = GAIN[GROUP[n]]
        out[:, k] = R[:, k] / g + (neutral[k] if k < 19 else 0.0)
    return out


def summary(Y, sp):
    m = sp if sp.sum() >= 5 else np.ones(len(Y), bool)
    Z = Y[m][:, OBS]
    return np.concatenate([Z.mean(0), np.percentile(Z, 90, axis=0)])


def speech_mask(X): return (X[:, 1] > 0.5) | (X[:, 0] > -1.5)


def train_observer(Xtr, ytr, Xva, yva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = None
    for wd in [1e-3, 1e-2, 1e-1]:
        torch.manual_seed(0)
        W = torch.zeros(Xtr.shape[1], 6, requires_grad=True); b = torch.zeros(6, requires_grad=True)
        xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(ytr)
        opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=300)
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(xt @ W + b, yt) + wd * (W ** 2).sum(); loss.backward(); return loss
        opt.step(closure)
        Wn, bn = W.detach().numpy(), b.detach().numpy()
        acc = float((np.argmax(((Xva - mu) / sd) @ Wn + bn, 1) == yva).mean())
        if best is None or acc > best[0]: best = (acc, Wn, bn, wd)
    acc, Wn, bn, wd = best
    return (lambda X: np.argmax(((X - mu) / sd) @ Wn + bn, 1)), acc, wd


def lp_speeds(seqs):
    """speeds after a 5-frame box filter (about 0.2 s): motion without tracker jitter"""
    sh, sb = [], []; k = np.ones(5) / 5
    for s in seqs:
        A, H, LR = AS.seq_channels(s)
        if len(H) < 16: continue
        Hs = np.stack([np.convolve(H[:, j], k, 'valid') for j in range(3)], 1); sh.append(np.linalg.norm(np.diff(Hs, axis=0), axis=1) * AS.FPS)
        U = np.stack([A[n] for n in ['AU1', 'AU2', 'AU4']], 1); Us = np.stack([np.convolve(U[:, j], k, 'valid') for j in range(3)], 1); sb.append(np.abs(np.diff(Us, axis=0)).max(1) * AS.FPS)
    sh, sb = np.concatenate(sh), np.concatenate(sb)
    return float(sh.mean()), float(sb.mean())


def style_measures(seqs):
    a = AS.analyze(seqs); lh, lb = lp_speeds(seqs)
    m = {'head range pitch (deg, p95)': a['amp_head_p95'][0], 'head range yaw': a['amp_head_p95'][1], 'head range roll': a['amp_head_p95'][2],
         'head speed, low-passed (deg/s)': lh, 'brow speed, low-passed (1/s)': lb,
         'brow raise inner range (AU1)': a['amp_p95']['AU1'], 'brow raise outer range (AU2)': a['amp_p95']['AU2'], 'brow lower range (AU4)': a['amp_p95']['AU4'],
         'eye widen range (AU5)': a['amp_p95']['AU5'], 'smile range (AU12)': a['amp_p95']['AU12'],
         'asymmetry brows': a['asym']['brow'], 'asymmetry brow lowering': a['asym']['browDown'], 'asymmetry smile': a['asym']['smile'],
         'head moves per minute': a['head_moves_per_min'], 'head move size (deg, median)': a['head_move_amp_median'], 'head move peak speed (deg/s)': a['head_move_peak_median'],
         'face moves per minute': a['face_moves_per_min'], 'blinks per minute': a['blink_per_min']}
    return m, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('feat_dir'); ap.add_argument('eval_dir'); ap.add_argument('model'); ap.add_argument('films_dir'); ap.add_argument('tracked_dir')
    ap.add_argument('--reading', type=float, default=None); ap.add_argument('--report')
    a = ap.parse_args()
    meta = json.load(open(a.model))['meta']; neutral = np.array(meta['neutral'], np.float32)
    clips = load_all(a.feat_dir); base, gneu = actor_baselines(clips); apply_baseline(clips, base, gneu); add_aux(clips)
    for c in clips: c['sp'] = speech_mask(c['X']); c['S'] = summary(c['Yn'], c['sp'])
    tr = [c for c in clips if c['actor'] not in TEST_ACTORS | VAL_ACTORS]; va = [c for c in clips if c['actor'] in VAL_ACTORS]; te = [c for c in clips if c['actor'] in TEST_ACTORS]
    obs, va_acc, wd = train_observer(np.stack([c['S'] for c in tr]), np.array([c['emo'] for c in tr]), np.stack([c['S'] for c in va]), np.array([c['emo'] for c in va]))
    index = {i['name']: i for i in json.load(open(os.path.join(a.eval_dir, 'index.json')))}
    te = [c for c in te if c['name'] in index and os.path.exists(os.path.join(a.eval_dir, c['name'] + '.auto.f32'))]
    R = {'criteria': CRITERIA, 'clips': len(te), 'observer': {'validation_accuracy_on_real_faces': va_acc, 'l2': wd}}
    y = np.array([c['emo'] for c in te])
    # ---------------- L: legibility ----------------
    leg = {'actor (tracked real face)': float((obs(np.stack([c['S'] for c in te])) == y).mean())}
    tracks = {}
    for kind in ['lin', 'auto', 'fix']:
        S = []
        for c in te:
            Rg = np.fromfile(os.path.join(a.eval_dir, f"{c['name']}.{kind}.f32"), np.float32).reshape(-1, K); n = min(len(Rg), len(c['Yn']))
            tracks.setdefault(c['name'], {})[kind] = Rg[:n]
            S.append(summary(rig_to_tracker(Rg[:n], neutral), c['sp'][:n]))
        pred = obs(np.stack(S)); leg[kind] = float((pred == y).mean())
        if kind == 'auto':
            read = np.array([EMOS.index(index[c['name']]['auto']['read']) if index[c['name']]['auto']['read'] else 0 for c in te])
            leg['auto, lines the network read correctly'] = float((pred[read == y] == y[read == y]).mean()); leg['reading accuracy in this run'] = float((read == y).mean())
            leg['auto, observer agrees with the emotion that was played'] = float((pred == read).mean())
        leg[kind + ' per emotion'] = {EMOS[e]: float((pred[y == e] == e).mean()) for e in range(6)}
    R['legibility'] = leg
    # ---------------- T: timing ----------------
    j = NAMES.index('AU26'); lip = {}
    for kind in ['lin', 'auto']:
        rs = []
        for c in te:
            p = tracks[c['name']][kind][:, j]; g = c['Yn'][:len(p), j]
            if p.std() > 1e-4 and g.std() > 1e-4: rs.append(np.corrcoef(p, g)[0, 1])
        lip[kind] = float(np.mean(rs))
    R['lip_sync_r'] = lip
    hits = tot = 0; chance_num = chance_den = 0
    for c in te:
        z = c['A'][:, :2]; n = len(z)
        pk = np.zeros(n, bool)
        for ch in range(2):
            v = z[:, ch]
            pk[1:-1] |= (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > 0.5)
        near = np.zeros(n, bool)
        for t in np.flatnonzero(pk): near[max(0, t - 4):t + 5] = True
        for ph in index[c['name']]['auto']['phrases']:
            a0, a1 = max(0, ph['t0']), min(n, ph['t1'])
            if a1 <= a0: continue
            chance_num += near[a0:a1].sum(); chance_den += a1 - a0
            for ac in ph['accents']:
                if 0 <= ac['t'] < n: tot += 1; hits += bool(near[ac['t']])
    hit, ch = hits / max(1, tot), chance_num / max(1, chance_den)
    R['accent_timing'] = {'accents': tot, 'on_actor_emphasis': hit, 'chance': float(ch), 'lift': float(hit / ch)}
    # ---------------- A: artifacts ----------------
    hi = [NAMES.index(n) for n in ('headPitch', 'headYaw', 'headRoll')]; art = {}
    for kind in ['lin', 'auto']:
        step = 0.0; closed = n_sil = 0
        for c in te:
            Rg = tracks[c['name']][kind]; step = max(step, float(np.abs(np.diff(Rg[:, hi], axis=0)).max()))
            sil = ~c['sp'][:len(Rg)]
            # silence well away from speech (the mouth may still be closing right after a word)
            far = sil.copy()
            for t in np.flatnonzero(~sil): far[max(0, t - 4):t + 7] = False
            closed += int((Rg[far, j] < 0.12).sum()); n_sil += int(far.sum())
        art[kind] = {'max_head_step_deg': step, 'jaw_closed_in_silence': closed / max(1, n_sil)}
    R['artifacts'] = art
    # ---------------- S: style, measured on the rendered avatars with the same face tracker ----------------
    runs = []
    for f in sorted(glob.glob(os.path.join(a.films_dir, '*', 'face.npz'))): runs += AS.film_runs(f)
    ref_f, _ = style_measures(runs); ref_a, _ = style_measures(AS.crema_seqs(a.feat_dir, n=1500))
    style = {'films': ref_f, 'actors': ref_a}
    for kind in ['linear', 'auto']:
        seqs = []
        for f in sorted(glob.glob(os.path.join(a.tracked_dir, f'*.{kind}.npz'))): seqs += AS.film_runs(f)
        if not seqs: continue
        m, full = style_measures(seqs); style[kind] = m; style[kind + '_seconds'] = full['seconds']
        # a measure can be undefined for a rendition (e.g. asymmetry of a brow lowering that never happens): it is left out
        ok = [k for k in m if m[k] is not None and ref_f[k] and ref_a[k]]
        closer = {k: (abs(np.log(max(m[k], 1e-6) / ref_f[k])) < abs(np.log(max(m[k], 1e-6) / ref_a[k]))) for k in ok}
        style[kind + '_closer_to_films'] = {k: bool(v) for k, v in closer.items()}
        style[kind + '_closer_share'] = float(np.mean(list(closer.values())))
        style[kind + '_max_ratio_to_films'] = float(max(m[k] / ref_f[k] for k in ok))
    R['style'] = style
    # ---------------- verdict ----------------
    C = CRITERIA; v = {}
    if a.reading is not None: v['R reading accuracy'] = (a.reading, a.reading >= C['reading_acc_min'])
    v['T lip sync r'] = (lip['auto'], lip['auto'] >= C['lip_r_min'])
    v['T accent lift'] = (R['accent_timing']['lift'], R['accent_timing']['lift'] >= C['accent_lift_min'])
    v['L legibility vs actor'] = (leg['auto'] / leg['actor (tracked real face)'], leg['auto'] / leg['actor (tracked real face)'] >= C['legible_ratio_min'])
    if 'auto' in style:
        v['S closer to films (share)'] = (style['auto_closer_share'], style['auto_closer_share'] >= C['style_closer_share_min'])
        v['S over-acting guard (max ratio to films)'] = (style['auto_max_ratio_to_films'], style['auto_max_ratio_to_films'] <= C['style_overact_max'])
    v['A jaw closed in silence'] = (art['auto']['jaw_closed_in_silence'], art['auto']['jaw_closed_in_silence'] >= C['jaw_closed_silence_min'])
    v['A max head step'] = (art['auto']['max_head_step_deg'], art['auto']['max_head_step_deg'] <= C['max_head_step_deg'])
    R['verdict'] = {k: {'value': float(x), 'pass': bool(p)} for k, (x, p) in v.items()}; R['pass'] = bool(all(p for _, p in v.values()))
    print(json.dumps({k: R[k] for k in ['clips', 'observer', 'legibility', 'lip_sync_r', 'accent_timing', 'artifacts']}, indent=1))
    if 'auto' in style:
        print('%-36s %9s %9s %9s %9s  closer to films' % ('style measure', 'films', 'actors', 'linear', 'anime'))
        nz = lambda v: float('nan') if v is None else v
        for k in ref_f: print('%-36s %9.3f %9.3f %9.3f %9.3f  %s' % (k, nz(ref_f[k]), nz(ref_a[k]), nz(style.get('linear', {}).get(k)), nz(style['auto'][k]), 'n/a' if k not in style['auto_closer_to_films'] else 'yes' if style['auto_closer_to_films'][k] else 'NO'))
        print('closer share: linear %.2f anime %.2f | max ratio to films: linear %.2f anime %.2f' % (style.get('linear_closer_share', float('nan')), style['auto_closer_share'], style.get('linear_max_ratio_to_films', float('nan')), style['auto_max_ratio_to_films']))
    for k, (x, p) in v.items(): print('%-44s %8.3f  %s' % (k, x, 'PASS' if p else 'FAIL'))
    print('OVERALL', 'PASS' if R['pass'] else 'FAIL')
    if a.report: json.dump(R, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
