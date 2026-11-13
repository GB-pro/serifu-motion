"""Animation-style statistics: animated films (face runs inside continuous shots) vs. recorded actors (CREMA-D).
Only numbers are produced. Usage: python anim_stats.py <films_dir> <crema_feat_dir> <out.json>
Everything is compared at 24 fps with the same light smoothing, so tracker jitter affects both sides alike.
"""
import os, sys, glob, json
import numpy as np

FPS = 24.0
AU = {
    'AU1': ['browInnerUp'], 'AU2': ['browOuterUpLeft', 'browOuterUpRight'], 'AU4': ['browDownLeft', 'browDownRight'],
    'AU5': ['eyeWideLeft', 'eyeWideRight'], 'AU7': ['eyeSquintLeft', 'eyeSquintRight'],
    'AU12': ['mouthSmileLeft', 'mouthSmileRight'], 'AU15': ['mouthFrownLeft', 'mouthFrownRight'], 'AU26': ['jawOpen'],
    'AU45': ['eyeBlinkLeft', 'eyeBlinkRight'],
}
UPPER = ['AU1', 'AU2', 'AU4', 'AU5']
PAIRS = {'brow': ('browOuterUpLeft', 'browOuterUpRight'), 'browDown': ('browDownLeft', 'browDownRight'), 'smile': ('mouthSmileLeft', 'mouthSmileRight')}


def smooth3(x):
    if len(x) < 3: return x
    p = np.pad(x, ((1, 1), (0, 0)), mode='edge')
    return (p[:-2] + 2 * p[1:-1] + p[2:]) / 4


def resample(x, fps_in, fps_out=FPS):
    if abs(fps_in - fps_out) < 1e-6: return x
    n = int(np.floor((len(x) - 1) * fps_out / fps_in)) + 1
    t = np.arange(n) * fps_in / fps_out
    i = np.clip(np.floor(t).astype(int), 0, len(x) - 2); f = (t - i)[:, None]
    return x[i] * (1 - f) + x[i + 1] * f


def film_runs(npz, min_len=20, min_face=0.07):
    d = np.load(npz, allow_pickle=True)
    bs, head, box, cut = d['bs'], d['head'], d['box'], d['cut']
    names = [str(n) for n in d['bs_names']]
    ok = np.isfinite(bs[:, 0])
    w = box[:, 2] - box[:, 0]; cx = (box[:, 0] + box[:, 2]) / 2; cy = (box[:, 1] + box[:, 3]) / 2
    med = np.median(cut[cut > 0]) if (cut > 0).any() else 1.0
    brk = (cut > max(14.0, 6 * med))
    runs = []; s = None
    for i in range(len(ok) + 1):
        good = i < len(ok) and ok[i] and w[i] >= min_face
        if good and s is not None:
            jump = abs(cx[i] - cx[i - 1]) > 0.06 or abs(cy[i] - cy[i - 1]) > 0.06 or abs(w[i] / w[i - 1] - 1) > 0.2 or brk[i]
            if jump:
                if i - s >= min_len: runs.append((s, i))
                s = i
        elif good and s is None: s = i
        elif not good and s is not None:
            if i - s >= min_len: runs.append((s, i))
            s = None
    return [dict(bs=bs[a:b], head=head[a:b], names=names, fps=float(d['fps'])) for a, b in runs]


def crema_seqs(feat_dir, n=1500, seed=0):
    files = sorted(glob.glob(os.path.join(feat_dir, '*.npz')))
    rng = np.random.default_rng(seed); rng.shuffle(files)
    out = []
    for f in files[:n]:
        d = np.load(f, allow_pickle=True)
        if len(d['bs_names']) == 0 or d['ok'].mean() < 0.95: continue
        bs = d['bs'].astype(np.float32); head = d['head'].astype(np.float32)
        if not np.isfinite(bs).all() or not np.isfinite(head).all(): continue
        out.append(dict(bs=bs, head=head, names=[str(x) for x in d['bs_names']], fps=30.0, emo=os.path.basename(f).split('_')[2]))
    return out


def seq_channels(s):
    idx = {n: i for i, n in enumerate(s['names'])}
    bs = resample(s['bs'], s['fps']); head = resample(s['head'], s['fps'])
    A = {k: smooth3(bs[:, [idx[n] for n in v]].mean(1, keepdims=True))[:, 0] for k, v in AU.items()}
    H = smooth3(np.degrees(np.unwrap(head[:, :3], axis=0)))
    LR = {k: (smooth3(bs[:, [idx[a]]])[:, 0], smooth3(bs[:, [idx[b]]])[:, 0]) for k, (a, b) in PAIRS.items()}
    return A, H, LR


def moves(speed, hi, lo):
    """segments where speed exceeds hi, extended while above lo -> list of (start, end)"""
    out = []; i = 0; n = len(speed)
    while i < n:
        if speed[i] > hi:
            a = i
            while a > 0 and speed[a - 1] > lo: a -= 1
            b = i
            while b < n - 1 and speed[b + 1] > lo: b += 1
            out.append((a, b + 1)); i = b + 1
        else: i += 1
    return out


def kurt(x):
    x = np.asarray(x, np.float64); x = x - x.mean(); v = (x ** 2).mean()
    return float((x ** 4).mean() / (v ** 2 + 1e-12) - 3)


def analyze(seqs):
    acc = dict(frames=0, hold_face=0, hold_head=0, hold_both=0, vel_head=[], vel_brow=[], amp={k: [] for k in AU}, amp_head=[],
               mv_dur=[], mv_amp=[], mv_peak=[], mv_over=[], fmv_dur=[], fmv_amp=[], blink_dur=[], blink_n=0, asym={k: [] for k in PAIRS},
               hold_len=[], active_brow=0)
    for s in seqs:
        A, H, LR = seq_channels(s)
        n = len(H)
        if n < 12: continue
        dH = np.diff(H, axis=0) * FPS; sp_h = np.linalg.norm(dH, axis=1)
        U = np.stack([A[k] for k in UPPER], 1); dU = np.diff(U, axis=0) * FPS; sp_u = np.abs(dU).max(1)
        acc['frames'] += n - 1
        hf = sp_u < 0.35; hh = sp_h < 6.0
        acc['hold_face'] += int(hf.sum()); acc['hold_head'] += int(hh.sum()); acc['hold_both'] += int((hf & hh).sum())
        # lengths of holds (face and head both still)
        run = 0
        for v in (hf & hh):
            if v: run += 1
            elif run: acc['hold_len'].append(run / FPS); run = 0
        if run: acc['hold_len'].append(run / FPS)
        acc['vel_head'].append(dH); acc['vel_brow'].append(dU[:, :3])
        for k in AU: acc['amp'][k].append(np.abs(A[k] - np.median(A[k])))
        acc['amp_head'].append(np.abs(H - np.median(H, axis=0)))
        # head moves
        for a, b in moves(sp_h, 18.0, 8.0):
            if b - a < 2 or a < 1 or b > n - 3: continue
            d = H[b] - H[a]; amp = float(np.linalg.norm(d))
            if amp < 2.0: continue
            acc['mv_dur'].append((b - a) / FPS); acc['mv_amp'].append(amp); acc['mv_peak'].append(float(sp_h[a:b].max()))
            # overshoot: does the head come back after the move by >12% of the move, within 0.35 s?
            e = min(n, b + int(0.35 * FPS) + 1)
            if e - b >= 4:
                u = d / (amp + 1e-9); proj = (H[b:e] - H[b]) @ u
                acc['mv_over'].append(float(-proj.min() / amp))
        # facial (brow/eye) moves
        for a, b in moves(sp_u, 1.2, 0.5):
            if b - a < 1 or a < 1 or b > n - 2: continue
            amp = float(np.abs(U[b] - U[a]).max())
            if amp < 0.12: continue
            acc['fmv_dur'].append((b - a) / FPS); acc['fmv_amp'].append(amp)
        # blinks
        bl = A['AU45'] > 0.5; i = 0
        while i < n:
            if bl[i]:
                j = i
                while j < n and bl[j]: j += 1
                if i > 0 and j < n and (j - i) / FPS < 0.6: acc['blink_dur'].append((j - i) / FPS); acc['blink_n'] += 1
                i = j
            else: i += 1
        for k, (l, r) in LR.items():
            m = (l + r) / 2; on = m > 0.25
            if on.any(): acc['asym'][k].append(np.abs(l - r)[on] / (m[on] + 1e-6))
    F = max(1, acc['frames'])
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    vh = np.concatenate(acc['vel_head']) if acc['vel_head'] else np.zeros((1, 3)); vb = np.concatenate(acc['vel_brow']) if acc['vel_brow'] else np.zeros((1, 3))
    out = dict(
        seconds=F / FPS, sequences=len(seqs),
        hold_ratio_face=acc['hold_face'] / F, hold_ratio_head=acc['hold_head'] / F, hold_ratio_both=acc['hold_both'] / F,
        hold_len_median=q(acc['hold_len'], 50), hold_len_p90=q(acc['hold_len'], 90),
        head_speed_mean=float(np.linalg.norm(vh, axis=1).mean()), head_speed_p95=q(np.linalg.norm(vh, axis=1), 95),
        head_vel_kurtosis=float(np.mean([kurt(vh[:, j]) for j in range(3)])), brow_vel_kurtosis=float(np.mean([kurt(vb[:, j]) for j in range(3)])),
        brow_speed_mean=float(np.abs(vb).mean()), brow_speed_p95=q(np.abs(vb), 95),
        amp_p95={k: q(np.concatenate(v), 95) if v else None for k, v in acc['amp'].items()},
        amp_head_p95=[q(np.concatenate(acc['amp_head'])[:, j], 95) for j in range(3)] if acc['amp_head'] else None,
        head_moves_per_min=len(acc['mv_dur']) / (F / FPS) * 60,
        head_move_dur_median=q(acc['mv_dur'], 50), head_move_amp_median=q(acc['mv_amp'], 50), head_move_amp_p90=q(acc['mv_amp'], 90),
        head_move_peak_median=q(acc['mv_peak'], 50),
        head_move_overshoot_share=float(np.mean(np.array(acc['mv_over']) > 0.12)) if acc['mv_over'] else None,
        head_move_overshoot_median=q(acc['mv_over'], 50),
        face_moves_per_min=len(acc['fmv_dur']) / (F / FPS) * 60, face_move_dur_median=q(acc['fmv_dur'], 50), face_move_amp_median=q(acc['fmv_amp'], 50),
        blink_per_min=acc['blink_n'] / (F / FPS) * 60, blink_dur_median=q(acc['blink_dur'], 50),
        asym={k: (float(np.concatenate(v).mean()) if v else None) for k, v in acc['asym'].items()},
    )
    return out


def main():
    films_dir, crema_dir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    res = {'films': {}, 'fps': FPS}
    allruns = []
    for f in sorted(glob.glob(os.path.join(films_dir, '*', 'face.npz'))):
        name = os.path.basename(os.path.dirname(f)); runs = film_runs(f)
        res['films'][name] = analyze(runs); allruns += runs
        print(name, 'runs', len(runs), 'sec %.0f' % res['films'][name]['seconds'], flush=True)
    res['animation'] = analyze(allruns)
    cs = crema_seqs(crema_dir)
    res['actors'] = analyze(cs)
    for e in ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']:
        res['actors_' + e] = analyze([c for c in cs if c['emo'] == e])
    json.dump(res, open(outp, 'w'), indent=1)
    keys = ['seconds', 'hold_ratio_face', 'hold_ratio_head', 'hold_ratio_both', 'hold_len_median', 'hold_len_p90', 'head_speed_mean', 'head_speed_p95', 'head_vel_kurtosis', 'brow_vel_kurtosis',
            'brow_speed_mean', 'brow_speed_p95', 'head_moves_per_min', 'head_move_dur_median', 'head_move_amp_median', 'head_move_amp_p90', 'head_move_peak_median',
            'head_move_overshoot_share', 'head_move_overshoot_median', 'face_moves_per_min', 'face_move_dur_median', 'face_move_amp_median', 'blink_per_min', 'blink_dur_median']
    cols = ['animation', 'actors'] + list(res['films'].keys())
    print('%-28s' % '', ' '.join('%12s' % c[:12] for c in cols))
    for k in keys:
        print('%-28s' % k, ' '.join('%12s' % ('-' if res[c][k] is None else '%.3f' % res[c][k]) if c in res else '%12s' % ('-' if res['films'][c][k] is None else '%.3f' % res['films'][c][k]) for c in cols))
    for grp in ['amp_p95', 'asym']:
        for k in res['animation'][grp]:
            print('%-28s' % (grp + '.' + k), ' '.join('%12s' % ('-' if (res[c] if c in res else res['films'][c])[grp][k] is None else '%.3f' % (res[c] if c in res else res['films'][c])[grp][k]) for c in cols))
    print('%-28s' % 'amp_head_p95(p,y,r)', ' '.join('%12s' % ','.join('%.0f' % v for v in ((res[c] if c in res else res['films'][c])['amp_head_p95'] or [])) for c in cols))


if __name__ == '__main__':
    main()
