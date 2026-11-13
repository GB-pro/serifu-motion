"""Tuning-time checks of trained networks in Python (the checks that count are eval_short.mjs / eval_live.mjs, which
run the app's own code). Reads the .pt files written next to the exported models.
python evalpy.py <feat_dir> <a.pt[,b.pt...]> <actors,csv> [--live-eval dir] [--tilt dB]
  --tilt: also read the recordings as if made with a microphone whose response tilts by that many dB over the band
"""
import argparse, os, sys, json, base64, numpy as np, torch
import train as T
from dataset import load_all, EMOS
import simnorm as SN


def load_one(p):
    if p.endswith('.json'):  # an exported model (the app's format)
        j = json.load(open(p)); meta = j['meta']
        state = {k: torch.tensor(np.frombuffer(base64.b64decode(v['b64']), np.float32).reshape(v['shape']).copy()) for k, v in j['weights'].items()}
        d = dict(state=state, G=meta.get('globals') or [], la=meta.get('lookahead', 0), C=meta['C'], causal=meta['causal'])
    else:
        d = torch.load(p)
    m = T.TCN(T.FEAT_DIM + len(d['G']), T.K + T.NE + T.NA, C=d['C'], causal=d['causal']); m.load_state_dict(d['state']); m.eval()
    return m, d


def load_models(paths):
    out = [load_one(p) for p in paths]
    assert all(o[1]['G'] == out[0][1]['G'] for o in out)
    return [o[0] for o in out], out[0][1]['G'], out[0][1]['la']


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('feat_dir'); ap.add_argument('models'); ap.add_argument('actors')
    ap.add_argument('--live-eval', default=''); ap.add_argument('--tilt', type=float, default=0.0)
    a = ap.parse_args(); actors = set(map(int, a.actors.split(',')))
    models, G, la = load_models(a.models.split(','))
    clips = [c for c in load_all(a.feat_dir, only=actors)]
    by = T.prepare_actor(clips, a.feat_dir)
    if a.tilt:  # a tilted microphone adds a constant to every cepstral coefficient: only the recording-level timbre inputs change
        NM = 26; m = np.arange(NM); h = a.tilt / 4.343 * (m - (NM - 1) / 2) / (NM - 1)
        off = np.array([np.sum(np.cos(np.pi * c * (m + 0.5) / NM) * h) for c in range(1, 13)])
        g0 = SN.global_features
        SN.global_features = lambda st: g0(dict(st, mfMean=np.asarray(st['mfMean']) + off, centMean=st['centMean'] * (1 + 0.02 * a.tilt), centStd=st['centStd'] * (1 + 0.02 * a.tilt)))
        for A in by.values(): A['glob'] = SN.global_features(A['stats'])
        print('microphone tilt %.0f dB: cepstral offsets' % a.tilt, off.round(1)[:4], '...')
    if a.live_eval:
        ev = {k: v for k, v in T.load_sessions(a.live_eval, by).items() if k in actors}
        res, per = T.eval_live(models, ev, G, la); print('live:', res, ' played at the end by emotion', dict(zip(EMOS, per)))
    else:
        cond = T.eval_conditions(models, by, G, la)
        print('reading by recording:', '  '.join(f'{k} {v[0]:.3f}' for k, v in cond.items()))
        print('  single line by emotion', dict(zip(EMOS, cond['single'][1])), ' one emotion', dict(zip(EMOS, cond['same'][1])))
