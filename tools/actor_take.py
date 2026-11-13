"""The tracked actor transferred to the rig as it is (the app's linear mapping), laid on the timeline of a comparison take.
python tools/actor_take.py <take.json> <feat_dir> <model.json>   ->  <take>.actor.f32 next to the take
Between two lines the pose glides from the end of one recording to the start of the next."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train'))
from dataset import load_all, actor_baselines, apply_baseline
from evaluate2 import tracker_to_rig, K
take_p, feat_dir, model = sys.argv[1:4]
tk = json.load(open(take_p)); T = tk['T']
neutral = np.array(json.load(open(model))['meta']['neutral'], np.float32)
clips = load_all(feat_dir); base, gneu = actor_baselines(clips); apply_baseline(clips, base, gneu)
by = {c['name']: c for c in clips}
out = np.zeros((T, K), np.float32); have = np.zeros(T, bool)
for c in tk['clips']:
    R = tracker_to_rig(by[c['name']]['Yn'], neutral); f1 = min(c['f1'], T); n = f1 - c['f0']
    if len(R) < n: R = np.concatenate([R, np.repeat(R[-1:], n - len(R), 0)])
    out[c['f0']:f1] = R[:n]; have[c['f0']:f1] = True
idx = np.flatnonzero(have)
for t in np.flatnonzero(~have):
    a = idx[idx < t]; b = idx[idx > t]
    if len(a) and len(b):
        u = (t - a[-1]) / (b[0] - a[-1]); u = u * u * (3 - 2 * u); out[t] = out[a[-1]] * (1 - u) + out[b[0]] * u
    else: out[t] = out[a[-1]] if len(a) else out[b[0]]
out.tofile(take_p[:-5] + '.actor.f32'); print('wrote', take_p[:-5] + '.actor.f32', T, 'frames')
