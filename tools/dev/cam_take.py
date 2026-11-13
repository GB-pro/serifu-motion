"""What a camera would see during a comparison take: the actor's raw tracked channels (tracker space), frame by frame.
python tools/dev/cam_take.py <take.json> <feat_dir>  ->  <take>.cam.f32 (T x 30)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'train'))
from dataset import targets, NAMES
take_p, feat = sys.argv[1:3]; tk = json.load(open(take_p)); T = tk['T']
out = np.full((T, len(NAMES)), np.nan, np.float32)
for c in tk['clips']:
    Y = targets(np.load(os.path.join(feat, c['name'] + '.npz'))); f1 = min(c['f1'], T); n = min(f1 - c['f0'], len(Y)); out[c['f0']:c['f0'] + n] = Y[:n]
for k in range(out.shape[1]):
    col = out[:, k]; m = np.isfinite(col); col[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), col[m])
out.tofile(take_p[:-5] + '.cam.f32'); print('wrote', take_p[:-5] + '.cam.f32')
