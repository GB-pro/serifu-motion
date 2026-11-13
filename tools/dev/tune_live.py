"""Tuning of the live director's listening rule on the validation speakers (never on evaluation speakers).
The realtime network(s) run once over the eval-order sessions; then the rule that turns their per-frame answers into
"the feeling being played" is replayed with different settings: how fast the running answer follows the network (ema),
how often the pose is reconsidered (reread) and how much better a new reading must be (margins).
python tools/dev/tune_live.py <feat_dir> <a.pt|a.json[,b...]> <actors,csv> <live_eval_dir>
"""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'train'))
import train as T, evalpy
from dataset import load_all, EMOS

feat, models_arg, actors_arg, live_dir = sys.argv[1:5]
actors = set(map(int, actors_arg.split(',')))
models, G, la = evalpy.load_models(models_arg.split(','))
by = T.prepare_actor(load_all(feat, only=actors), feat)
ev = {k: v for k, v in T.load_sessions(live_dir, by).items() if k in actors}
post = [(s, *T.live_posteriors(models, s, G, la)) for ss in ev.values() for s in ss]


def run(**kw):
    tot = {k: np.zeros(2) for k in ('h5', 'h1', 'he', 'p5', 'p1', 'pe')}; sw = 0; nl = 0
    for sess, P, sp in post:
        heard, played = T.live_director(P, sp, la, **kw); Tn = len(P)
        for (c, t0, n, emo) in sess['lines']:
            idx = np.flatnonzero(sp[t0:min(Tn, t0 + n + 6)])
            if not len(idx): continue
            s0, e0 = t0 + idx[0], t0 + idx[-1]; nl += 1
            sw += int((np.diff(played[s0:e0 + 1]) != 0).sum())
            for key, i in (('5', s0 + 15), ('1', s0 + 30), ('e', e0)):
                if i <= e0 and i < Tn: tot['h' + key] += (heard[i] == emo, 1); tot['p' + key] += (played[i] == emo, 1)
    r = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
    return 'heard %.3f %.3f %.3f  played %.3f %.3f %.3f  pose changes inside a line %.2f' % (r['h5'], r['h1'], r['he'], r['p5'], r['p1'], r['pe'], sw / nl)


print('lines', sum(len(s['lines']) for s, _, _ in post), 'models', len(models))
print('first versions (every pause of 0.27 s starts from scratch):', run(new_line=8, cont=False))
for nl in (12, 15, 20, 30):
    print('a pause under %d frames continues the line:' % nl, run(new_line=nl))
for ema in (0.1, 0.15, 0.25, 0.4):
    print('ema %.2f:' % ema, run(ema=ema))
for rr in ((8, 16, 12), (8, 16, 8), (6, 12, 6), (8, 14, 6)):
    for mg in ((0.03, 0.08), (0.03, 0.05), (0.0, 0.03)):
        print('reread %s margins %s:' % (rr, mg), run(reread=rr, margins=mg))
