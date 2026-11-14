# python3 -I tools/dev/diar_eval.py <L.json> <R.json> <turns.txt>... : builds a who-speaks reference from the jaw movement of the left and right
# face (video halves) and scores speaker turns against it (best mapping of speaker numbers; only frames where exactly one mouth clearly moves)
import sys, json, numpy as np
L, R = (json.load(open(f)) for f in sys.argv[1:3])
def act(d):
  j = np.array(d['jaw']); s = np.array(d['seen']) > 0; T = len(j)
  w = 15; v = np.array([j[max(0, t - w):t + w].std() if s[t] else 0 for t in range(T)]); return v
a, b = act(L), act(R); T = min(len(a), len(b))
ref = np.full(T, -1); ref[(a > 1.8 * b) & (a > 0.03)] = 0; ref[(b > 1.8 * a) & (b > 0.03)] = 1
print('reference frames: left', (ref == 0).sum(), 'right', (ref == 1).sum(), 'unsure', (ref < 0).sum())
for f in sys.argv[3:]:
  hyp = np.full(T, -1)
  for line in open(f):
    s, e, k = line.split(); s, e, k = int(float(s) * 30), int(float(e) * 30), int(k); hyp[s:min(e, T)] = k
  m = (ref >= 0) & (hyp >= 0); acc = max(((hyp[m] == ref[m]).mean()), ((hyp[m] == 1 - ref[m]).mean())) if m.sum() else 0
  print(f.split('/')[-1], 'frames scored', m.sum(), 'agreement %.1f%%' % (100 * acc))
