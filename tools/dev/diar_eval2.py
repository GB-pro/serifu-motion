# python3 -I tools/dev/diar_eval2.py <jawL.json> <jawR.json> <turns.txt>... : who-speaks reference at 10 fps from the jaw movement of the
# left and right face (std of jaw opening over 1 s), scored only where one mouth clearly moves more; best mapping of speaker numbers
import sys, json, numpy as np
a, b = (np.array(json.load(open(f))) for f in sys.argv[1:3]); T = min(len(a), len(b))
def mv(j):
  return np.array([(lambda w: w[w >= 0].std() if (w >= 0).sum() > 5 else 0)(j[max(0, t - 5):t + 5]) for t in range(T)])
A, B = mv(a), mv(b)
ref = np.full(T, -1); ref[(A > 2 * B) & (A > 0.04)] = 0; ref[(B > 2 * A) & (B > 0.04)] = 1
print('reference (10 fps): left', (ref == 0).sum(), 'right', (ref == 1).sum(), 'unsure', (ref < 0).sum())
for f in sys.argv[3:]:
  hyp = np.full(T, -1)
  for line in open(f):
    s, e, k = line.split(); hyp[int(float(s) * 10):min(T, int(float(e) * 10))] = int(k)
  m = (ref >= 0) & (hyp >= 0); acc = max((hyp[m] == ref[m]).mean(), (hyp[m] == 1 - ref[m]).mean()) if m.sum() else 0
  print('%-12s scored %4d  agreement %.1f%%' % (f.split('/')[-1], m.sum(), 100 * acc))
