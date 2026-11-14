# python3 -I tools/dev/diar_eval3.py <ref.json (10 fps labels, -1 unsure)> <turns.txt>... : agreement with the best mapping of speaker numbers
import sys, json, numpy as np
ref = np.array(json.load(open(sys.argv[1]))); T = len(ref)
for f in sys.argv[2:]:
  hyp = np.full(T, -1)
  for line in open(f):
    s, e, k = line.split(); hyp[int(float(s) * 10):min(T, int(float(e) * 10))] = int(k)
  m = (ref >= 0) & (hyp >= 0); acc = max((hyp[m] == ref[m]).mean(), (hyp[m] == 1 - ref[m]).mean()) if m.sum() else 0
  print('%-12s scored %4d of %4d  agreement %.1f%%' % (f.split('/')[-1], m.sum(), (ref >= 0).sum(), 100 * acc))
