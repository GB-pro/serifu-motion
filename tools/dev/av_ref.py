# python3 -I tools/dev/av_ref.py <wav 16k> <jawL.json> <jawR.json> <out.json> : who-speaks reference (10 fps) from audio-visual sync: over each 2 s
# the face whose jaw opening follows the loudness of the sound best is the speaker; kept only where one face clearly wins
import sys, json, wave, numpy as np
w = wave.open(sys.argv[1]); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
a, b = (np.array(json.load(open(f))) for f in sys.argv[2:4]); T = min(len(a), len(b), len(x) // 1600)
env = np.array([np.sqrt((x[t * 1600:(t + 1) * 1600] ** 2).mean()) for t in range(T)]); db = 20 * np.log10(env + 1e-5)
thr = np.percentile(db, 95) - 25
ref = np.full(T, -1)
def corr(j, e):
  m = j >= 0
  if m.sum() < 12 or j[m].std() < 0.01: return 0
  return np.corrcoef(j[m], e[m])[0, 1]
for t in range(T):
  if db[t] < thr: continue
  s, e = max(0, t - 10), min(T, t + 10)
  ca, cb = corr(a[s:e], db[s:e]), corr(b[s:e], db[s:e])
  if ca > 0.3 and ca > cb + 0.25: ref[t] = 0
  elif cb > 0.3 and cb > ca + 0.25: ref[t] = 1
json.dump(ref.tolist(), open(sys.argv[4], 'w'))
print('speech frames', (db >= thr).sum(), 'left', (ref == 0).sum(), 'right', (ref == 1).sum())
