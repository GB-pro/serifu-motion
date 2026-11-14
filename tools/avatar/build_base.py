"""Build the head asset the app loads (app/avatar/head/base.json + base.bin) from the ICT face model (ict.py).

  python build_base.py <out dir>

Kept: skin of the face, back of the head with neck and shoulders, inside of the mouth, eye sockets, gums and tongue,
teeth, eyelashes. Left out: the eyeballs and the meshes for realistic eye rendering (the app draws its own eyes).
The asset is the model as published (realistic proportions, centimetres); a character's proportions are applied by the
app when it loads the asset (app/avatar/head.js: the same change of shape for the neutral head and for every expression).

base.bin, little endian, in this order (sizes in base.json):
  positions float32 n*3 | triangles uint32 t*3 | region uint8 n (index into regions) | rings uint8 n*3 (steps from the
  mouth opening, from the left and from the right eye opening, capped at 255) | for every shape: indices uint32 m,
  deltas int16 m*3 (times scale / 32767)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ict

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
M = ict.load(os.path.expanduser('~/data/avatar/ict_cache.npz')); V = M['V']
KEEP = ['face', 'head', 'mouth', 'socketL', 'socketR', 'gums', 'teeth', 'lashL', 'lashR']
old = np.concatenate([np.arange(*ict.PARTS[k]) for k in KEEP]); new_of = -np.ones(len(V), np.int64); new_of[old] = np.arange(len(old))
n = len(old); P = V[old]
region = np.zeros(n, np.uint8); regions = []
for k in KEEP:
    a, b = ict.PARTS[k]; regions.append(k); region[new_of[a]:new_of[b - 1] + 1] = len(regions) - 1
tris = []; tri_region = []
for f in M['faces']:
    g = new_of[f]
    if (g < 0).any(): continue
    for i in range(1, len(g) - 1): tris.append((g[0], g[i], g[i + 1])); tri_region.append(region[g[0]])
tris = np.array(tris, np.uint32); tri_region = np.array(tri_region)
order = np.argsort(tri_region, kind='stable'); tris = tris[order]; tri_region = tri_region[order]   # one draw group per region
groups = [[int(np.searchsorted(tri_region, r)), int((tri_region == r).sum())] for r in range(len(regions))]

# openings of the face skin: edges that belong to one triangle only, chained into loops
fa, fb = ict.PARTS['face']; ft = tris[tri_region == 0].astype(np.int64)
e = np.sort(np.concatenate([ft[:, [0, 1]], ft[:, [1, 2]], ft[:, [2, 0]]]), axis=1); key = e[:, 0] * n + e[:, 1]
uniq, cnt = np.unique(key, return_counts=True); be = uniq[cnt == 1]; be = np.stack([be // n, be % n], 1)
adj = {}
for a, b in be: adj.setdefault(int(a), []).append(int(b)); adj.setdefault(int(b), []).append(int(a))
loops = []; seen = set()
for s in adj:
    if s in seen: continue
    loop = [s]; seen.add(s); cur = s
    while True:
        nx = [v for v in adj[cur] if v not in seen]
        if not nx: break
        cur = nx[0]; seen.add(cur); loop.append(cur)
    loops.append(np.array(loop))
info = [(len(l), P[l].mean(0).round(2).tolist()) for l in loops]; print('openings of the face skin (vertices, centre):', info)
pick = lambda c: min(loops, key=lambda l: np.linalg.norm(P[l].mean(0) - np.array(c)) + (1e3 if len(l) > 400 else 0))
mouth_loop, eyeL_loop, eyeR_loop = pick([0, -3, 10]), pick([3.1, 3.5, 9]), pick([-3.1, 3.5, 9])
print('mouth opening', len(mouth_loop), P[mouth_loop].mean(0).round(2), 'eye openings', len(eyeL_loop), P[eyeL_loop].mean(0).round(2), len(eyeR_loop))

# steps (edge rings) from each opening, over all kept triangles (so the count also runs into the mouth and the sockets)
nb = [[] for _ in range(n)]
for a, b, c in tris.astype(np.int64): nb[a] += [b, c]; nb[b] += [a, c]; nb[c] += [a, b]
# vertices of other parts that coincide with a vertex of the opening continue the count there
def rings(loop):
    d = np.full(n, 255, np.int64); d[loop] = 0; front = list(loop)
    pos = {tuple(np.round(P[i], 4)): i for i in loop}
    for i in range(n):
        t = tuple(np.round(P[i], 4))
        if t in pos and d[i] == 255: d[i] = 0; front.append(i)
    k = 0
    while front and k < 254:
        k += 1; nxt = []
        for u in front:
            for v in nb[u]:
                if d[v] == 255: d[v] = k; nxt.append(v)
        front = nxt
    return d.astype(np.uint8)
R = np.stack([rings(mouth_loop), rings(eyeL_loop), rings(eyeR_loop)], 1)

# eyes: sphere through the eyeball vertices that are not on the bulge of the cornea
def sphere(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]; x = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]; c = x[:3]; return c, float(np.sqrt(x[3] + c @ c))
eyes = {}
for k in ('eyeL', 'eyeR'):
    Q = V[slice(*ict.PARTS[k])]; c, r = sphere(Q); keep = np.linalg.norm(Q - c, axis=1) < r * 1.03
    for _ in range(3): c, r = sphere(Q[keep]); keep = np.abs(np.linalg.norm(Q - c, axis=1) - r) < 0.06 * r
    eyes[k] = dict(centre=c.round(4).tolist(), radius=round(r, 4)); print(k, eyes[k], 'vertices used', int(keep.sum()), 'of', len(Q))

names = [k for k in sorted(M['shapes']) if not k.startswith('PupilDilate')]
blob = [P.astype('<f4').tobytes(), tris.astype('<u4').tobytes(), region.tobytes(), R.tobytes()]; shapes = []
for k in names:
    d = M['shapes'][k][old]; idx = np.where(np.linalg.norm(d, axis=1) > 2e-3)[0]; scale = float(np.abs(d[idx]).max())
    q = np.round(d[idx] / scale * 32767).astype('<i2'); blob += [idx.astype('<u4').tobytes(), q.tobytes()]; shapes.append(dict(name=k, count=int(len(idx)), scale=scale))
open(os.path.join(out, 'base.bin'), 'wb').write(b''.join(blob))
meta = dict(source='ICT-FaceKit, ICT Face Model Light (USC Institute for Creative Technologies), MIT licence; see LICENSE-ICT-FaceKit.txt',
            vertices=n, triangles=len(tris), regions=regions, groups=groups, eyes=eyes, shapes=shapes,
            openings=dict(mouth=len(mouth_loop), eyeL=len(eyeL_loop), eyeR=len(eyeR_loop)))
json.dump(meta, open(os.path.join(out, 'base.json'), 'w'))
print('vertices', n, 'triangles', len(tris), 'shapes', len(shapes), 'file', round(os.path.getsize(os.path.join(out, 'base.bin')) / 1e6, 2), 'MB', 'groups', dict(zip(regions, groups)))
