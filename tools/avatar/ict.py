"""The head model this app's characters are built from: ICT-FaceKit "ICT Face Model Light" (USC Institute for Creative
Technologies, MIT licence, github.com/ICT-VGL/ICT-FaceKit): a neutral head and one mesh per expression shape, all with
the same 26,719 vertices. This module reads the OBJ files once and keeps them as arrays (cache file next to the script's
output), and names the parts of the mesh as the README of the model does."""
import os, json
import numpy as np

ICT = os.environ.get('ICT_DIR', '/home/user/ict-vgl/ict-facekit/FaceXModel')
PARTS = dict(face=(0, 9409), head=(9409, 11248), mouth=(11248, 13294), socketL=(13294, 13678), socketR=(13678, 14062),
             gums=(14062, 17039), teeth=(17039, 21451), eyeL=(21451, 23021), eyeR=(23021, 24591), fluidL=(24591, 24795),
             fluidR=(24795, 24999), blendL=(24999, 25023), blendR=(25023, 25047), occL=(25047, 25199), occR=(25199, 25351),
             lashL=(25351, 26035), lashR=(26035, 26719))


def read_obj(path, faces=False):
    V, UV, F, FT, mats, cur = [], [], [], [], [], None
    with open(path) as f:
        for line in f:
            if line.startswith('v '): V.append(line.split()[1:4])
            elif faces and line.startswith('vt '): UV.append(line.split()[1:3])
            elif faces and line.startswith('usemtl'): cur = line.split()[1]
            elif faces and line.startswith('f '):
                p = [t.split('/') for t in line.split()[1:]]
                F.append([int(t[0]) - 1 for t in p]); FT.append([int(t[1]) - 1 if len(t) > 1 and t[1] else -1 for t in p]); mats.append(cur)
    V = np.array(V, np.float64)
    return (V, np.array(UV, np.float64), F, FT, mats) if faces else V


def load(cache):
    """-> dict(V neutral (n,3), uv, faces (list of vertex index lists), face_uv, mats, shapes {name: delta (n,3) float32})"""
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return dict(V=d['V'], uv=d['uv'], faces=list(d['faces']), face_uv=list(d['face_uv']), mats=list(d['mats']), shapes={k[2:]: d[k] for k in d.files if k.startswith('s_')})
    V, uv, F, FT, mats = read_obj(os.path.join(ICT, 'generic_neutral_mesh.obj'), faces=True)
    shapes = {}
    for fn in sorted(os.listdir(ICT)):
        if not fn.endswith('.obj') or fn.startswith('generic') or fn.startswith('identity'): continue
        shapes[fn[:-4]] = (read_obj(os.path.join(ICT, fn)) - V).astype(np.float32)
    np.savez_compressed(cache, V=V, uv=uv, faces=np.array(F, dtype=object), face_uv=np.array(FT, dtype=object), mats=np.array(mats, dtype=object), **{'s_' + k: v for k, v in shapes.items()})
    return dict(V=V, uv=uv, faces=F, face_uv=FT, mats=mats, shapes=shapes)
