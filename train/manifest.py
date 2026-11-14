"""manifest.py <spec[@limit],...> <out.tsv>: files (name, corpus, speaker, emotion, wav, raw frames) for embed_lines.mjs.
spec: as in corpus.py, or crema:train | crema:val (CREMA-D by the actor roles of train.py)."""
import sys, os, glob
sys.path.insert(0, '/root/sm/train')
import numpy as np, corpus as CP, train as T
H = os.path.expanduser
out = []
for item in sys.argv[1].split(','):
    spec, _, lim = item.partition('@'); lim = int(lim) if lim else None
    if spec.startswith('crema'):
        part = spec.split(':')[1]; held = T.TEST_ACTORS | T.VAL_ACTORS
        actors = sorted(T.VAL_ACTORS) if part == 'val' else [a for a in range(1001, 1092) if a not in held]
        rows = []
        for a in actors:
            for p in sorted(glob.glob(H(f'~/data/crema/feat/{a}_*.raw.f32'))):
                nm = os.path.basename(p)[:-8]; rows.append((nm, spec, f'crema.{a}', CP.EMOS.index(nm.split('_')[2]), p[:-8] + '.wav', p))
    else:
        code = spec.split(':')[0]; rows = []
        for c in CP.load(spec):
            wav = H(f'~/data/ja/wav16/{c["name"]}.wav') if code == 'ja_vs' else H(f'~/data/jvnv/wav16/{c["name"]}.wav') if code == 'jvnv' else H(f'~/data/fleurs/ja/x/ja_jp/audio/{spec.split(":")[1]}/{c["name"]}.wav') if code.startswith('fleurs') else os.path.join(CP.UMD, 'wav', code, c['name'] + '.wav')
            rows.append((c['name'], spec, c['speaker'], c['emo'], wav, c['path']))
    if lim and len(rows) > lim: rows = [rows[i] for i in np.linspace(0, len(rows) - 1, lim).round().astype(int)]
    miss = sum(1 for r in rows if not os.path.exists(r[4])); print(spec, len(rows), 'lines', f'({miss} without audio)' if miss else '', flush=True)
    out += [r for r in rows if os.path.exists(r[4])]
open(sys.argv[2], 'w').write(''.join('\t'.join(map(str, r)) + '\n' for r in out))
