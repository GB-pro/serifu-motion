"""Lines of the corpora other than CREMA-D: who speaks, which emotion, raw frames (feats_raw.mjs).

The roles of the corpora (training, development, final check) are fixed in SPLITS.md. Nothing here reads audio:
the raw frames are made once with the app's own extractor.

Folders (not part of the repository):
  <umd>/raw/<code>/<record>+<code>+<speaker>+<gender>+<emotion>+<valence>+<iso3>+<locale>.raw.f32   collection
  <fleurs>/<lang>/raw/<split>/<id>.raw.f32                                                             FLEURS
  <ja>/<session>_S<nnn>_<EMO>_XX.raw.f32                                                               声優統計コーパス
  <jvnv>/<speaker>_<emotion>_<session>_<id>.raw.f32                                                    JVNV
"""
import os, glob
import numpy as np
import simnorm as SN

EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
LABEL = dict(neu=0, hap=1, sad=2, ang=3, fea=4, dis=5)   # other labels of the collection (sur, cal, amu, ...) are left out
UMD = os.path.expanduser(os.environ.get('UMD_DIR', '~/data/umd'))
FLEURS = os.path.expanduser(os.environ.get('FLEURS_DIR', '~/data/fleurs'))
JA_VS = os.path.expanduser(os.environ.get('JA_VS_DIR', '~/data/ja/feat'))
JVNV = os.path.expanduser(os.environ.get('JVNV_DIR', '~/data/jvnv/raw'))
JVNV_EMO = dict(anger=3, disgust=5, fear=4, happy=1, sad=2)   # no neutral in JVNV; surprise is left out
JVNV_TRAIN = {'F1', 'M1'}                                      # F2 and M2 are kept for the final check

# Roles, as fixed in SPLITS.md (2026-11-14) before anything was measured.
ESD_TEST = {'esd.0009', 'esd.0010', 'esd.0019', 'esd.0020'}; ESD_DEV = {'esd.0008', 'esd.0018'}
FINAL = ('ravdess', 'EmoDB', 'cafe', 'esd:test', 'fleurs_ja:test', 'jvnv:final')
DEV = ('tess', 'oreau2', 'aesdd', 'urdu', 'esd:dev', 'fleurs_ja:dev', 'ja_vs')
TRAIN_OPEN = ('jl-corpus', 'fleurs_ja:train', 'jvnv:train')
TRAIN_ALL = TRAIN_OPEN + ('esd:train', 'MELD', 'EmoV-DB_sorted', 'LEGOv2', 'ShEMO', 'MESS', 'enterface_db', 'ekorpus', 'Emotional_EMA')


def _raw(p):
    a = np.fromfile(p, np.float32)
    return a[:len(a) // SN.RAW_DIM * SN.RAW_DIM].reshape(-1, SN.RAW_DIM)


def load(spec, min_frames=12, limit=None):
    """spec: a corpus code of the collection, 'esd:train|dev|test', 'fleurs_<lang>:<split>', 'jvnv:train|final' or 'ja_vs'.
    -> list of lines: name, corpus, speaker, emo (0..5), lang, R (raw frames, loaded lazily through line['R'])."""
    code, _, part = spec.partition(':'); out = []
    if code == 'ja_vs':
        for p in sorted(glob.glob(os.path.join(JA_VS, '*.raw.f32'))):
            nm = os.path.basename(p)[:-8]; sess, _, emo, _ = nm.split('_')
            out.append(dict(name=nm, corpus='ja_vs', speaker='vs.' + sess[:3], session=sess, emo=EMOS.index(emo), lang='ja', path=p))
    elif code == 'jvnv':
        for p in sorted(glob.glob(os.path.join(JVNV, '*.raw.f32'))):
            nm = os.path.basename(p)[:-8]; spk, emo = nm.split('_')[:2]
            if emo not in JVNV_EMO or (part and (spk in JVNV_TRAIN) != (part == 'train')): continue
            out.append(dict(name=nm, corpus=spec, speaker='jvnv.' + spk, emo=JVNV_EMO[emo], lang='ja', path=p))
    elif code.startswith('fleurs_'):
        lang = code[7:]
        for p in sorted(glob.glob(os.path.join(FLEURS, lang, 'raw', part, '*.raw.f32'))):
            nm = os.path.basename(p)[:-8]   # no speaker identity in FLEURS: every line is a recording of its own
            out.append(dict(name=nm, corpus=spec, speaker=f'fleurs.{lang}.{nm}', emo=0, lang=lang, path=p, alone=True))
    else:
        songs = set()
        if code == 'ravdess':
            songs = set(l.strip() for l in open(os.path.join(UMD, 'ravdess_songfiles.txt')) if l.strip())
        for p in sorted(glob.glob(os.path.join(UMD, 'raw', code, '*.raw.f32'))):
            nm = os.path.basename(p)[:-8]; f = nm.split('+')
            if len(f) != 8 or f[4] not in LABEL or nm in songs: continue
            spk = f[2]
            if code == 'esd':
                role = 'test' if spk in ESD_TEST else 'dev' if spk in ESD_DEV else 'train'
                if part and role != part: continue
            out.append(dict(name=nm, corpus=spec, speaker=spk, emo=LABEL[f[4]], lang=f[7], path=p))
    if limit and len(out) > limit:   # an even, reproducible subset
        idx = np.linspace(0, len(out) - 1, limit).round().astype(int); out = [out[i] for i in idx]
    keep = []
    for c in out:
        c['R'] = _raw(c['path'])
        if len(c['R']) >= min_frames: keep.append(c)
    return keep


def sessions(lines, size=72):
    """Recordings for the 'long recording' reading: the lines of a speaker, emotions interleaved, at most `size` lines
    each (CREMA-D has 82 lines per actor, the Japanese sessions 75). Lines without speaker identity stay alone."""
    by = {}
    for c in lines: by.setdefault(c.get('session') or c['speaker'], []).append(c)
    out = []
    for spk, ls in sorted(by.items()):
        if ls[0].get('alone') or ls[0].get('session'): out.append(ls); continue
        per = {}
        for c in sorted(ls, key=lambda c: c['name']): per.setdefault(c['emo'], []).append(c)
        order = []; i = 0
        while any(i < len(v) for v in per.values()):
            order += [v[i] for e, v in sorted(per.items()) if i < len(v)]; i += 1
        n = max(1, int(np.ceil(len(order) / size))); step = int(np.ceil(len(order) / n))
        out += [order[k:k + step] for k in range(0, len(order), step)]
    return out
