"""How much does the reading lean on the room noise of the training recordings?

CREMA-D was recorded in ordinary rooms: between the words the level is only about 20 dB below the speech level, and
less than that when the actor speaks quietly. A studio recording (or a noise gate, or a phone's noise suppression) is
silent between the words. This check reads the lines of the validation speakers twice: as recorded, and with every
frame that is not speech lowered by 45 dB (statistics and normalisation recomputed, as the app would).

  python floor_check.py <feat> <line.json,line_b.json,relative.json>

The committee is weighted as in the app (app/core/generator.js): a single line is read by the 'line' readers only, a
long recording by all three with the 'relative' reader at 0.4.
"""
import sys
import numpy as np
import train as T, evalpy as EP, simnorm as SN
from dataset import load_all, EMOS

LOWER = 45.0


def gate(R, e_speech):
    """Lower every frame that is neither voiced nor within 15 dB of the speech level."""
    R = R.copy(); ns = (R[:, 1] <= 0) & (R[:, 0] - e_speech < -15); R[ns, 0] -= LOWER
    return R


def read(models, w, xs):
    gapx = np.repeat(T.SIL[None], T.GAP_EVAL, 0); parts = []; spans = []; t = 0
    for x in xs: parts += [gapx, x]; spans.append((t + T.GAP_EVAL, t + T.GAP_EVAL + len(x))); t += T.GAP_EVAL + len(x)
    xx = np.concatenate(parts + [gapx]).astype(np.float32); P = 0
    for m, wm in zip(models, w):
        if not wm: continue
        lg = T.run_padded(m, xx, 0, pad=30)[:, T.K:T.K + T.NE]; e = np.exp(lg - lg.max(1, keepdims=True)); P = P + wm * e / e.sum(1, keepdims=True)
    out = []
    for (a, b), x in zip(spans, xs):
        p = P[a:b]; sp = (x[:, 1] > 0.5) | (x[:, 0] > -1.5); out.append(int((p[sp] if sp.any() else p).mean(0).argmax()))
    return out


def main(feat, paths):
    models = [EP.load_one(p)[0] for p in paths.split(',')]
    clips = load_all(feat, only=T.VAL_ACTORS); by = T.prepare_actor(clips, feat)
    res = {}
    for a, A in sorted(by.items()):
        cl = sorted(A['clips'], key=lambda c: c['name']); es = A['stats']['eSpeech']
        for name, raws in (('as recorded', [c['R'] for c in cl]), ('silent between the words', [gate(c['R'], es) for c in cl])):
            st = SN.speaker_stats(raws)
            r_long = read(models, (0.3, 0.3, 0.4), [SN.normalize(r, st)[:c['n']] for r, c in zip(raws, cl)])
            r_one = [read(models, (0.5, 0.5, 0.0), [SN.normalize(r, SN.speaker_stats([r]))[:c['n']]])[0] for r, c in zip(raws, cl)]
            for key, rr in (('long recording', r_long), ('single line', r_one)):
                t = res.setdefault((key, name), np.zeros((T.NE, T.NE), int))
                for c, e in zip(cl, rr): t[c['emo'], e] += 1
    print(f'validation speakers {sorted(by)}: {len(clips)} lines; share of lines whose emotion is read correctly\n')
    for (key, name), C in res.items():
        print(f'{key}, {name}: {np.trace(C) / C.sum():.3f}   right by emotion: ' + '  '.join(f'{EMOS[i]} {C[i, i] / max(1, C[i].sum()):.2f}' for i in range(T.NE))
              + '   answers: ' + '  '.join(f'{EMOS[j]} {C[:, j].sum()}' for j in range(T.NE)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
