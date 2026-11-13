"""Does the reading work on sentences the networks never saw?

CREMA-D has 12 sentences, and every other check in this folder uses the same 12 for training and for testing.
Here four networks are trained like app/models/emo_offline.json, each without three of the twelve sentences
(used neither for the weights nor for choosing the epoch):

  SKIP_SENT=DFA,IEO,IOM python train.py <feat> <dir>/foldA.json --mix .3,.2,.25,.25
  (foldB: ITH,ITS,IWL    foldC: IWW,MTI,TAI    foldD: TIE,TSI,WSI)

Then   python sent_holdout.py <feat> <dir> <a network trained on all twelve .json>
reads the lines of the validation speakers (never the evaluation speakers). Every line is read three times:
  unseen       by the fold network that never saw its sentence
  all twelve   by the network trained on all twelve sentences (what every other check measures)
  seen, 9      by another fold network, which knows the sentence but was also trained on 9 sentences only
               (what the smaller training set costs by itself)
in three kinds of recording: the line on its own; the lines of its three sentences as one recording; and the same
lines with the speaker statistics of all the speaker's lines (a long recording).
"""
import sys, os
import numpy as np
import train as T, evalpy as EP, simnorm as SN
from dataset import load_all, EMOS

FOLDS = dict(A='DFA,IEO,IOM', B='ITH,ITS,IWL', C='IWW,MTI,TAI', D='TIE,TSI,WSI')
OTHER = dict(A='B', B='C', C='D', D='A')  # a fold network that did see the sentences of this fold
READERS = ('unseen', 'all twelve', 'seen, 9')
CONDS = ('single line', 'three sentences as one recording', 'long recording')


def main(feat, d, base):
    clips = load_all(feat, only=T.VAL_ACTORS); by = T.prepare_actor(clips, feat)
    nets = {k: EP.load_one(os.path.join(d, f'fold{k}.pt'))[0] for k in FOLDS}; nets['all'] = EP.load_one(base)[0]
    fold_of = {s: k for k, v in FOLDS.items() for s in v.split(',')}
    R = {}  # (condition, reader) -> rows of (fold, emotion acted, right)

    def note(cond, reader, k, rec, read):
        R.setdefault((cond, reader), []).extend((k, c['emo'], int(e == c['emo'])) for c, e in zip(rec, read))
    for a, A in sorted(by.items()):
        cl = sorted(A['clips'], key=lambda c: c['name'])
        for k in FOLDS:
            rec = [c for c in cl if fold_of[c['sent']] == k]; st = SN.speaker_stats([c['R'] for c in rec])
            for reader, m in zip(READERS, (nets[k], nets['all'], nets[OTHER[k]])):
                note(CONDS[0], reader, k, rec, [T.read_recording([m], [c], SN.speaker_stats([c['R']]), A, [], 0)[0] for c in rec])
                note(CONDS[1], reader, k, rec, T.read_recording([m], rec, st, A, [], 0))
                note(CONDS[2], reader, k, rec, T.read_recording([m], rec, None, A, [], 0))
    print(f'validation speakers {sorted(by)}: {len(clips)} lines; share of lines whose emotion is read correctly (chance 0.167)\n')
    for cond in CONDS:
        n = len(R[(cond, 'unseen')])
        print(f'{cond} (n {n})')
        for r in READERS:
            per_fold = '  '.join(f'{k} {np.mean([v[2] for v in R[(cond, r)] if v[0] == k]):.3f}' for k in FOLDS)
            xe = np.array([(v[1], v[2]) for v in R[(cond, r)]])
            per_emo = '  '.join(f'{EMOS[e]} {xe[xe[:, 0] == e, 1].mean():.2f}' for e in range(len(EMOS)))
            print(f'  {r:<11} {xe[:, 1].mean():.3f}   by fold: {per_fold}   by emotion: {per_emo}')
        u = np.array([v[2] for v in R[(cond, 'unseen')]]); s = np.array([v[2] for v in R[(cond, 'all twelve')]])
        b, c = int(((s == 1) & (u == 0)).sum()), int(((s == 0) & (u == 1)).sum())
        se = np.sqrt(b + c) / n
        print(f'  unseen minus all twelve: {(u.mean() - s.mean()) * 100:+.1f} points (standard error {se * 100:.1f}); lines right only when the sentence was seen {b}, right only when unseen {c}\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
