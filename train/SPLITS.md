# Speakers: who is used for what

CREMA-D has 91 actors (1001 to 1091). No clip of an evaluation speaker is ever used for training or tuning.

| Set | Actors | Use |
|---|---|---|
| Training | 1001 to 1075, minus the fresh set below (66 actors) | Network weights |
| Validation | 1076 to 1082 (7 actors) | Every tuning decision: constants, thresholds, model choice |
| Held-out set 1 | 1083 to 1091 (9 actors) | Evaluation runs 1 to 4. Used four times, so no longer a clean test |
| Fresh set (held-out set 2) | 1008, 1016, 1024, 1032, 1040, 1048, 1056, 1064, 1072 (9 actors) | Iteration 5: one check against the targets below, after the models are frozen. Done on 2026-11-14, so used once |

## Fresh set: rule and timing

Rule: every actor in the old training range 1001 to 1075 whose number is divisible by 8. The rule was written down on
2026-11-14 (05:10 JST), before any model of iteration 5 was trained and without looking at any result of these
actors. Until iteration 5 they were ordinary training speakers; from iteration 5 on, the networks are trained
without them (`TEST_ACTORS` in `train.py` lists held-out set 1 and the fresh set).

## Targets fixed in advance for iteration 5 (reading of the emotion, 6 classes)

Measured on the fresh set with `eval_short.mjs` and `eval_live.mjs`, once, with frozen models:

- a recording of a single line: at least 55 %
- a recording of about 13 lines, all in the same emotion: at least 55 %
- a long recording (all lines of the speaker): at least 64 %
- live: the emotion being played at the end of the line is the acted one in at least 55 % of the lines

Held-out set 1 is reported next to it as a fifth use, for comparison with the earlier runs only.

## Result of the check (2026-11-14)

Networks and app code were frozen in commit 062641a (06:36 JST); the fresh set was measured once, after that, from
that commit. Raw output: `results/*_it5_*`.

| Target | Fresh set | Met | Held-out set 1, run 4 → iteration 5 | Validation, run 4 → iteration 5 |
|---|---|---|---|---|
| single line, at least 55 % | 67.3 % | yes | 43.1 → 58.0 % | 49.9 → 68.4 % |
| about 13 lines in one emotion, at least 55 % | 63.1 % | yes | 40.8 → 53.8 % | 47.5 → 63.9 % |
| long recording, at least 64 % | 73.1 % | yes | 66.0 → 65.6 % | 72.8 → 73.1 % |
| live, played at the end of the line, at least 55 % | 63.1 % | yes | 47.8 → 53.4 % | 54.6 → 65.5 % |

How to read it:

- All four targets are met on the fresh set, each by 8 points or more. Speaker by speaker, 7 of the 9 fresh speakers
  meet all four (`results/eval_it5_by_actor.txt`).
- The fresh speakers turned out to be easier than held-out set 1 by 7 to 10 points in every condition, about as easy
  as the validation speakers. On held-out set 1 the same networks gain 15 and 13 points in the two short conditions and
  6 points live, but stay just under 55 % for the one-emotion recording (53.8 %) and for live (53.4 %); 3 of its 9
  speakers meet all four targets. The size of the gain is the more portable number, not the level.
- What the networks of run 4 would score on the fresh set as unseen speakers cannot be measured: they were trained
  on those speakers.
- The eight criteria of the main evaluation still hold on all three sets (`results/eval2_it5_*.json`). On held-out
  set 1 legibility against the actor went from 1.07 to 1.04 with the observer of run 4 (1.00 with the observer
  trained without the fresh speakers), and lip sync from 0.61 to 0.59.
- Every speaker of CREMA-D has now been used for training, tuning or a check. A further clean check needs speakers
  from outside this corpus.

## Check on Japanese speech: fixed before any Japanese audio was measured (2026-11-14, 07:25 JST)

Nothing Japanese has been used for training or tuning, and no Japanese audio has been run through the networks so far.

- Material: 声優統計コーパス (voice-statistics.github.io). Three professional voice actresses read 100 phoneme-balanced
  sentences each in three ways: normal, happy, angry. The audio may be used for analysis only and may not be
  redistributed, so only numbers are reported: no audio and no features of it go into this repository or the app.
- Networks and app code: as frozen for iteration 5 (commit 062641a). Nothing is changed for this check.
- Sessions: the 100 sentences of each actress in four blocks of 25 (sentences 1 to 25, 26 to 50, 51 to 75, 76 to 100),
  which gives 12 sessions of 75 lines, 25 per emotion. "normal" counts as NEU.
- Measures: `eval_short.mjs` and `eval_live.mjs`, as for the targets above. The reading is over all six classes, so
  chance is 1 in 6, and SAD, FEA and DIS are possible wrong answers.
- Marks, the same numbers as the English targets: a single line at least 55 %; a recording of 25 lines in one emotion
  at least 55 %; a long recording (the 75 lines of a session) at least 64 %; live, played at the end of the line, at
  least 55 %.
- For reference, English on the same three emotions (NEU, HAP, ANG): fresh speakers 74 %, 69 %, 81 %, 67 %; held-out
  set 1: 55 %, 52 %, 65 %, 51 % (single line, one emotion, long, live).
- Limits known in advance: three speakers, all female, read sentences, three of the six emotions.
