# Speakers: who is used for what

CREMA-D has 91 actors (1001 to 1091). No clip of an evaluation speaker is ever used for training or tuning.

| Set | Actors | Use |
|---|---|---|
| Training | 1001 to 1075, minus the fresh set below (66 actors) | Network weights |
| Validation | 1076 to 1082 (7 actors) | Every tuning decision: constants, thresholds, model choice |
| Held-out set 1 | 1083 to 1091 (9 actors) | Evaluation runs 1 to 4. Used four times, so no longer a clean test |
| Fresh set (held-out set 2) | 1008, 1016, 1024, 1032, 1040, 1048, 1056, 1064, 1072 (9 actors) | Iteration 5: one check against the targets below, after the models are frozen |

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
