// Serifu Motion — animation interpretation layer.
//
// The network (trained on recorded actors) only answers two questions about a line of dialogue:
//   WHAT is being expressed (emotion, intensity) and WHEN the speaker stresses something (accents, pauses, pitch).
// This layer decides HOW a cartoon character plays it, following the working method of character animators:
//   * one clear key pose per phrase (pose-to-pose), reached slightly BEFORE the words, with anticipation,
//     overshoot and settle, then a moving hold instead of continuous tracking;
//   * accents on stressed syllables that hit a few frames early (head, brows, eyes, torso, squash & stretch);
//   * overlapping action: eyes lead, then brows, mouth, head, torso; hair follows in the rig;
//   * blinks and eye darts placed as punctuation (start of a thought, end of a phrase, before a big accent);
//   * asymmetry and exaggeration (unilateral brows / mouth corners, tilted head, pushed poses).
// It is written as a list of timed events (Director) played by a frame-stepped Performer, so the same Performer
// runs offline (events planned with look-ahead) and live (events emitted causally).
import { IDS, IDX, K, EMOS, CHANNELS } from './au.js';
import { FEAT_DIM, FPS } from './features.js';
import { rng, gauss, clamp, lerp, smooth, softmax } from './util.js';

const SIDED = new Set(['headYaw', 'headRoll', 'gazeX', 'torsoRoll', 'torsoShift', 'torsoYaw', 'browAsym', 'mouthAsym', 'headX']);
const E = Object.fromEntries(EMOS.map((e, i) => [e, i]));

// ---------------------------------------------------------------------------------------------------------
// Pose library: key poses per emotion at full intensity, written for side = +1 (sided channels are mirrored).
// Units: AUs 0..1, head angles in degrees, headZ / torso in rig units (see avatar/rig.js).
// ---------------------------------------------------------------------------------------------------------
export const IDLE = { AU12: 0.16, AU6: 0.05, AU2: 0.04, headRoll: 1.5, torsoLean: 0.5 };
export const POSES = {
  NEU: [
    { name: 'attentive', AU1: 0.16, AU2: 0.2, AU12: 0.2, AU6: 0.06, headPitch: -2, headRoll: 3.5, torsoLean: 1.2 },
    { name: 'explaining', AU1: 0.1, AU2: 0.26, browAsym: 0.4, AU12: 0.14, headYaw: 5, headRoll: -4.5, torsoRoll: 1.2, torsoYaw: 2 },
    { name: 'considering', AU4: 0.2, AU7: 0.2, AU12: 0.08, AU14: 0.15, mouthAsym: 0.25, headPitch: 3, headRoll: 5.5, headYaw: -3 },
  ],
  HAP: [
    { name: 'beaming', AU6: 0.8, AU12: 1.0, AU1: 0.25, AU2: 0.4, AU5: 0.12, headPitch: -5, headRoll: 7, torsoLean: -1, torsoY: 2, torsoShrug: -2.5, squash: 0.2, eyeScale: 0.25, pupil: 0.35 },
    { name: 'delighted', AU6: 1.0, AU12: 1.0, AU7: 0.25, AU1: 0.2, AU2: 0.3, headPitch: -8, headRoll: -5.5, headYaw: 4, torsoLean: -3, torsoShrug: -4.5, torsoY: 2.5, squash: 0.1, pupil: 0.3 },
    { name: 'cheeky', AU6: 0.6, AU12: 0.9, mouthAsym: 0.55, browAsym: 0.55, AU2: 0.3, headRoll: 9, headYaw: -5, headPitch: 2, torsoRoll: 2.5, torsoYaw: -3, eyeScale: 0.1, pupil: 0.25 },
  ],
  SAD: [
    { name: 'downcast', AU1: 1.0, AU4: 0.45, AU15: 0.9, AU17: 0.5, AU43: 0.45, AU7: 0.1, headPitch: 12, headRoll: 5, gazeY: -0.35, torsoLean: 2.5, torsoShrug: 3, torsoY: -3, squash: -0.15, pupil: 0.25 },
    { name: 'hurt', AU1: 1.0, AU2: 0.2, AU4: 0.3, AU15: 0.75, AU17: 0.7, AU5: 0.12, headPitch: 4, headRoll: -8, headYaw: 3, torsoShrug: -2, torsoY: -1.5, torsoLean: 1, eyeScale: 0.2, pupil: 0.5 },
    { name: 'resigned', AU1: 0.75, AU4: 0.3, AU15: 0.65, AU43: 0.6, AU24: 0.3, headPitch: 8, headYaw: -8, gazeX: -0.3, gazeY: -0.3, torsoLean: 1.5, torsoY: -2.2, torsoRoll: -1.5 },
  ],
  ANG: [
    { name: 'glare', AU4: 1.0, AU7: 0.7, AU5: 0.25, AU9: 0.35, AU10: 0.3, AU15: 0.4, AU24: 0.5, AU17: 0.3, headPitch: 5, gazeY: 0.2, headZ: 2.5, headRoll: 2.5, torsoLean: 4, torsoShrug: -5, squash: -0.1, pupil: -0.35 },
    { name: 'outburst', AU4: 1.0, AU5: 0.6, AU7: 0.3, AU9: 0.5, AU10: 0.55, AU16: 0.4, AU20: 0.3, headPitch: -3, headZ: 3, headRoll: 3, torsoLean: 4.5, torsoShrug: -6, eyeScale: 0.15, pupil: -0.5 },
    { name: 'cold', AU4: 0.85, AU7: 0.8, AU14: 0.4, AU15: 0.3, mouthAsym: -0.4, browAsym: -0.45, headRoll: -5, headYaw: 7, headPitch: 4, gazeX: -0.3, torsoYaw: 5, torsoShrug: -2, torsoLean: 1.5 },
  ],
  FEA: [
    { name: 'startled', AU1: 0.9, AU2: 0.8, AU4: 0.35, AU5: 1.0, AU20: 0.7, AU16: 0.3, headPitch: -4, headZ: -3.5, torsoLean: -4, torsoShrug: -9, torsoY: 1, eyeScale: 0.45, pupil: -0.4, squash: 0.25 },
    { name: 'cowering', AU1: 1.0, AU4: 0.6, AU5: 0.5, AU7: 0.2, AU20: 0.6, AU15: 0.4, headPitch: 7, headRoll: 6, headYaw: -6, headZ: -2, torsoLean: -2, torsoShrug: -10, torsoY: -2, torsoRoll: -2, eyeScale: 0.3, pupil: -0.3, squash: -0.2 },
    { name: 'dread', AU1: 0.8, AU2: 0.4, AU4: 0.5, AU5: 0.8, AU20: 0.4, AU24: 0.3, headYaw: 8, gazeX: -0.5, headZ: -2, torsoYaw: 6, torsoShrug: -7, torsoLean: -2.5, eyeScale: 0.4, pupil: -0.35 },
  ],
  DIS: [
    { name: 'yuck', AU9: 1.0, AU10: 0.8, AU4: 0.7, AU7: 0.7, AU15: 0.4, AU17: 0.3, mouthAsym: 0.5, browAsym: -0.3, headZ: -2.5, headYaw: -8, headRoll: 6, headPitch: -3, torsoLean: -3, torsoYaw: -5, torsoShrug: -4 },
    { name: 'sneer', AU9: 0.7, AU10: 1.0, AU4: 0.6, AU7: 0.5, AU14: 0.3, mouthAsym: 0.8, browAsym: -0.6, headRoll: -7, headPitch: -5, gazeY: -0.3, AU43: 0.3, torsoLean: -2, torsoRoll: -1.5 },
    { name: 'repulsed', AU9: 0.9, AU10: 0.6, AU4: 0.8, AU6: 0.5, AU7: 0.9, AU15: 0.7, AU17: 0.5, AU16: 0.3, headPitch: 5, headYaw: 10, headZ: -3, torsoLean: -3.5, torsoShrug: -6, torsoYaw: 4, squash: -0.2 },
  ],
};

// How each emotion moves: spring tempo / damping, accent recipe, blink rhythm, breathing, what the eyes do in pauses.
export const ACTING = {
  NEU: { tempo: 1.0, zeta: 0.55, accents: ['nod', 'nod', 'tilt', 'brow'], amp: 0.8, blink: 3.9, blinkDur: 1, breath: 3.8, breathAmp: 0.5, drift: 1.0, tremble: 0, relax: 0.55, look: 'think', browF0: 1.0, lead: 8 },
  HAP: { tempo: 1.08, zeta: 0.42, accents: ['bounce', 'bounce', 'tilt', 'up'], amp: 1.0, blink: 3.3, blinkDur: 1, breath: 3.2, breathAmp: 0.7, drift: 1.2, tremble: 0, relax: 0.7, look: 'glance', browF0: 1.2, lead: 8 },
  SAD: { tempo: 0.6, zeta: 0.9, accents: ['droop', 'sway', 'droop'], amp: 0.6, blink: 5.2, blinkDur: 1.7, breath: 5.0, breathAmp: 0.9, drift: 0.6, tremble: 0, relax: 0.85, look: 'down', browF0: 0.5, lead: 11 },
  ANG: { tempo: 1.2, zeta: 0.5, accents: ['thrust', 'thrust', 'shake'], amp: 1.15, blink: 7.0, blinkDur: 0.9, breath: 2.4, breathAmp: 1.0, drift: 0.5, tremble: 0.15, relax: 0.8, look: 'stare', browF0: 0.0, lead: 7 },
  FEA: { tempo: 1.35, zeta: 0.4, accents: ['recoil', 'jerk', 'recoil'], amp: 0.9, blink: 2.4, blinkDur: 0.8, breath: 1.9, breathAmp: 1.1, drift: 0.7, tremble: 1.0, relax: 0.8, look: 'dart', browF0: 0.6, lead: 5 },
  DIS: { tempo: 0.88, zeta: 0.6, accents: ['away', 'nod', 'away'], amp: 0.85, blink: 4.6, blinkDur: 1.2, breath: 3.6, breathAmp: 0.6, drift: 0.7, tremble: 0, relax: 0.75, look: 'aside', browF0: 0.2, lead: 8 },
};

// default acting personality (characters override: see avatar/characters.js `acting`)
export const DEFAULT_PROFILE = { energy: 1, tempo: 1, bounce: 1, tilt: 1, asym: 1, brow: 1, eye: 1, body: 1 };

// ---------------------------------------------------------------------------------------------------------
// Accent gestures: additive keyframed curves around the "hit" frame (time in frames, value = share of amplitude).
// Every accent has anticipation (a small move the other way), a fast hit, and a settle with a small rebound.
// ---------------------------------------------------------------------------------------------------------
function gesture(kind, A, side, key, prof, fast = false) {
  const B = prof.body * (key ? 1 : 0.45), q = prof.bounce;
  const c = {};
  // live mode has no look-ahead: the anticipation is compressed so the hit still lands on the syllable
  const k = (id, pts, s = 1) => { const h = id === 'headPitch' || id === 'headYaw' || id === 'headRoll' ? HEAD_ACCENT : 1; c[id] = pts.map(([t, v]) => [fast && t < 0 ? t * 0.45 : t, v * s * h]); };
  switch (kind) {
    case 'nod':
      k('headPitch', [[-7, 0], [-3, -0.32], [0, 1], [4, -0.14 * q], [9, 0.04], [14, 0]], 6.5 * A);
      k('headZ', [[-7, 0], [-3, -0.3], [0, 1], [6, 0.1], [12, 0]], 1.3 * A);
      k('AU2', [[-6, 0], [-2, 1], [3, 0.85], [11, 0]], 0.3 * A * prof.brow); k('AU1', [[-6, 0], [-2, 1], [3, 0.85], [11, 0]], 0.22 * A * prof.brow);
      k('torsoLean', [[-5, 0], [-1, -0.25], [2, 1], [8, -0.1 * q], [15, 0]], 1.6 * A * B);
      k('squash', [[-5, 0], [-2, 0.35], [1, -0.5], [5, 0.15 * q], [10, 0]], 0.5 * A * q * (key ? 1 : 0.5));
      break;
    case 'bounce': // happy: springy up-down with a tilt and a brow flash
      k('headPitch', [[-7, 0], [-3, -0.45], [0, 1], [4, -0.3 * q], [8, 0.12 * q], [13, 0]], 6 * A);
      k('headRoll', [[-6, 0], [0, 1], [5, 0.75], [14, 0]], 4.5 * A * side * prof.tilt);
      k('headY', [[-7, 0], [-3, 0.5], [0, -1], [4, 0.35 * q], [9, 0]], 1.4 * A);
      k('AU2', [[-6, 0], [-2, 1], [4, 0.9], [12, 0]], 0.38 * A * prof.brow); k('AU1', [[-6, 0], [-2, 1], [4, 0.9], [12, 0]], 0.25 * A * prof.brow);
      k('AU5', [[-5, 0], [-1, 1], [4, 0.6], [10, 0]], 0.25 * A * prof.eye);
      k('torsoY', [[-6, 0], [-2, 0.6], [1, -1], [5, 0.45 * q], [10, -0.1 * q], [15, 0]], 2.6 * A * B);
      k('torsoShrug', [[-5, 0], [-1, -1], [5, 0.2], [12, 0]], 2.5 * A * B);
      k('squash', [[-6, 0], [-3, 0.6], [1, -0.75], [5, 0.3 * q], [10, 0]], 0.6 * A * q * (key ? 1 : 0.55));
      break;
    case 'up': // chin-up emphasis (delight, question)
      k('headPitch', [[-7, 0], [-3, 0.35], [0, -1], [5, 0.12 * q], [12, 0]], 6 * A);
      k('headRoll', [[-6, 0], [0, 1], [6, 0.7], [15, 0]], 3.5 * A * side * prof.tilt);
      k('AU2', [[-5, 0], [-1, 1], [5, 0.9], [13, 0]], 0.45 * A * prof.brow); k('AU1', [[-5, 0], [-1, 1], [5, 0.9], [13, 0]], 0.35 * A * prof.brow);
      k('AU5', [[-5, 0], [-1, 1], [5, 0.7], [12, 0]], 0.35 * A * prof.eye);
      k('torsoY', [[-6, 0], [-3, -0.5], [0, 1], [6, -0.15 * q], [13, 0]], 2.2 * A * B);
      k('squash', [[-6, 0], [-3, -0.5], [0, 0.8], [5, -0.15 * q], [11, 0]], 0.6 * A * q * (key ? 1 : 0.5));
      break;
    case 'tilt': // side accent
      k('headRoll', [[-6, 0], [-3, -0.3], [0, 1], [5, 0.8], [16, 0]], 6 * A * side * prof.tilt);
      k('headYaw', [[-6, 0], [0, 1], [6, 0.7], [16, 0]], 4 * A * -side);
      k('headPitch', [[-6, 0], [-3, -0.3], [0, 1], [5, 0.2], [12, 0]], 3 * A);
      k('browAsym', [[-5, 0], [-1, 1], [6, 0.8], [14, 0]], 0.5 * A * side * prof.asym);
      k('AU2', [[-5, 0], [-1, 1], [5, 0.7], [12, 0]], 0.2 * A * prof.brow);
      k('torsoRoll', [[-4, 0], [2, 1], [8, 0.7], [18, 0]], 1.5 * A * side * B);
      break;
    case 'brow': // brow flash with a small head lift
      k('AU2', [[-5, 0], [-1, 1], [5, 0.9], [12, 0]], 0.5 * A * prof.brow); k('AU1', [[-5, 0], [-1, 1], [5, 0.9], [12, 0]], 0.4 * A * prof.brow);
      k('AU5', [[-5, 0], [-1, 1], [4, 0.6], [10, 0]], 0.25 * A * prof.eye);
      k('headPitch', [[-6, 0], [-2, 0.25], [1, -1], [7, 0.1], [13, 0]], 3 * A);
      break;
    case 'thrust': // angry: jab forward and down, brows clamp, no softening rebound
      k('headZ', [[-5, 0], [-2, -0.45], [0, 1], [3, 0.7], [9, 0.15], [14, 0]], 2.6 * A);
      k('headPitch', [[-5, 0], [-2, -0.3], [0, 1], [3, 0.6], [10, 0]], 3.8 * A);
      k('AU4', [[-4, 0], [0, 1], [5, 0.7], [12, 0]], 0.3 * A); k('AU7', [[-4, 0], [0, 1], [5, 0.6], [12, 0]], 0.3 * A);
      k('AU9', [[-3, 0], [0, 1], [4, 0.5], [10, 0]], 0.3 * A);
      k('torsoLean', [[-4, 0], [-1, -0.35], [1, 1], [5, 0.6], [14, 0]], 3.2 * A * B);
      k('torsoShrug', [[-4, 0], [0, -1], [6, -0.4], [14, 0]], 2.5 * A * B);
      k('squash', [[-4, 0], [-2, 0.3], [1, -0.6], [5, 0]], 0.5 * A * (key ? 1 : 0.5));
      break;
    case 'shake': // emphatic "no"
      k('headYaw', [[-5, 0], [0, 1], [6, -0.85], [12, 0.5], [18, -0.2], [24, 0]], 5.5 * A * side);
      k('headPitch', [[-5, 0], [0, 1], [12, 0.6], [22, 0]], 2.5 * A);
      k('AU4', [[-3, 0], [1, 1], [12, 0.6], [20, 0]], 0.2 * A);
      k('torsoYaw', [[-3, 0], [2, 1], [8, -0.8], [14, 0.4], [24, 0]], 2.5 * A * side * B);
      break;
    case 'droop': // sad: slow heavy sink, inner brows pinch up
      k('headPitch', [[-10, 0], [-5, -0.15], [2, 1], [12, 0.75], [26, 0]], 5 * A);
      k('headRoll', [[-8, 0], [3, 1], [14, 0.8], [28, 0]], 3 * A * side * prof.tilt);
      k('AU1', [[-8, 0], [0, 1], [10, 0.8], [24, 0]], 0.25 * A * prof.brow);
      k('AU43', [[-6, 0], [2, 1], [12, 0.7], [24, 0]], 0.25 * A);
      k('torsoY', [[-8, 0], [4, -1], [16, -0.7], [30, 0]], 1.8 * A * B);
      break;
    case 'sway': // sad: slow small shake
      k('headYaw', [[-8, 0], [0, 1], [9, -0.8], [18, 0.4], [28, 0]], 4.5 * A * side);
      k('headPitch', [[-8, 0], [2, 1], [18, 0.7], [28, 0]], 2.5 * A);
      k('AU1', [[-6, 0], [2, 1], [14, 0.7], [24, 0]], 0.2 * A * prof.brow);
      break;
    case 'recoil': // fear: jerk back, eyes pop, shoulders jump
      k('headZ', [[-3, 0], [0, -1], [3, -0.7], [8, -0.25], [14, 0]], 2.6 * A);
      k('headPitch', [[-3, 0], [0, -1], [4, -0.5], [12, 0]], 4 * A);
      k('AU5', [[-3, 0], [0, 1], [5, 0.7], [12, 0]], 0.4 * A * prof.eye); k('eyeScale', [[-3, 0], [0, 1], [5, 0.6], [12, 0]], 0.35 * A * prof.eye);
      k('pupil', [[-3, 0], [0, -1], [6, -0.6], [14, 0]], 0.35 * A);
      k('AU1', [[-3, 0], [0, 1], [6, 0.6], [12, 0]], 0.25 * A * prof.brow);
      k('torsoShrug', [[-3, 0], [0, -1], [5, -0.6], [14, 0]], 4 * A * B);
      k('torsoLean', [[-2, 0], [2, -1], [7, -0.6], [15, 0]], 2.5 * A * B);
      k('squash', [[-3, 0], [-1, -0.5], [1, 0.9], [5, -0.2], [9, 0]], 0.6 * A * (key ? 1 : 0.5));
      break;
    case 'jerk': // fear: quick sideways flinch
      k('headYaw', [[-4, 0], [0, 1], [4, 0.65], [8, 0.8], [15, 0]], 5.5 * A * side);
      k('headRoll', [[-3, 0], [0, 1], [6, 0.6], [13, 0]], 3 * A * -side);
      k('gazeX', [[-4, 0], [-2, 1], [4, 1], [7, 0]], 0.35 * A * side);
      k('AU5', [[-3, 0], [0, 1], [5, 0.6], [11, 0]], 0.3 * A * prof.eye);
      k('torsoShrug', [[-3, 0], [0, -1], [6, -0.5], [14, 0]], 3 * A * B);
      break;
    case 'away': // disgust: turn away and pull back, nose wrinkles
      k('headYaw', [[-6, 0], [-3, -0.2], [0, 1], [6, 0.8], [18, 0]], 6.5 * A * -side);
      k('headZ', [[-5, 0], [0, -1], [7, -0.7], [18, 0]], 1.8 * A);
      k('headRoll', [[-5, 0], [0, 1], [7, 0.7], [18, 0]], 3.5 * A * side * prof.tilt);
      k('AU9', [[-4, 0], [0, 1], [6, 0.7], [14, 0]], 0.35 * A); k('AU10', [[-4, 0], [0, 1], [6, 0.7], [14, 0]], 0.3 * A);
      k('AU7', [[-4, 0], [0, 1], [6, 0.6], [14, 0]], 0.25 * A);
      k('torsoLean', [[-4, 0], [2, -1], [9, -0.7], [20, 0]], 2 * A * B);
      break;
    default: break;
  }
  return c;
}
// x unchanged up to `a`, then eased toward the limit L
const soft = (x, a, L) => (x <= a ? x : a + (L - a) * Math.tanh((x - a) / (L - a)));
function evalCurve(pts, dt) {
  if (dt <= pts[0][0] || dt >= pts[pts.length - 1][0]) return 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    if (dt >= a[0] && dt <= b[0]) return lerp(a[1], b[1], smooth((dt - a[0]) / (b[0] - a[0])));
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------------
// Performer: plays timed events, frame by frame.
// ---------------------------------------------------------------------------------------------------------
// channel classes: [frequency Hz, damping ratio, delay in frames after a pose event] -> overlapping action
const DYN = { eye: [4.2, 0.62, 0], brow: [3.7, 0.52, 0], nose: [3.4, 0.6, 1], mouth: [3.3, 0.6, 1], head: [2.0, 0.5, 2], squash: [3.4, 0.36, 1], torso: [1.4, 0.55, 4], pupil: [2.2, 0.8, 0] };
const CLS = {};
for (const c of CHANNELS) CLS[c.id] = c.group === 'gaze' ? 'eye' : c.group;
Object.assign(CLS, { squash: 'squash', pupil: 'pupil', eyeScale: 'eye', AU43: 'eye', AU6: 'eye' });
const VISEME = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];
// mouth-pose channels that must give way while the mouth is busy talking
const SPEAK_MUTE = { AU24: 1.0, AU17: 0.8, AU28: 1.0, AU15: 0.3, AU14: 0.4 };
const BLINK = [0.55, 1.0, 1.0, 0.5, 0.18, 0.04];
const ANTIC = ['headPitch', 'headRoll', 'headYaw', 'headZ', 'torsoLean', 'torsoY', 'torsoYaw', 'torsoRoll'];
const HEAD_ROT = ['headPitch', 'headYaw', 'headRoll'];

export class Performer {
  constructor(opts = {}) {
    this.prof = { ...DEFAULT_PROFILE, ...(opts.profile || {}) };
    this.r = rng((opts.seed ?? 7) ^ 0x9e3779b9);
    this.x = new Float32Array(K); this.v = new Float32Array(K); this.goal = new Float32Array(K);
    for (const id in IDLE) { this.x[IDX[id]] = IDLE[id]; this.goal[IDX[id]] = IDLE[id]; }
    this.tempo = 1; this.zeta = ACTING.NEU.zeta; this.act = ACTING.NEU;
    this.pending = []; // delayed goal changes {t, k, v}
    this.gest = [];    // active additive gestures {t0, curves, end}
    this.blinkT = -1; this.blinkDur = 1; this.blinkAmt = 1;
    this.gaze = [0, 0]; this.gazeGoal = [0, 0]; this.micro = [0, 0]; this.nextMicro = 20;
    this.sp = 0;       // smoothed "is talking"
    this.jaw = 0; this.vis = {}; for (const id of VISEME) this.vis[id] = 0;
    this.breath = this.r() * 6.28; this.drift = new Float32Array(10); this.drift2 = new Float32Array(10);
    this.speakEmo = 'NEU'; this.inten = 0.6;
    this.t = 0;
    this.out = new Float32Array(K); this.prevHead = null;
  }
  // ---- events ----
  setPose(pose, t, opts = {}) { // pose: {id: value}; unspecified channels go to 0 unless opts.partial
    const ds = opts.stagger ?? 1;
    const seen = new Set();
    for (const id in pose) { const k = IDX[id]; if (k === undefined) continue; seen.add(k); this.pending.push({ t: t + Math.round(DYN[CLS[id]][2] * ds), k, v: pose[id] }); }
    if (!opts.partial) for (let k = 0; k < K; k++) { const id = IDS[k]; if (seen.has(k) || id === 'AU45' || VISEME.includes(id) && !(id in pose) && false) continue; if (!seen.has(k)) this.pending.push({ t: t + Math.round(DYN[CLS[id]][2] * ds), k, v: 0 }); }
  }
  setActing(emo, inten) { this.act = ACTING[emo] || ACTING.NEU; this.tempo = this.act.tempo; this.zeta = this.act.zeta; this.speakEmo = emo; this.inten = inten; }
  addGesture(t0, curves) { let end = 0; for (const id in curves) end = Math.max(end, curves[id][curves[id].length - 1][0]); this.gest.push({ t0, curves, end }); }
  // anticipation: head and body go a little the other way from where the next pose will take them (measured from
  // where they are now), while the face gathers; the pose event that follows overrides all of it
  anticipate(to, amount = 0.3, face = {}) {
    for (const id of ANTIC) { const k = IDX[id], d = (to[id] || 0) - this.goal[k]; this.pending.push({ t: this.t + Math.round(DYN[CLS[id]][2] * 0.5), k, v: this.goal[k] - amount * d }); }
    for (const id in face) { const k = IDX[id]; this.pending.push({ t: this.t, k, v: id === 'squash' ? face[id] : Math.max(this.goal[k], 0) + face[id] }); }
  }
  blink(dur = 1, amt = 1) { if (this.blinkT < 0) { this.blinkT = 0; this.blinkDur = dur; this.blinkAmt = amt; } }
  lookAt(x, y) { this.gazeGoal = [x, y]; }
  event(ev) {
    switch (ev.type) {
      case 'pose': this.setPose(ev.pose, this.t, ev); break;
      case 'act': this.setActing(ev.emo, ev.inten); break;
      case 'gest': this.addGesture(ev.hit ?? this.t, ev.curves); break;
      case 'blink': this.blink(ev.dur, ev.amt); break;
      case 'gaze': this.lookAt(ev.x, ev.y); break;
      case 'antic': this.anticipate(ev.to, ev.amount, ev.face); break;
      default: break;
    }
  }
  // ---- one frame ----
  // inp: { speaking: 0|1, vis: {AU26..} viseme targets in rig units (or null), browF0: number, tex: [pitch,yaw,roll] }
  step(inp = {}) {
    const t = this.t, r = this.r, P = this.prof, act = this.act, dt = 1 / FPS;
    // delayed goal changes (overlap between face, head and torso)
    if (this.pending.length) { const keep = []; for (const p of this.pending) { if (p.t <= t) this.goal[p.k] = p.v; else keep.push(p); } this.pending = keep; }
    // springs: every channel overshoots a little and settles (amount depends on the emotion and the character)
    const sub = 2, h = dt / sub;
    for (let k = 0; k < K; k++) {
      const d = DYN[CLS[IDS[k]]]; if (!d) continue;
      const w = 2 * Math.PI * d[0] * this.tempo * P.tempo, z = clamp(d[1] * (this.zeta / 0.55) / (0.6 + 0.4 * P.bounce), 0.25, 1.1);
      let x = this.x[k], v = this.v[k]; const g = this.goal[k];
      for (let s = 0; s < sub; s++) { v += (w * w * (g - x) - 2 * z * w * v) * h; x += v * h; }
      this.x[k] = x; this.v[k] = v;
    }
    const y = this.out; y.set(this.x);
    // speaking state
    const speaking = inp.speaking ? 1 : 0;
    this.sp += (speaking - this.sp) * (speaking ? 0.5 : 0.25);
    for (const id in SPEAK_MUTE) y[IDX[id]] *= 1 - SPEAK_MUTE[id] * this.sp;
    // additive gestures (accents)
    if (this.gest.length) {
      const keep = [];
      for (const g of this.gest) {
        const d = t - g.t0; if (d > g.end) continue; keep.push(g);
        for (const id in g.curves) y[IDX[id]] += evalCurve(g.curves[id], d);
      }
      this.gest = keep;
    }
    // brows ride the pitch of the voice a little; head keeps a trace of the learned natural motion
    if (inp.browF0) { const b = inp.browF0 * act.browF0 * P.brow; y[IDX.AU1] += 0.6 * b; y[IDX.AU2] += b; }
    if (inp.tex) { y[IDX.headPitch] += inp.tex[0]; y[IDX.headYaw] += inp.tex[1]; y[IDX.headRoll] += inp.tex[2]; }
    // ---- life: breathing, slow drift (moving hold), tremble ----
    this.breath += 2 * Math.PI / (act.breath * FPS);
    const br = Math.sin(this.breath) * act.breathAmp * (1 - 0.5 * this.sp);
    y[IDX.torsoY] += 0.9 * br; y[IDX.torsoShrug] -= 0.7 * Math.sin(this.breath - 0.4) * act.breathAmp; y[IDX.headPitch] -= 0.5 * Math.sin(this.breath - 0.9) * act.breathAmp;
    const DR = [['headPitch', 1.1], ['headYaw', 1.4], ['headRoll', 0.9], ['torsoRoll', 0.45], ['torsoShift', 0.5], ['torsoLean', 0.4], ['AU1', 0.025], ['AU2', 0.025], ['AU12', 0.03], ['headZ', 0.3]];
    for (let i = 0; i < DR.length; i++) {
      this.drift[i] = 0.965 * this.drift[i] + 0.262 * gauss(r); this.drift2[i] = 0.93 * this.drift2[i] + 0.07 * this.drift[i];
      y[IDX[DR[i][0]]] += this.drift2[i] * 2.2 * DR[i][1] * act.drift;
    }
    if (act.tremble > 0) { const tr = act.tremble * this.inten * (0.6 + 0.4 * Math.sin(t * 0.21)); y[IDX.headYaw] += 0.55 * tr * Math.sin(t * 1.9); y[IDX.headRoll] += 0.35 * tr * Math.sin(t * 2.3 + 1); y[IDX.torsoShrug] += 0.5 * tr * Math.sin(t * 2.1 + 2); }
    // ---- lip sync: pose-to-pose mouth, quick to open, a touch slower to close, shaped by the emotion ----
    if (inp.vis) {
      // the mouth only shapes sound while there is speech: between phrases it returns to the pose's own mouth
      const live = inp.mouth !== undefined ? inp.mouth : Math.min(1, this.sp * 1.5);
      const jt = clamp((inp.vis.AU26 || 0) * live, 0, 1.15);
      this.jaw = jt >= this.jaw ? lerp(this.jaw, jt, 0.85) : Math.max(jt, this.jaw - (jt < 0.1 ? 0.34 : 0.17));
      for (const id of VISEME) if (id !== 'AU26') this.vis[id] += ((inp.vis[id] || 0) * live - this.vis[id]) * 0.7;
      const j = this.jaw, I = this.inten, e = this.speakEmo;
      const jawScale = e === 'SAD' ? 0.78 : e === 'ANG' ? 1.12 : e === 'FEA' ? 1.05 : 1;
      y[IDX.AU26] += Math.min(0.95, j * jawScale);
      for (const id of VISEME) if (id !== 'AU26') y[IDX[id]] += Math.min(0.85, this.vis[id]);
      if (e === 'ANG') { y[IDX.AU10] += 0.45 * j * I; y[IDX.AU16] += 0.5 * j * I; y[IDX.AU20] += 0.25 * j * I; }
      else if (e === 'HAP') { y[IDX.AU20] += 0.15 * j * I; y[IDX.AU12] += 0.08 * j; }
      else if (e === 'FEA') { y[IDX.AU20] += 0.3 * j * I; y[IDX.AU16] += 0.2 * j * I; }
      else if (e === 'DIS') { y[IDX.AU10] += 0.25 * j * I; }
      // a wide-open mouth stretches the face a little
      y[IDX.squash] += 0.22 * Math.max(0, j - 0.35) * P.bounce;
    }
    // ---- eyes: saccadic gaze with tiny darts, blink as an event ----
    if (--this.nextMicro <= 0) { const a = act.look === 'dart' ? 0.12 : 0.035; this.micro = [gauss(r) * a, gauss(r) * a * 0.6]; this.nextMicro = act.look === 'dart' ? 6 + Math.floor(r() * 14) : 16 + Math.floor(r() * 40); }
    for (let i = 0; i < 2; i++) this.gaze[i] += (this.gazeGoal[i] + this.micro[i] - this.gaze[i]) * 0.7;
    y[IDX.gazeX] += this.gaze[0]; y[IDX.gazeY] += this.gaze[1];
    let b = 0;
    if (this.blinkT >= 0) {
      const u = this.blinkT / this.blinkDur, i0 = Math.floor(u), f = u - i0;
      b = i0 >= BLINK.length ? 0 : lerp(BLINK[i0], BLINK[i0 + 1] ?? 0, f) * this.blinkAmt;
      this.blinkT++; if (u >= BLINK.length) this.blinkT = -1;
      y[IDX.AU4] += 0.08 * b; y[IDX.AU6] += 0.1 * b; // the brow dips and the cheek lifts with the lid
    }
    y[IDX.AU45] = b;
    // ---- limits ----
    // the head saturates softly: stacked poses and accents must never hide the face or break the neck
    // (chin up is limited more than chin down: seen from the front, a big head tipped far back shows mostly chin)
    y[IDX.headPitch] = y[IDX.headPitch] > 0 ? soft(y[IDX.headPitch], 9, 14) : -soft(-y[IDX.headPitch], 8, 12.5);
    y[IDX.headRoll] = Math.sign(y[IDX.headRoll]) * soft(Math.abs(y[IDX.headRoll]), 10, 15);
    y[IDX.headYaw] = Math.sign(y[IDX.headYaw]) * soft(Math.abs(y[IDX.headYaw]), 15, 22);
    y[IDX.torsoLean] = Math.sign(y[IDX.torsoLean]) * soft(Math.abs(y[IDX.torsoLean]), 4.5, 7.5);
    y[IDX.squash] = Math.sign(y[IDX.squash]) * soft(Math.abs(y[IDX.squash]), 0.45, 0.8);
    for (let k = 0; k < K; k++) { const c = CHANNELS[k]; const lo = c.range[0], hi = c.range[1]; y[k] = clamp(y[k], lo < 0 ? lo * 1.3 : 0, hi * (hi === 1 ? 1.25 : 1.3)); }
    // the head never turns faster than a quick take (about 160 deg/s): when moves stack, the turn spreads over more frames
    if (!this.prevHead) this.prevHead = HEAD_ROT.map((id) => y[IDX[id]]);
    HEAD_ROT.forEach((id, i) => { const k = IDX[id], d = y[k] - this.prevHead[i]; y[k] = this.prevHead[i] + Math.sign(d) * soft(Math.abs(d), 2.8, 4.2); this.prevHead[i] = y[k]; });
    this.t++;
    return y;
  }
}

// ---------------------------------------------------------------------------------------------------------
// Speech analysis shared by the offline and live directors.
// ---------------------------------------------------------------------------------------------------------
const speechFlag = (F, t) => (F[t * FEAT_DIM + 1] > 0.5 || F[t * FEAT_DIM] > -1.5 ? 1 : 0);

export function analyzeSpeech(F, T, aux = null, auxDim = 3, cuts = null) {
  const raw = new Uint8Array(T); for (let t = 0; t < T; t++) raw[t] = speechFlag(F, t);
  // close short gaps, drop tiny islands
  const sp = Uint8Array.from(raw);
  for (let t = 0; t < T;) { if (!sp[t]) { let e = t; while (e < T && !sp[e]) e++; if (t > 0 && e < T && e - t <= 4) sp.fill(1, t, e); t = e; } else t++; }
  for (let t = 0; t < T;) { if (sp[t]) { let e = t; while (e < T && sp[e]) e++; if (e - t < 3) sp.fill(0, t, e); t = e; } else t++; }
  // phrases: speech runs separated by pauses of >= 0.3 s
  const runs = []; for (let t = 0; t < T;) { if (sp[t]) { let e = t; while (e < T && sp[e]) e++; runs.push([t, e]); t = e; } else t++; }
  const phrases = [];
  for (const [a, b] of runs) { const last = phrases[phrases.length - 1]; if (last && a - last.t1 < 9) last.t1 = b; else phrases.push({ t0: a, t1: b }); }
  for (let i = phrases.length - 1; i >= 0; i--) if (phrases[i].t1 - phrases[i].t0 < 6) { sp.fill(0, phrases[i].t0, phrases[i].t1); phrases.splice(i, 1); } // a click or a breath is not a phrase
  // where the feeling is known to change inside a phrase (a mark on the timeline begins or ends there), the phrase is
  // played as two: cut at the quietest moment nearby, and only when both halves are long enough to act
  if (cuts) for (const c0 of cuts) {
    const i = phrases.findIndex((ph) => c0 - ph.t0 >= 15 && ph.t1 - c0 >= 15); if (i < 0) continue;
    let c = c0, lo = Infinity; for (let t = c0 - 6; t <= c0 + 6; t++) if (F[t * FEAT_DIM] < lo) { lo = F[t * FEAT_DIM]; c = t; }
    phrases.splice(i, 1, { t0: phrases[i].t0, t1: c }, { t0: c, t1: phrases[i].t1 });
  }
  // smoothed energy and pitch
  const en = new Float32Array(T), f0 = new Float32Array(T);
  for (let t = 0; t < T; t++) { let s = 0, c = 0; for (let j = -1; j <= 1; j++) { const u = t + j; if (u >= 0 && u < T) { s += F[u * FEAT_DIM] * (j ? 1 : 2); c += j ? 1 : 2; } } en[t] = s / c; }
  { let last = 0; for (let t = 0; t < T; t++) { if (F[t * FEAT_DIM + 1] > 0.5) last = F[t * FEAT_DIM + 2]; f0[t] = last; } for (let t = T - 2; t >= 0; t--) if (F[t * FEAT_DIM + 1] <= 0.5 && F[(t + 1) * FEAT_DIM + 1] > 0.5 && f0[t] === 0) f0[t] = f0[t + 1]; }
  const f0s = new Float32Array(T); for (let t = 0; t < T; t++) { let s = 0, c = 0; for (let j = -2; j <= 2; j++) { const u = t + j; if (u >= 0 && u < T) { s += f0[u]; c++; } } f0s[t] = s / c; }
  // syllable nuclei = energy peaks inside speech; prominence from energy, pitch and the learned accent signals
  for (const ph of phrases) {
    let me = 0, mf = 0, n = 0; for (let t = ph.t0; t < ph.t1; t++) { me += en[t]; mf += f0s[t]; n++; } me /= n; mf /= n;
    ph.meanE = me; ph.meanF0 = mf;
    const nuc = [];
    for (let t = Math.max(1, ph.t0); t < Math.min(T - 1, ph.t1); t++) {
      if (!(en[t] >= en[t - 1] && en[t] > en[t + 1]) || en[t] < -1.3) continue;
      let lo = en[t]; for (let j = 1; j <= 6; j++) { if (t - j >= 0) lo = Math.min(lo, en[t - j]); } let lo2 = en[t]; for (let j = 1; j <= 6; j++) { if (t + j < T) lo2 = Math.min(lo2, en[t + j]); }
      const prom = en[t] - Math.max(lo, lo2); if (prom < 0.1) continue;
      let ah = 0, ab = 0, emph; if (aux) { for (let j = -2; j <= 2; j++) { const u = clamp(t + j, 0, T - 1); ah = Math.max(ah, aux[u * auxDim]); ab = Math.max(ab, aux[u * auxDim + 1]); } emph = emphasisAt(aux, t, T, auxDim); }
      const score = 1.6 * (en[t] - me) + 0.8 * prom + 0.55 * (f0s[t] - mf) + 0.45 * ah + 0.3 * ab;
      if (nuc.length && t - nuc[nuc.length - 1].t < 3) { if (score > nuc[nuc.length - 1].score) nuc[nuc.length - 1] = { t, score, emph }; } else nuc.push({ t, score, emph });
    }
    ph.nuclei = nuc;
    // question / rising ending: pitch at the end vs. the middle of the phrase
    const L = ph.t1 - ph.t0; let endF = 0, ne = 0, midF = 0, nm = 0;
    for (let t = ph.t0; t < ph.t1; t++) { if (F[t * FEAT_DIM + 1] <= 0.5) continue; const u = (t - ph.t0) / L; if (u > 0.78) { endF += F[t * FEAT_DIM + 2]; ne++; } else if (u > 0.25 && u < 0.7) { midF += F[t * FEAT_DIM + 2]; nm++; } }
    ph.rise = ne > 2 && nm > 3 ? endF / ne - midF / nm : 0;
  }
  return { speech: sp, phrases, en, f0: f0s };
}

// Emphasis learned from the actors (how strongly a performer's head or brows would move here, in standard deviations
// of that performer's own motion): the stronger of the head and brow signals around the syllable.
export const ACCENT_STRONG = 0.8, ACCENT_SOFT = 0.5;
function emphasisAt(aux, t, T, auxDim = 3) { let v = -9; for (let j = -1; j <= 1; j++) { const u = clamp(t + j, 0, T - 1); v = Math.max(v, aux[u * auxDim], 0.9 * aux[u * auxDim + 1]); } return v; }
const accentStrength = (emph) => clamp(0.3 + 0.45 * emph, 0.3, 1);

function pickAccents(ph, r) {
  if (!ph.nuclei.length) return [];
  const dur = (ph.t1 - ph.t0) / FPS;
  if (ph.nuclei[0].emph !== undefined) {
    // Which syllables get a gesture is decided by the learned emphasis, not by loudness alone: a strong accent wherever a
    // performer would stress, one soft accent in a line that is delivered evenly, none in a flat one. Few, clear accents
    // with holds between them read as acting; a head that hits every syllable reads as a bobble.
    const cand = ph.nuclei.slice().sort((a, b) => b.emph - a.emph), max = Math.max(1, Math.round(dur / 0.7)), out = [];
    for (const n of cand) {
      if (out.length >= max) break;
      if (out.some((o) => Math.abs(o.t - n.t) < 12)) continue;
      if (n.emph >= ACCENT_STRONG || (!out.length && n.emph >= ACCENT_SOFT)) out.push({ t: n.t, strength: accentStrength(n.emph), key: out.length === 0, emph: n.emph });
    }
    return out.sort((a, b) => a.t - b.t);
  }
  const nuc = ph.nuclei.slice().sort((a, b) => b.score - a.score);
  const max = Math.max(1, Math.round(dur / 0.55));
  const mean = nuc.reduce((s, n) => s + n.score, 0) / nuc.length, sd = Math.sqrt(nuc.reduce((s, n) => s + (n.score - mean) ** 2, 0) / nuc.length) + 0.25;
  const out = [];
  for (const n of nuc) {
    if (out.length >= max) break;
    if (out.some((o) => Math.abs(o.t - n.t) < 10)) continue;
    if (out.length && n.score < mean - 0.2 * sd) continue;
    out.push({ t: n.t, strength: clamp(0.55 + 0.25 * (n.score - mean) / sd, 0.25, 1), key: out.length === 0 });
  }
  return out.sort((a, b) => a.t - b.t);
}

// how far key poses and accents are pushed beyond the written values (tuned against the animated-film statistics:
// head range p95 about 13 / 15 / 9 degrees pitch / yaw / roll, against 5 / 3 / 3 for the seated actors).
// After the third self-evaluation the head was taken back (calibrated on the validation actors only): head moves were
// about twice as large and half again as fast as in the tracked films. Accents now turn the head 0.8 as far, yaw and
// roll of the poses are pushed less, a new phrase changes side less often, and the turn rate is limited lower.
const PUSH = { headPitch: 1.3, headYaw: 1.1, headRoll: 1.15, headZ: 1.2, torso: 1.3 };
const ACCENT_PUSH = 1.25;
const HEAD_ACCENT = 0.8;   // head rotation of accent gestures, as a share of the written curves
const ACCENT_MAX = 1.45;
function instPose(base, side, inten, prof) {
  const o = {};
  for (const id in base) {
    if (id === 'name') continue;
    let v = base[id] * (SIDED.has(id) ? side : 1);
    const g = CHANNELS[IDX[id]].group;
    // exaggeration is not linear: strong poses are pushed, mild ones stay readable
    let s = inten;
    if (g === 'head') s *= prof.energy * (id === 'headRoll' || id === 'headYaw' ? prof.tilt : 1) * (PUSH[id] || 1);
    else if (g === 'torso') s *= prof.energy * prof.body * PUSH.torso;
    else if (id === 'browAsym' || id === 'mouthAsym') s = Math.min(1.2, inten * prof.asym);
    else if (g === 'brow') s *= prof.brow;
    else if (id === 'eyeScale' || id === 'pupil' || id === 'AU5') s *= prof.eye;
    else if (id === 'squash') s *= prof.bounce;
    o[id] = v * s;
  }
  return o;
}
function blend(a, b, w) { const o = {}; for (const id in a) o[id] = a[id] * (1 - w); for (const id in b) o[id] = (o[id] || 0) + b[id] * w; return o; }
function scalePose(p, s, keep = null) { const o = {}; for (const id in p) o[id] = p[id] * (keep && keep.has(id) ? 1 : s); return o; }

// ---------------------------------------------------------------------------------------------------------
// Offline director: plans the whole performance with look-ahead and renders it.
//   F: features (T*FEAT_DIM); reg: network regression, denormalized (T*KM); logits (T*6); aux (T*3) or null
//   returns { track: Float32Array(T*K) rig-space channels, emo: Float32Array(T*6), plan }
// ---------------------------------------------------------------------------------------------------------
const VIS_GAIN = { AU26: 2.7, AU22: 2.2, AU18: 1.8, AU20: 2.0, AU16: 1.8, AU10: 0.6, AU24: 1.0, AU28: 1.4 };
// resting level of each mouth shape while talking (tracker units): only what rises above it shapes the mouth
const VIS_FLOOR = { AU10: 0.12, AU18: 0.04, AU24: 0.03, AU28: 0.02 };
export function visemesFromNet(reg, o, meta, out) {
  const n = meta.neutral, mu = meta.gt_mean || meta.y_mean, g = meta.gain;
  for (const id of VISEME) { const k = IDX[id]; let v = reg[o + k]; if (g) v = mu[k] + (v - mu[k]) * g[k]; out[id] = Math.max(0, (v - n[k] - (VIS_FLOOR[id] || 0)) * VIS_GAIN[id]); }
  return out;
}

// Mouth shapes measured by the camera (live mode) -> viseme targets for the performer. The camera sees the real
// mouth, so the gains are lower than for the network's cautious prediction.
const CAM_GAIN = { AU26: 1.7, AU22: 1.5, AU18: 1.4, AU20: 1.4, AU16: 1.3, AU10: 0.8, AU24: 1.0, AU28: 1.2 };
export function visemesFromCamera(au, meta, out = {}) {
  const n = meta.neutral;
  for (const id of VISEME) { const k = IDX[id]; out[id] = Math.max(0, ((au[id] || 0) - n[k] * 0.6 - (VIS_FLOOR[id] || 0)) * CAM_GAIN[id]); }
  return out;
}

export function directOffline({ F, T, reg, logits, aux, meta, seed = 7, profile = null, overrides = [], exaggeration = 1 }) {
  const KMn = meta.names.length, NE = EMOS.length, prof = { ...DEFAULT_PROFILE, ...(profile || {}) };
  const r = rng(seed);
  const cuts = []; for (const ov of overrides) for (const x of [ov.start, ov.end]) { const c = Math.round(x * FPS); if (overrides.some((o) => o !== ov && (Math.abs(o.start - x) < 0.02 || Math.abs(o.end - x) < 0.02) && o.emo !== ov.emo) && !cuts.includes(c)) cuts.push(c); } // only where one mark hands over to another with a different feeling
  const an = analyzeSpeech(F, T, aux, 3, cuts.sort((a, b) => a - b));
  const phrases = an.phrases;
  // ---- what: emotion + intensity per phrase (posterior averaged over the phrase, shared a little with its neighbours) ----
  const post = phrases.map((ph) => { const p = new Array(NE).fill(0); let n = 0; for (let t = ph.t0; t < ph.t1; t++) { const q = softmax(Array.from(logits.subarray(t * NE, t * NE + NE))); for (let e = 0; e < NE; e++) p[e] += q[e]; n++; } return p.map((v) => v / Math.max(1, n)); });
  const glob = new Array(NE).fill(0); { let n = 0; phrases.forEach((ph, i) => { const w = ph.t1 - ph.t0; for (let e = 0; e < NE; e++) glob[e] += post[i][e] * w; n += w; }); for (let e = 0; e < NE; e++) glob[e] /= Math.max(1, n); }
  let prevSide = r() < 0.5 ? -1 : 1, prevVar = -1, prevEmo = -1;
  phrases.forEach((ph, i) => {
    // neighbouring phrases share their reading, the less the longer the pause between them (a long pause is a new thought)
    const share = (gap) => 0.19 * clamp(1.4 - gap / 30, 0.25, 1);
    const wp = phrases[i - 1] ? share(ph.t0 - phrases[i - 1].t1) : 0.1, wn = phrases[i + 1] ? share(phrases[i + 1].t0 - ph.t1) : 0.1;
    const p = post[i].map((v, e) => (1 - wp - wn) * v + wp * (post[i - 1] ? post[i - 1][e] : glob[e]) + wn * (post[i + 1] ? post[i + 1][e] : glob[e]));
    const order = p.map((v, e) => e).sort((a, b) => p[b] - p[a]), best = order[0], second = order[1];
    const conf = p[best], margin = p[best] - p[second];
    let auxI = 0.5; if (aux) { let s = 0, n = 0; for (let t = ph.t0; t < ph.t1; t++) { s += aux[t * 3 + 2]; n++; } auxI = clamp(s / Math.max(1, n), 0, 1); }
    const arousal = clamp(0.5 + 0.35 * ph.meanE + 0.1 * Math.abs(ph.meanF0), 0, 1);
    const inten0 = clamp(0.6 + 0.4 * auxI + 0.3 * (arousal - 0.5), 0.45, 1.1); // how strongly the line is delivered
    ph.p = p; ph.emo = EMOS[best]; ph.conf = conf; ph.inten = inten0 * clamp(0.72 + 0.5 * conf, 0.72, 1); // an unsure reading is played smaller
    ph.auto = { emo: EMOS[best], conf };
    // an ambiguous reading is played as an attitude that fits both candidates (e.g. joy/fear -> wide-eyed excitement)
    ph.emo2 = p[second] >= 0.6 * p[best] && p[second] >= 0.2 ? EMOS[second] : null;
    ph.w2 = ph.emo2 ? Math.pow(p[second], 1.5) / (Math.pow(p[best], 1.5) + Math.pow(p[second], 1.5)) : 0;
    const mid = (ph.t0 + ph.t1) / 2 / FPS;
    // a user override fixes the emotion of the phrase; without an explicit intensity the delivery of the line decides it
    for (const ov of overrides) if (mid >= ov.start && mid <= ov.end) { ph.emo = ov.emo; ph.inten = clamp(ov.intensity ?? inten0, 0.2, 1.3); ph.conf = 1; ph.p = EMOS.map((e) => (e === ov.emo ? 1 : 0)); ph.emo2 = ov.emo2 || null; ph.w2 = ov.emo2 ? 0.3 : 0; ph.fixed = true; }
    ph.inten *= exaggeration;
    const vars = POSES[ph.emo]; let v = Math.floor(r() * vars.length);
    if (E[ph.emo] === prevEmo && v === prevVar) v = (v + 1) % vars.length;
    // a strong, sure reading gets the most open pose of the emotion first
    if (E[ph.emo] !== prevEmo && ph.inten > 0.8 && r() < 0.6) v = 0;
    ph.variant = v; ph.variant2 = ph.emo2 ? Math.floor(r() * POSES[ph.emo2].length) : 0;
    ph.side = -prevSide * (r() < 0.6 ? 1 : -1); prevSide = ph.side; prevVar = v; prevEmo = E[ph.emo];
    ph.accents = pickAccents(ph, r);
    ph.question = ph.rise > 0.9;
  });
  // ---- how: events ----
  const ev = []; // {t, type, ...}
  const push = (t, e) => ev.push({ t: Math.max(0, Math.round(t)), ...e });
  let prevEnd = -30, lastBlink = -30, lastEmo = null;
  const blinkAt = (t, dur, amt = 1) => { if (t - lastBlink >= 10) { push(t, { type: 'blink', dur, amt }); lastBlink = t; return true; } return false; };
  phrases.forEach((ph, i) => {
    const act = ACTING[ph.emo], I = ph.inten, next = phrases[i + 1];
    let pose = instPose(POSES[ph.emo][ph.variant], ph.side, I, prof);
    if (ph.emo2) pose = blend(pose, instPose(POSES[ph.emo2][ph.variant2], ph.side, I, prof), ph.w2);
    const full = blend(IDLE, pose, clamp(I * 1.2));
    for (const id in pose) full[id] = pose[id] + (IDLE[id] || 0) * (1 - clamp(I * 1.2));
    ph.pose = full;
    // the pose is in place before the first word (the springs need a few frames to arrive); anticipation precedes
    // it if there is room: a short move away from where the head is about to go, and the face gathers before it opens
    const lead = act.lead, tp = Math.max(prevEnd + 3, ph.t0 - lead), room = tp - (prevEnd + 1);
    ph.tp = tp;
    push(tp, { type: 'act', emo: ph.emo, inten: I });
    const ant = ph.emo === 'SAD' ? 0 : room >= 10 ? 6 : room >= 6 ? 4 : 0;
    if (ant) {
      const opens = (full.AU5 || 0) + (full.eyeScale || 0) > 0.3 || (full.AU1 || 0) + (full.AU2 || 0) > 0.6, closes = (full.AU4 || 0) > 0.5;
      const face = opens ? { AU7: 0.35 * I, AU6: 0.2 * I, AU4: 0.15 * I } : closes ? { AU5: 0.2 * I, AU1: 0.12 * I, AU2: 0.2 * I } : { AU7: 0.2 * I, AU6: 0.12 * I };
      face.squash = (full.squash || 0) >= 0 && (full.headPitch || 0) <= 2 ? -0.45 * I : 0.3 * I;
      push(tp - ant, { type: 'antic', to: full, amount: ant >= 6 ? 0.34 : 0.24, face });
    }
    // a blink covers a change of feeling; a line that continues the same feeling starts with open eyes more often
    if (r() < (ph.emo === 'FEA' ? 0.3 : ph.emo !== lastEmo ? 0.85 : 0.35)) blinkAt(tp - 2, act.blinkDur);
    lastEmo = ph.emo;
    push(tp - 3, { type: 'gaze', x: 0, y: 0 });
    push(tp, { type: 'pose', pose: full });
    // accents (the hit lands 2 frames before the stressed syllable)
    let side = ph.side, ai = Math.floor(r() * act.accents.length);
    const L = ph.t1 - ph.t0;
    for (const a of ph.accents) {
      const kind = a.key && ph.question ? 'up' : act.accents[ai % act.accents.length]; ai++;
      const A = Math.min(ACCENT_MAX, ACCENT_PUSH * act.amp * (0.5 + 0.7 * a.strength) * (0.55 + 0.6 * I) * prof.energy * (a.key ? 1.3 : 1)) * exaggeration;
      const hit = a.t - 2;
      // while the key pose is still arriving, the arrival is the accent: a second move on top of it only blurs both
      if (hit < tp + 9) { a.kind = 'pose'; a.amp = 0; continue; }
      push(Math.max(0, hit - 12), { type: 'gest', hit, curves: gesture(kind, A, side, a.key, prof) });
      a.kind = kind; a.amp = A;
      if (a.key && r() < 0.25 && hit - 7 > tp + 4) blinkAt(hit - 7, act.blinkDur * 0.9);
      side = -side;
    }
    // a long phrase gets a second attitude (weight shift) after its key accent or at its middle
    if (L > 58) {
      const key = ph.accents.find((a) => a.key); let ts = key && key.t - ph.t0 > 18 && ph.t1 - key.t > 24 ? key.t + 10 : ph.t0 + Math.round(L * 0.55);
      const alt = instPose(POSES[ph.emo][(ph.variant + 1 + Math.floor(r() * 2)) % POSES[ph.emo].length], -ph.side, I * 0.92, prof);
      const shift = {}; for (const id in alt) { const g = CHANNELS[IDX[id]].group; if (g === 'head' || g === 'torso' || id === 'browAsym' || id === 'mouthAsym') shift[id] = alt[id]; }
      for (const id in full) { const g = CHANNELS[IDX[id]].group; if ((g === 'head' || g === 'torso' || id === 'browAsym' || id === 'mouthAsym') && !(id in shift)) shift[id] = 0; }
      push(ts, { type: 'pose', pose: shift, partial: true });
      if (r() < 0.4) blinkAt(ts - 1, act.blinkDur);
    }
    // rising ending: brows go up and stay there while the head tilts in (asking)
    if (ph.question) { const q = { AU1: (full.AU1 || 0) + 0.3 * I, AU2: (full.AU2 || 0) + 0.45 * I, headRoll: (full.headRoll || 0) + 5 * ph.side * prof.tilt, headPitch: (full.headPitch || 0) - 3, AU5: (full.AU5 || 0) + 0.15 }; push(ph.t1 - 9, { type: 'pose', pose: q, partial: true }); }
    // end of the phrase: settle into a relaxed version of the pose, blink, and let the eyes wander in a long pause
    const pause = (next ? next.t0 : T + 60) - ph.t1;
    const te = ph.t1 + 5;
    if (next) {
      push(te, { type: 'pose', pose: scalePose(full, act.relax, null) });
      if (r() < 0.4) blinkAt(ph.t1 + 3 + Math.floor(r() * 3), act.blinkDur);
      if (pause > 44) { // long enough to look away and be back before the next line is prepared
        const look = act.look, s = r() < 0.5 ? -1 : 1, tl = ph.t1 + 9 + Math.floor(r() * 5);
        if (look === 'think') push(tl, { type: 'gaze', x: 0.45 * s, y: 0.4 });
        else if (look === 'glance') push(tl, { type: 'gaze', x: 0.4 * s, y: 0.15 });
        else if (look === 'down') push(tl, { type: 'gaze', x: 0.15 * s, y: -0.5 });
        else if (look === 'aside') push(tl, { type: 'gaze', x: 0.55 * s, y: -0.2 });
        else if (look === 'dart') { push(tl, { type: 'gaze', x: 0.6 * s, y: 0.05 }); push(tl + 7, { type: 'gaze', x: -0.55 * s, y: 0 }); push(tl + 13, { type: 'gaze', x: 0.3 * s, y: -0.1 }); }
        if (look !== 'stare') { push(tl + 2, { type: 'pose', pose: { headYaw: (full.headYaw || 0) * act.relax + 5 * s, headPitch: (full.headPitch || 0) * act.relax + (look === 'down' ? 4 : look === 'think' ? -3 : 0) }, partial: true }); }
      }
    } else {
      // last phrase: hold the feeling for a beat, then come back to rest
      push(te, { type: 'pose', pose: scalePose(full, act.relax) });
      if (r() < 0.6) blinkAt(ph.t1 + 4, act.blinkDur);
      push(ph.t1 + 26, { type: 'act', emo: 'NEU', inten: 0.5 });
      push(ph.t1 + 26, { type: 'pose', pose: { ...IDLE } });
      push(ph.t1 + 24, { type: 'gaze', x: 0, y: 0 });
    }
    prevEnd = ph.t1;
  });
  // ordinary blinks in the gaps the punctuation left
  { let t = 8 + Math.floor(r() * 30); const blinks = ev.filter((e) => e.type === 'blink').map((e) => e.t).sort((a, b) => a - b);
    while (t < T) { const ph = phrases.find((p) => t >= p.tp && t < p.t1 + 20); const act = ACTING[ph ? ph.emo : 'NEU'];
      if (!blinks.some((b) => Math.abs(b - t) < act.blink * FPS * 0.45)) { push(t, { type: 'blink', dur: act.blinkDur, amt: 1 }); blinks.push(t); if (r() < 0.08) push(t + 8, { type: 'blink', dur: act.blinkDur * 0.8, amt: 1 }); }
      t += Math.round(act.blink * FPS * (0.6 + 0.8 * r())); } }
  ev.sort((a, b) => a.t - b.t);
  // ---- per-frame inputs: visemes (smoothed), pitch-riding brows, a trace of the learned head motion ----
  const vis = []; { const tmp = {}; const rawV = []; for (let t = 0; t < T; t++) rawV.push({ ...visemesFromNet(reg, t * KMn, meta, tmp) });
    for (let t = 0; t < T; t++) { const o = {}; for (const id of VISEME) { const a = rawV[Math.max(0, t - 1)][id], b = rawV[t][id], c = rawV[Math.min(T - 1, t + 1)][id]; o[id] = id === 'AU26' ? Math.max(b, 0.25 * a + 0.5 * b + 0.25 * c) : 0.25 * a + 0.5 * b + 0.25 * c; } vis.push(o); } }
  for (const ph of phrases) for (const a of ph.accents) for (let j = -3; j <= 3; j++) { const u = a.t + j; if (u >= 0 && u < T) vis[u].AU26 *= 1 + 0.35 * a.strength * (1 - Math.abs(j) / 4); }
  const browF0 = new Float32Array(T), tex = new Float32Array(T * 3);
  const hi = ['headPitch', 'headYaw', 'headRoll'].map((n) => meta.names.indexOf(n));
  for (const ph of phrases) {
    const m = [0, 0, 0]; let n = 0; for (let t = ph.t0; t < ph.t1; t++) { for (let j = 0; j < 3; j++) m[j] += reg[t * KMn + hi[j]]; n++; }
    for (let t = Math.max(0, ph.t0 - 6); t < Math.min(T, ph.t1 + 6); t++) {
      const w = smooth(Math.min(1, (t - (ph.t0 - 6)) / 6, (ph.t1 + 6 - t) / 6));
      if (t >= ph.t0 && t < ph.t1) browF0[t] = clamp(an.f0[t] - ph.meanF0, -0.8, 1.8) * 0.085 * ph.inten;
      const tc = clamp(t, ph.t0, ph.t1 - 1); for (let j = 0; j < 3; j++) tex[t * 3 + j] = w * 0.4 * (reg[tc * KMn + hi[j]] - m[j] / Math.max(1, n));
    }
  }
  // smooth the pitch-riding so brows glide, not flicker
  for (let pass = 0; pass < 2; pass++) for (let t = 1; t < T; t++) browF0[t] = 0.6 * browF0[t - 1] + 0.4 * browF0[t];
  // the mouth shapes sound from 4 frames before a phrase (lips part ahead of the voice) to 3 frames after it
  const gate0 = new Float32Array(T), gate = new Float32Array(T);
  for (let t = 0; t < T; t++) if (an.speech[t]) gate0.fill(1, Math.max(0, t - 4), Math.min(T, t + 4));
  for (let t = 0; t < T; t++) gate[t] = 0.25 * gate0[Math.max(0, t - 1)] + 0.5 * gate0[t] + 0.25 * gate0[Math.min(T - 1, t + 1)];
  // ---- perform ----
  const perf = new Performer({ seed, profile: prof });
  const track = new Float32Array(T * K), emo = new Float32Array(T * NE);
  let ei = 0;
  for (let t = 0; t < T; t++) {
    while (ei < ev.length && ev[ei].t <= t) perf.event(ev[ei++]);
    const y = perf.step({ speaking: an.speech[t], mouth: gate[t], vis: vis[t], browF0: browF0[t], tex: [tex[t * 3], tex[t * 3 + 1], tex[t * 3 + 2]] });
    track.set(y, t * K);
    const ph = phrases.find((p) => t >= p.tp && t < p.t1 + 12);
    if (ph) for (let e = 0; e < NE; e++) emo[t * NE + e] = ph.p[e]; else emo[t * NE] = 1;
  }
  const plan = { phrases: phrases.map((ph) => ({ t0: ph.t0, t1: ph.t1, tp: ph.tp, emo: ph.emo, emo2: ph.emo2, w2: +ph.w2.toFixed(2), conf: +ph.conf.toFixed(3), inten: +ph.inten.toFixed(3), fixed: !!ph.fixed, auto: { emo: ph.auto.emo, conf: +ph.auto.conf.toFixed(3) }, p: ph.p.map((v) => +v.toFixed(3)), pose: POSES[ph.emo][ph.variant].name, side: ph.side, question: ph.question, accents: ph.accents.filter((a) => a.kind).map((a) => ({ t: a.t, kind: a.kind, strength: +a.strength.toFixed(2), key: a.key })) })), events: ev.length };
  return { track, emo, plan, speech: an.speech };
}

// ---------------------------------------------------------------------------------------------------------
// Live director: the same acting, decided causally (only the network's small look-ahead is available).
//   step({feat, ahead, reg, p, aux, voiced, camVis}) -> Float32Array(K) rig-space frame
// ---------------------------------------------------------------------------------------------------------
// live: a pause of at least this many frames (0.5 s) starts a new line; a shorter one continues the line being spoken
const NEW_LINE_GAP = 15;

export class LiveDirector {
  constructor(meta, opts = {}) {
    this.meta = meta; this.seed = opts.seed ?? 7; this.r = rng(this.seed ^ 0x51ed);
    this.prof = { ...DEFAULT_PROFILE, ...(opts.profile || {}) };
    this.exaggeration = opts.exaggeration ?? 1;
    // how much more likely a new reading must be than the feeling being played before the pose follows it
    // (at the first re-reading after 8 speech frames, and at the later ones); set on the validation speakers
    this.switchMargins = opts.switchMargins || [0.03, 0.08];
    this.perf = new Performer({ seed: this.seed, profile: this.prof });
    this.NE = EMOS.length;
    this.t = 0; this.sil = 99; this.inPhrase = false; this.phraseT = 0;
    this.ctx = new Float32Array(this.NE).fill(1 / this.NE); // slow context of the conversation
    this.acc = new Float32Array(this.NE); this.accN = 0;     // evidence inside the current phrase
    this.recent = new Float32Array(this.NE).fill(1 / this.NE); // the network's latest answer, lightly smoothed
    this.emo = 'NEU'; this.inten = 0.6; this.side = 1; this.variant = 0; this.pose = { ...IDLE };
    this.eMean = 0; this.fMean = 0; this.eN = 0;
    this.scoreMean = 0.4; this.scoreVar = 0.3; this.lastAccent = -99; this.lastKey = -99; this.accentIdx = 0;
    this.nextBlink = 40; this.tmp = {}; this.browF0 = 0; this.auxI = 0.5; this.auxN = 0;
    this.lookedAway = false;
    this.autoBlink = true; // false while a camera supplies the blinks
  }
  setProfile(profile) { this.prof = { ...DEFAULT_PROFILE, ...(profile || {}) }; this.perf.prof = this.prof; }
  _blink(dur) { if (this.autoBlink) this.perf.blink(dur); }
  emoDisplay() { const p = Array.from(this.accN > 3 ? this.recent : this.ctx); const s = p.reduce((a, b) => a + b, 0) || 1; return p.map((v) => v / s); }
  // the feeling of the line as far as it has been heard: the causal network's latest answer (it already sums up what
  // it heard), with a little of the conversation's context
  _reread() {
    const p = new Array(this.NE); let s = 0;
    for (let e = 0; e < this.NE; e++) { p[e] = 0.85 * this.recent[e] + 0.15 * this.ctx[e]; s += p[e]; }
    for (let e = 0; e < this.NE; e++) p[e] /= s;
    let best = 0; for (let e = 1; e < this.NE; e++) if (p[e] > p[best]) best = e;
    const sorted = p.slice().sort((a, b) => b - a);
    return { emo: EMOS[best], p, conf: sorted[0], margin: sorted[0] - sorted[1] };
  }
  _decide() {
    const p = new Array(this.NE); let s = 0;
    for (let e = 0; e < this.NE; e++) { p[e] = (this.acc[e] + 4 * this.ctx[e]) / (this.accN + 4); s += p[e]; }
    for (let e = 0; e < this.NE; e++) p[e] /= s;
    let best = 0; for (let e = 1; e < this.NE; e++) if (p[e] > p[best]) best = e;
    const sorted = p.slice().sort((a, b) => b - a);
    return { emo: EMOS[best], p, conf: sorted[0], margin: sorted[0] - sorted[1] };
  }
  _pose(emo, inten, opts = {}) {
    const r = this.r, vars = POSES[emo];
    if (emo !== this.emo || opts.fresh) { let v = Math.floor(r() * vars.length); if (emo === this.emo && v === this.variant) v = (v + 1) % vars.length; this.variant = v; this.side = -this.side * (r() < 0.8 ? 1 : -1); }
    this.emo = emo; this.inten = inten;
    const pose = instPose(vars[this.variant], this.side, inten, this.prof), full = {};
    for (const id in IDLE) full[id] = IDLE[id] * (1 - clamp(inten * 1.2));
    for (const id in pose) full[id] = pose[id] + (full[id] || 0);
    this.pose = full;
    this.perf.setActing(emo, inten); this.perf.setPose(full, this.perf.t);
  }
  step({ feat, ahead = [], reg, p, aux = null, voiced = 0, camVis = null }) {
    const perf = this.perf, r = this.r, t = this.t;
    for (let e = 0; e < this.NE; e++) this.ctx[e] = 0.995 * this.ctx[e] + 0.005 * p[e];
    const sp = (f) => (f[1] > 0.5 || f[0] > -1.5 ? 1 : 0);
    const coming = ahead.some(sp);
    // ---- phrase boundaries ----
    if (!this.inPhrase && (voiced || coming)) {
      if (this.sil >= NEW_LINE_GAP) {
        // a new thought: blink, eyes back to the listener, take a first reading of the feeling and commit to a pose
        this.inPhrase = true; this.phraseT = 0; this.acc.fill(0); this.accN = 0; this.eMean = 0; this.fMean = 0; this.eN = 0; this.auxI = 0.5; this.auxN = 0;
        const d = this._decide();
        perf.lookAt(0, 0); if (d.emo !== 'FEA' || r() < 0.4) this._blink(ACTING[d.emo].blinkDur);
        this._pose(d.emo, clamp(0.45 + 0.5 * d.conf, 0.4, 0.8) * this.exaggeration, { fresh: true });
        this.lookedAway = false;
      } else {
        // The same line goes on after a short pause (a breath, a stop consonant, a hesitation). What has been heard of
        // the line stays, and so does the feeling being played: the performer takes the pose up again. (The first
        // versions started from scratch after every pause of 0.27 s, guessed the feeling again from the context, and
        // after a pause of exactly 6 or 7 frames did not listen to the rest of the line at all. Measured on the
        // validation speakers, 44% of the lines ended in one of these two states.)
        this.inPhrase = true; this.phraseT = 0;
        perf.setPose(this.pose, perf.t);
      }
    }
    if (voiced) this.sil = 0; else this.sil++;
    if (this.inPhrase) {
      this.phraseT++;
      if (voiced) { for (let e = 0; e < this.NE; e++) { this.acc[e] += p[e]; this.recent[e] += (this.accN ? 0.15 : 1) * (p[e] - this.recent[e]); } this.accN++; this.eMean += (feat[0] - this.eMean) / (++this.eN); if (feat[1] > 0.5) this.fMean += (feat[2] - this.fMean) / Math.max(1, this.eN); if (aux) { this.auxI += (clamp(aux[2], 0, 1) - this.auxI) / (++this.auxN); } }
      // Nothing is known about a line before it is spoken, and the network knows more with every syllable (on unseen
      // speakers about 45% right after 0.3 s, 57% after 1 s, 63% by the end of the line). So the first pose is a guess
      // from the context, and the reading is taken again at short intervals; the pose follows once it clearly differs.
      if (voiced && (this.accN === 8 || this.accN === 16 || (this.accN > 16 && (this.accN - 16) % 12 === 0))) {
        const d = this._reread(), cur = d.p[E[this.emo]];
        const inten = clamp((0.5 + 0.4 * this.auxI + 0.25 * (clamp(0.5 + 0.35 * this.eMean, 0, 1) - 0.5)) * clamp(0.55 + 0.9 * d.margin + 0.35 * d.conf, 0.55, 1), 0.35, 1.05) * this.exaggeration;
        if (d.emo !== this.emo && d.conf - cur > this.switchMargins[this.accN === 8 ? 0 : 1]) { this._blink(ACTING[d.emo].blinkDur); this._pose(d.emo, inten); }
        else if (this.accN === 8) this._pose(this.emo, inten);
      }
      if (this.sil === 6) { // the phrase is over: settle, blink
        this.inPhrase = false;
        const act = ACTING[this.emo];
        perf.setPose(scalePose(this.pose, act.relax), perf.t);
        if (r() < 0.7) this._blink(act.blinkDur);
      }
    } else {
      const act = ACTING[this.emo];
      if (this.sil === 30 && act.look !== 'stare' && !this.lookedAway) {
        const s = r() < 0.5 ? -1 : 1; this.lookedAway = true;
        const g = { think: [0.45 * s, 0.4], glance: [0.4 * s, 0.15], down: [0.15 * s, -0.5], aside: [0.55 * s, -0.2], dart: [0.6 * s, 0.05] }[act.look] || [0, 0];
        perf.lookAt(g[0], g[1]);
        perf.setPose({ headYaw: (this.pose.headYaw || 0) * act.relax + 5 * s }, perf.t + 2, { partial: true });
      }
      if (this.sil === 80) { perf.setActing('NEU', 0.5); perf.setPose({ ...IDLE }, perf.t); perf.lookAt(0, 0); this.emo = 'NEU'; this.pose = { ...IDLE }; }
    }
    // ---- accents: a syllable peak seen 2 frames ahead ----
    if (this.inPhrase && ahead.length >= 3) {
      const e0 = ahead[0][0], e1 = ahead[1][0], e2 = ahead[2][0];
      if (e1 >= e0 && e1 > e2 && e1 > -1.2 && sp(ahead[1])) {
        const score = 1.6 * (e1 - this.eMean) + 0.55 * (ahead[1][2] - this.fMean) + (aux ? 0.45 * aux[0] + 0.3 * aux[1] : 0);
        const sd = Math.sqrt(this.scoreVar) + 0.25, z = (score - this.scoreMean) / sd;
        this.scoreMean += 0.08 * (score - this.scoreMean); this.scoreVar += 0.08 * ((score - this.scoreMean) ** 2 - this.scoreVar);
        // the learned emphasis decides (as offline); a soft accent only when nothing has been stressed for a while
        const emph = aux ? Math.max(aux[0], 0.9 * aux[1]) : null;
        const go = emph === null ? z > -0.1 : emph >= ACCENT_STRONG || (emph >= ACCENT_SOFT && t - this.lastAccent > 40);
        if (go && t - this.lastAccent >= 12 && this.phraseT > 3) {
          const act = ACTING[this.emo], strength = emph === null ? clamp(0.55 + 0.25 * z, 0.25, 1) : accentStrength(emph);
          const key = strength > 0.75 && t - this.lastKey > 45;
          const A = Math.min(ACCENT_MAX, ACCENT_PUSH * act.amp * (0.5 + 0.7 * strength) * (0.55 + 0.6 * this.inten) * this.prof.energy * (key ? 1.3 : 1)) * this.exaggeration;
          const kind = act.accents[this.accentIdx++ % act.accents.length];
          perf.addGesture(perf.t + 3, gesture(kind, A, (this.accentIdx % 2 ? 1 : -1) * this.side, key, this.prof, true));
          this.lastAccent = t; if (key) this.lastKey = t;
          this.boost = 1 + 0.35 * strength; this.boostT = 6;
        }
      }
    }
    // ---- ordinary blinks ----
    if (--this.nextBlink <= 0) { const act = ACTING[this.emo]; this._blink(act.blinkDur); this.nextBlink = Math.round(act.blink * FPS * (0.6 + 0.8 * r())); }
    // ---- frame inputs ----
    const vis = camVis || visemesFromNet(reg, 0, this.meta, this.tmp);
    if (this.boostT > 0) { vis.AU26 = (vis.AU26 || 0) * lerp(1, this.boost, this.boostT / 6); this.boostT--; }
    const target = this.inPhrase && feat[1] > 0.5 ? clamp(feat[2] - this.fMean, -0.8, 1.8) * 0.085 * this.inten : 0;
    this.browF0 += (target - this.browF0) * 0.3;
    this.t++;
    // a camera sees the mouth itself, so its shape is shown whether or not the microphone hears a voice;
    // the mouth made from the voice alone is only shown while there is speech
    const inp = { speaking: voiced, vis, browF0: this.browF0 }; if (camVis) inp.mouth = 1;
    return perf.step(inp);
  }
}
