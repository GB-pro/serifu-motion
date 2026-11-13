// Serifu Motion — speech-driven emo generation (offline and streaming).
// Layers: learned TCN (speech -> AU/head/torso means) + learned dispersion gain + data-fitted stochastic layer
// (AR(1) residual motion, blink events, gaze saccades). Camera input is NOT used here.
import { EmoNet, StreamingEmoNet } from './model.js';
import { IDS, K, IDX, EMOS, MOUTH_CHANNELS } from './au.js';
import { FEAT_DIM, StreamingFeatures } from './features.js';

export function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const NOISE_CH = new Set(['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU12', 'gazeX', 'gazeY', 'headPitch', 'headYaw', 'headRoll', 'headZ', 'torsoRoll', 'torsoShift', 'torsoShrug', 'torsoLean']);
const NOISE_SCALE = { AU1: 0.35, AU2: 0.35, AU4: 0.3, AU5: 0.3, AU6: 0.25, AU7: 0.25, AU12: 0.25, gazeX: 0.3, gazeY: 0.3, headPitch: 0.7, headYaw: 0.7, headRoll: 0.7, headZ: 0.5, torsoRoll: 0.6, torsoShift: 0.6, torsoShrug: 0.5, torsoLean: 0.5 };

export class PostProcessor {
  constructor(meta, opts = {}) {
    this.m = meta; this.o = { noise: 1, blinks: true, saccades: true, gain: true, ...opts };
    this.r = rng(opts.seed ?? 12345);
    this.ar = new Float32Array(K);
    this.blinkT = -1; this.sinceBlink = 99; this.sac = [0, 0]; this.sacTarget = [0, 0]; this.sinceSac = 0; this.nextSac = 1;
    this.prevVoiced = 0; this.pauseLen = 0;
    this.mu = meta.gt_mean || meta.y_mean;
  }
  // y: Float32Array(K) model output (denormalized, in place); voiced: speech activity for this frame
  frame(y, voiced) {
    const m = this.m, o = this.o, r = this.r;
    if (o.gain && m.gain) for (let k = 0; k < K; k++) { if (IDS[k] === 'AU45') continue; y[k] = this.mu[k] + (y[k] - this.mu[k]) * m.gain[k]; }
    if (o.noise > 0 && m.resid_std) for (let k = 0; k < K; k++) {
      const id = IDS[k]; if (!NOISE_CH.has(id)) continue;
      const a = Math.min(0.97, Math.max(0.6, m.resid_ar1[k] || 0.9));
      this.ar[k] = a * this.ar[k] + Math.sqrt(1 - a * a) * gauss(r);
      y[k] += this.ar[k] * m.resid_std[k] * NOISE_SCALE[id] * o.noise;
    }
    // speech pause tracking
    if (voiced) this.pauseLen = 0; else this.pauseLen++;
    const pauseOnset = this.prevVoiced && !voiced; this.prevVoiced = voiced;
    // blinks: event process (rate from data, more likely right after phrase ends and when the net predicts eyelid closure)
    const bi = IDX.AU45;
    if (o.blinks) {
      const pred = y[bi];
      let hz = (m.blink_per_min || 12) / 60 / 30;
      hz *= 1 + 4 * Math.max(0, pred - 0.25);
      if (this.pauseLen > 0 && this.pauseLen < 10) hz *= 2.5;
      if (pauseOnset) hz += 0.12;
      this.sinceBlink++;
      if (this.blinkT < 0 && this.sinceBlink > 12 && r() < hz) { this.blinkT = 0; this.sinceBlink = 0; }
      let b = 0;
      if (this.blinkT >= 0) { const t = this.blinkT; b = t < 2 ? (t + 1) / 2.2 : t < 3 ? 1 : Math.max(0, 1 - (t - 2) / 4); this.blinkT++; if (this.blinkT > 7) this.blinkT = -1; }
      y[bi] = b;
    }
    // gaze saccades
    if (o.saccades) {
      this.sinceSac++;
      if (this.sinceSac >= this.nextSac) {
        const amp = (m.saccade_amp || 0.1) * 2.2;
        const back = r() < 0.55; // often return toward the listener
        this.sacTarget = back ? [gauss(r) * amp * 0.3, gauss(r) * amp * 0.25] : [gauss(r) * amp, gauss(r) * amp * 0.6];
        this.sinceSac = 0;
        const rate = Math.max(0.25, m.saccade_per_sec || 0.5);
        this.nextSac = Math.max(8, Math.round(-Math.log(1 - r()) / rate * 30));
      }
      this.sac[0] += (this.sacTarget[0] - this.sac[0]) * 0.55; this.sac[1] += (this.sacTarget[1] - this.sac[1]) * 0.55;
      y[IDX.gazeX] += this.sac[0]; y[IDX.gazeY] += this.sac[1];
    }
    for (let k = 0; k < 20; k++) y[k] = Math.max(0, Math.min(1, y[k]));
    return y;
  }
}

export function softmax(a) { const mx = Math.max(...a); const e = a.map((v) => Math.exp(v - mx)); const s = e.reduce((p, q) => p + q, 0); return e.map((v) => v / s); }

export class EmoGenerator {
  constructor(offlineJson, realtimeJson) {
    this.net = new EmoNet(offlineJson);
    this.rt = realtimeJson ? new EmoNet(realtimeJson) : null;
    this.meta = this.net.meta;
  }
  // F: normalized features (T*FEAT_DIM). returns {T, base(T*K), emo(T*6), voiced(T)}
  generate(F, T, opts = {}) {
    const net = this.net, meta = net.meta, D = net.Dout;
    // pad with silence context so clip edges behave like the training sequences
    const P = 30, TT = T + 2 * P, X = new Float32Array(TT * FEAT_DIM);
    for (let t = 0; t < TT; t++) { const s = t - P; if (s >= 0 && s < T) X.set(F.subarray(s * FEAT_DIM, (s + 1) * FEAT_DIM), t * FEAT_DIM); else X[t * FEAT_DIM] = -4; }
    const Y = net.run(X, TT);
    const base = new Float32Array(T * K), emo = new Float32Array(T * EMOS.length), voiced = new Uint8Array(T);
    const pp = new PostProcessor(meta, opts);
    const y = new Float32Array(K);
    for (let t = 0; t < T; t++) {
      const o = (t + P) * D;
      for (let k = 0; k < K; k++) y[k] = Y[o + k] * meta.y_std[k] + meta.y_mean[k];
      voiced[t] = F[t * FEAT_DIM + 1] > 0.5 || F[t * FEAT_DIM] > -1.5 ? 1 : 0;
      pp.frame(y, voiced[t]);
      base.set(y, t * K);
      const p = softmax(Array.from(Y.subarray(o + K, o + K + EMOS.length)));
      emo.set(p, t * EMOS.length);
    }
    // smooth emotion posteriors over ~1.5 s for a stable per-phrase reading
    const E = EMOS.length, W = 22, sm = new Float32Array(emo.length);
    for (let t = 0; t < T; t++) for (let e = 0; e < E; e++) { let s = 0, c = 0; for (let j = Math.max(0, t - W); j <= Math.min(T - 1, t + W); j++) { s += emo[j * E + e]; c++; } sm[t * E + e] = s / c; }
    return { T, base, emo: sm, voiced };
  }
}

// Realtime: microphone samples in, emo frames out (delayed by the causal model's lookahead, ~133 ms)
export class LiveEmo {
  constructor(realtimeJson, opts = {}) {
    this.net = new EmoNet(realtimeJson);
    this.sn = new StreamingEmoNet(this.net);
    this.feat = new StreamingFeatures();
    this.pp = new PostProcessor(this.net.meta, opts);
    this.meta = this.net.meta;
    this.emo = new Float32Array(EMOS.length).fill(1 / EMOS.length);
    this.lastVoiced = [];
  }
  push(samples16k) {
    const frames = this.feat.push(samples16k);
    const out = [];
    const meta = this.meta;
    for (const f of frames) {
      const o = this.sn.step(f);
      const y = new Float32Array(K);
      for (let k = 0; k < K; k++) y[k] = o[k] * meta.y_std[k] + meta.y_mean[k];
      this.lastVoiced.push(f[1] > 0.5 || f[0] > -1.5 ? 1 : 0);
      const v = this.lastVoiced.length > meta.lookahead ? this.lastVoiced.shift() : 0;
      this.pp.frame(y, v);
      const p = softmax(Array.from(o.subarray(K, K + EMOS.length)));
      for (let e = 0; e < EMOS.length; e++) this.emo[e] = this.emo[e] * 0.97 + p[e] * 0.03;
      out.push({ y, emo: Array.from(this.emo), voiced: v });
    }
    return out;
  }
}
export { MOUTH_CHANNELS };
