// Serifu Motion — speech-driven emo generation (offline and streaming).
// 1) learned TCN: speech -> emotion, accent timing, intensity, mouth shapes and natural motion means (trained on actors)
// 2) animation interpretation layer (animate.js): plays that reading the way a cartoon character would  [style "anime"]
//    or a direct, linear rendition of the learned motion with a fitted stochastic layer               [style "natural"]
// Camera input is NOT used here. All tracks leave this module in rig space (K channels, see au.js).
import { EmoNet, StreamingEmoNet } from './model.js';
import { IDS, K, KM, IDX, EMOS, CHANNELS, MOUTH_CHANNELS } from './au.js';
import { FEAT_DIM, StreamingFeatures, globalFeatures } from './features.js';
import { rng, gauss, softmax, clamp } from './util.js';
import { directOffline, LiveDirector, analyzeSpeech } from './animate.js';
export { rng, softmax };

const NOISE_CH = new Set(['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU12', 'gazeX', 'gazeY', 'headPitch', 'headYaw', 'headRoll', 'headZ', 'torsoRoll', 'torsoShift', 'torsoShrug', 'torsoLean']);
const NOISE_SCALE = { AU1: 0.35, AU2: 0.35, AU4: 0.3, AU5: 0.3, AU6: 0.25, AU7: 0.25, AU12: 0.25, gazeX: 0.3, gazeY: 0.3, headPitch: 0.7, headYaw: 0.7, headRoll: 0.7, headZ: 0.5, torsoRoll: 0.6, torsoShift: 0.6, torsoShrug: 0.5, torsoLean: 0.5 };

// "natural" style: tracker-space frame (KM values) -> rig space (K values) with one linear gain per channel group
export const LINEAR_STYLE = { brow: 1.7, eye: 1.4, nose: 1.4, mouth: 1.25, gaze: 1.2, head: 1.6, torso: 1.4 };
export function linearToRig(y, meta, out = new Float32Array(K), style = LINEAR_STYLE) {
  const n = meta.neutral; out.fill(0);
  for (let k = 0; k < KM; k++) {
    const c = CHANNELS[k]; let v = y[k]; const g = style[c.group] ?? 1;
    if (c.group === 'head' || c.group === 'torso' || c.group === 'gaze') v = v * g;
    else if (IDS[k] === 'AU45') v = clamp((v - 0.12) / 0.55); // the tracker reports ~0.12 with open eyes
    else v = Math.max(0, (v - n[k]) * g);                      // the tracker's resting value is the avatar's rest pose
    out[k] = v;
  }
  return out;
}

export class PostProcessor {
  constructor(meta, opts = {}) {
    this.m = meta; this.o = { noise: 1, blinks: true, saccades: true, gain: true, ...opts };
    this.r = rng(opts.seed ?? 12345);
    this.ar = new Float32Array(KM); this.ar2 = new Float32Array(KM);
    this.smooth = opts.noiseSmooth ?? 0.9;
    // empirical std of the smoothed (2-pole) noise so the amplitude matches the data's residual std
    { let a1 = 0, a2 = 0, s = 0; const rr = rng(99), a = this.smooth; for (let i = 0; i < 6000; i++) { a1 = a * a1 + Math.sqrt(1 - a * a) * gauss(rr); a2 = a * a2 + (1 - a) * a1; if (i > 500) s += a2 * a2; } this.noiseNorm = 1 / Math.sqrt(s / 5500); }
    this.blinkT = -1; this.sinceBlink = 99; this.sac = [0, 0]; this.sacTarget = [0, 0]; this.sinceSac = 0; this.nextSac = 1;
    this.prevVoiced = 0; this.pauseLen = 0;
    this.mu = meta.gt_mean || meta.y_mean;
  }
  // y: Float32Array(K) model output (denormalized, in place); voiced: speech activity for this frame
  frame(y, voiced) {
    const m = this.m, o = this.o, r = this.r;
    if (o.gain && m.gain) for (let k = 0; k < KM; k++) { if (IDS[k] === 'AU45') continue; y[k] = this.mu[k] + (y[k] - this.mu[k]) * m.gain[k]; }
    if (o.noise > 0 && m.resid_std) for (let k = 0; k < KM; k++) {
      const id = IDS[k]; if (!NOISE_CH.has(id)) continue;
      const a = this.smooth;
      this.ar[k] = a * this.ar[k] + Math.sqrt(1 - a * a) * gauss(r);
      this.ar2[k] = a * this.ar2[k] + (1 - a) * this.ar[k];
      y[k] += this.ar2[k] * this.noiseNorm * m.resid_std[k] * NOISE_SCALE[id] * o.noise;
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


// the say of the 'relative' readers in the committee once a recording holds four lines or more (see runNet)
export const RELATIVE_SHARE = 0.4;

export class EmoGenerator {
  // extraJsons: further offline networks (same architecture, different random seeds). Their emotion and accent
  // readings are averaged with the main network's: a committee misreads less often than any single member.
  constructor(offlineJson, realtimeJson, extraJsons = []) {
    this.net = new EmoNet(offlineJson);
    this.rt = realtimeJson ? new EmoNet(realtimeJson) : null;
    this.extra = (extraJsons || []).map((j) => new EmoNet(j));
    this.meta = this.net.meta;
  }
  // Network input: the frames with silence around them (clip edges then behave like the training sequences), and
  // after each frame the recording-level inputs this model was trained with (meta.globals), constant over the recording.
  _input(net, F, T, P, g) {
    const D = net.Din, G = net.meta.globals || [], TT = T + 2 * P, X = new Float32Array(TT * D);
    if (G.length && !g) throw new Error('this model needs the statistics of the recording (speakerStats)');
    for (let t = 0; t < TT; t++) {
      const s = t - P, o = t * D;
      if (s >= 0 && s < T) X.set(F.subarray(s * FEAT_DIM, (s + 1) * FEAT_DIM), o); else X[o] = -4;
      for (let j = 0; j < G.length; j++) X[o + FEAT_DIM + j] = g[G[j]];
    }
    return X;
  }
  // raw reading of the speech: regression (denormalized, T*KM), emotion logits (T*6), aux accent/intensity signals (T*3)
  // stats: the statistics the features were normalised with (speakerStats of the recording)
  runNet(F, T, stats = null) {
    const net = this.net, meta = net.meta, D = net.Dout, NE = EMOS.length, NA = (meta.aux || []).length;
    const g = stats ? globalFeatures(stats) : null, same = (a, b) => a.Din === b.Din && String(a.meta.globals || []) === String(b.meta.globals || []);
    const P = 30, TT = T + 2 * P, X = this._input(net, F, T, P, g);
    const Y = net.run(X, TT);
    const reg = new Float32Array(T * KM), logits = new Float32Array(T * NE), aux = NA ? new Float32Array(T * NA) : null;
    for (let t = 0; t < T; t++) {
      const o = (t + P) * D;
      for (let k = 0; k < KM; k++) reg[t * KM + k] = Y[o + k] * meta.y_std[k] + meta.y_mean[k];
      for (let e = 0; e < NE; e++) logits[t * NE + e] = Y[o + KM + e];
      for (let a = 0; a < NA; a++) aux[t * NA + a] = Y[o + KM + NE + a];
    }
    if (this.extra.length) {
      // Committee: the emotion probabilities and the accent / intensity signals of several networks are combined
      // (mouth and motion stay with the main network). Two kinds of reader sit on it (meta.reads):
      //   'line'     trained on recordings of every kind: reads a line mostly from what is inside the line itself
      //   'relative' trained on long recordings only: reads a line against the rest of the recording
      // A relative reader is the better one when the recording holds several lines to compare, and has nothing to
      // go on when it holds a single line. So its say grows with the number of lines (weights set on the validation
      // speakers): none for one line, RELATIVE_SHARE from four lines on.
      const nets = [net, ...this.extra], kind = nets.map((n) => n.meta.reads || 'relative');
      const nRel = kind.filter((k) => k === 'relative').length, nLine = nets.length - nRel;
      let share = nLine ? 0 : 1;
      if (nRel && nLine) share = RELATIVE_SHARE * clamp((analyzeSpeech(F, T).phrases.length - 1) / 3, 0, 1);
      const w = kind.map((k) => (k === 'relative' ? share / nRel : (1 - share) / nLine));
      const prob = new Float32Array(T * NE), ax = NA ? new Float32Array(T * NA) : null;
      nets.forEach((n, m) => {
        if (!w[m]) return;
        const Z = m === 0 ? Y : n.run(same(n, net) ? X : this._input(n, F, T, P, g), TT), Dz = n.Dout;
        for (let t = 0; t < T; t++) {
          const o = (t + P) * Dz, q = softmax(Array.from(Z.subarray(o + KM, o + KM + NE)));
          for (let e = 0; e < NE; e++) prob[t * NE + e] += w[m] * q[e];
          for (let a = 0; a < NA; a++) ax[t * NA + a] += w[m] * Z[o + KM + NE + a];
        }
      });
      for (let i = 0; i < prob.length; i++) logits[i] = Math.log(prob[i] + 1e-9);
      if (aux) aux.set(ax);
    }
    return { reg, logits, aux };
  }
  // F: normalized features (T*FEAT_DIM). returns {T, base(T*K, rig space), emo(T*6), voiced(T), plan}
  // opts: { seed, style: 'anime' | 'natural', profile (character acting), overrides (emotion ranges), exaggeration }
  generate(F, T, opts = {}) { return this.direct(this.runNet(F, T, opts.stats || null), F, T, opts); }
  // Second half of generate(): turn a reading of the speech (from runNet) into motion. Cheap, so the app keeps the
  // reading and calls this again whenever the user changes an emotion, the avatar, the style or the strength.
  direct(net, F, T, opts = {}) {
    const meta = this.meta, NE = EMOS.length;
    const { reg, logits, aux } = net;
    if ((opts.style || 'anime') !== 'natural') {
      const out = directOffline({ F, T, reg, logits, aux, meta, seed: opts.seed ?? 7, profile: opts.profile, overrides: opts.overrides || [], exaggeration: opts.exaggeration ?? 1 });
      return { T, base: out.track, emo: out.emo, voiced: out.speech, plan: out.plan };
    }
    const base = new Float32Array(T * K), emo = new Float32Array(T * NE), voiced = new Uint8Array(T);
    const pp = new PostProcessor(meta, opts);
    const y = new Float32Array(KM), rig = new Float32Array(K);
    for (let t = 0; t < T; t++) {
      y.set(reg.subarray(t * KM, (t + 1) * KM));
      voiced[t] = F[t * FEAT_DIM + 1] > 0.5 || F[t * FEAT_DIM] > -1.5 ? 1 : 0;
      pp.frame(y, voiced[t]);
      base.set(linearToRig(y, meta, rig), t * K);
      emo.set(softmax(Array.from(logits.subarray(t * NE, (t + 1) * NE))), t * NE);
    }
    // smooth emotion posteriors over ~1.5 s for a stable per-phrase reading
    const W = 22, sm = new Float32Array(emo.length);
    for (let t = 0; t < T; t++) for (let e = 0; e < NE; e++) { let s = 0, c = 0; for (let j = Math.max(0, t - W); j <= Math.min(T - 1, t + W); j++) { s += emo[j * NE + e]; c++; } sm[t * NE + e] = s / c; }
    return { T, base, emo: sm, voiced, plan: null };
  }
}

// Realtime: microphone samples in, emo frames out (delayed by the causal model's lookahead, ~133 ms)
export class LiveEmo {
  // opts: { seed, style: 'anime' | 'natural', profile, extra: further realtime networks whose reading of the feeling is
  // averaged with the main network's (a committee, as offline; mouth, accents and motion stay with the main network) }
  constructor(realtimeJson, opts = {}) {
    this.net = new EmoNet(realtimeJson);
    this.sn = new StreamingEmoNet(this.net);
    this.extra = (opts.extra || []).filter(Boolean).map((j) => new StreamingEmoNet(new EmoNet(j)));
    this.meta = this.net.meta;
    this.feat = new StreamingFeatures({ globals: this.meta.globals || [] });
    this.style = opts.style || 'anime';
    this.pp = new PostProcessor(this.meta, opts);
    this.dir = new LiveDirector(this.meta, opts);
    this.emo = new Float32Array(EMOS.length).fill(1 / EMOS.length);
    this.lastVoiced = []; this.lastFeat = [];
  }
  setStyle(style) { this.style = style; }
  setProfile(profile) { this.dir.setProfile(profile); }
  // false while a camera supplies the blinks (the director then stops placing its own)
  setAutoBlink(on) { this.dir.autoBlink = !!on; }
  // camera mouth (rig-space viseme AUs) may replace the network's mouth in the anime performer
  push(samples16k, camVis = null) {
    const frames = this.feat.push(samples16k);
    const out = [];
    const meta = this.meta, NE = EMOS.length, NA = (meta.aux || []).length;
    for (const f of frames) {
      const o = this.sn.step(f);
      const reg = new Float32Array(KM);
      for (let k = 0; k < KM; k++) reg[k] = o[k] * meta.y_std[k] + meta.y_mean[k];
      // the causal net answers for the frame `lookahead` steps back: keep features aligned with its output
      this.lastVoiced.push(f[1] > 0.5 || f[0] > -1.5 ? 1 : 0); this.lastFeat.push(f);
      const v = this.lastVoiced.length > meta.lookahead ? this.lastVoiced.shift() : 0;
      const fa = this.lastFeat.length > meta.lookahead ? this.lastFeat.shift() : null;
      const logits = Array.from(o.subarray(KM, KM + NE)), p = softmax(logits);
      if (this.extra.length) {
        // committee, as offline: 'line' readers and 'relative' readers (see runNet). A relative reader compares the line
        // with the voice heard so far, so its say grows with what is known of the voice (calibration 0..1, about four
        // lines) up to half.
        const kinds = [this.net, ...this.extra.map((x) => x.net)].map((n) => n.meta.reads || 'relative'), nRel = kinds.filter((k) => k === 'relative').length, nLine = kinds.length - nRel;
        const share = nRel && nLine ? 0.5 * this.feat.calibration : nLine ? 0 : 1, w = kinds.map((k) => (k === 'relative' ? share / nRel : (1 - share) / nLine));
        for (let e = 0; e < NE; e++) p[e] *= w[0];
        this.extra.forEach((ex, i) => { const z = ex.step(f), q = softmax(Array.from(z.subarray(KM, KM + NE))); for (let e = 0; e < NE; e++) p[e] += w[i + 1] * q[e]; });
      }
      for (let e = 0; e < NE; e++) this.emo[e] = this.emo[e] * 0.97 + p[e] * 0.03;
      let y;
      if (this.style === 'natural') { const yy = Float32Array.from(reg); this.pp.frame(yy, v); y = linearToRig(yy, meta); }
      // until the look-ahead buffer has filled (the first few frames of a session) the character simply rests
      else if (!fa) y = Float32Array.from(this.dir.perf.step({ speaking: 0 }));
      else y = Float32Array.from(this.dir.step({ feat: fa, ahead: this.lastFeat, reg, p, aux: NA ? o.subarray(KM + NE, KM + NE + NA) : null, voiced: v, camVis }));
      out.push({ y, emo: this.style === 'natural' ? Array.from(this.emo) : this.dir.emoDisplay(), voiced: v });
    }
    return out;
  }
}
export { MOUTH_CHANNELS };
