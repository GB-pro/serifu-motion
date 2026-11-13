// Serifu Motion — language-independent prosodic / spectral feature extractor.
// Shared by the training pipeline (Node) and the app (browser) so that features are identical.
// Input: mono Float32Array at 16 kHz. Output: frames at 30 fps.

export const SR = 16000;
export const FPS = 30;
export const RAW_DIM = 19; // see rawFrame()
export const FEAT_DIM = 20;
const NFFT = 512;
const NMEL = 26;
const NMFCC = 12;
const F0_MIN = 65, F0_MAX = 650;
const YIN_WIN = 800;

// ---------- FFT (radix-2, in place) ----------
const fftCache = new Map();
function fftTables(n) {
  if (fftCache.has(n)) return fftCache.get(n);
  const rev = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(-2 * Math.PI * i / n); sin[i] = Math.sin(-2 * Math.PI * i / n); }
  const t = { rev, cos, sin };
  fftCache.set(n, t);
  return t;
}
function fft(re, im) {
  const n = re.length, { rev, cos, sin } = fftTables(n);
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0, k = 0; j < half; j++, k += step) {
        const tr = re[i + j + half] * cos[k] - im[i + j + half] * sin[k];
        const ti = re[i + j + half] * sin[k] + im[i + j + half] * cos[k];
        re[i + j + half] = re[i + j] - tr; im[i + j + half] = im[i + j] - ti;
        re[i + j] += tr; im[i + j] += ti;
      }
    }
  }
}

// ---------- mel filterbank + DCT ----------
const hz2mel = (h) => 2595 * Math.log10(1 + h / 700);
const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);
let MELFB = null, DCT = null, HANN = null;
function initTables() {
  if (MELFB) return;
  HANN = new Float64Array(NFFT);
  for (let i = 0; i < NFFT; i++) HANN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NFFT - 1));
  const lo = hz2mel(80), hi = hz2mel(7600);
  const pts = [];
  for (let i = 0; i < NMEL + 2; i++) pts.push(Math.floor((NFFT + 1) * mel2hz(lo + (hi - lo) * i / (NMEL + 1)) / SR));
  MELFB = [];
  for (let m = 1; m <= NMEL; m++) {
    const f = new Float64Array(NFFT / 2 + 1);
    for (let k = pts[m - 1]; k < pts[m]; k++) f[k] = (k - pts[m - 1]) / Math.max(1, pts[m] - pts[m - 1]);
    for (let k = pts[m]; k < pts[m + 1]; k++) f[k] = (pts[m + 1] - k) / Math.max(1, pts[m + 1] - pts[m]);
    MELFB.push(f);
  }
  DCT = [];
  for (let c = 1; c <= NMFCC; c++) {
    const row = new Float64Array(NMEL);
    for (let m = 0; m < NMEL; m++) row[m] = Math.cos(Math.PI * c * (m + 0.5) / NMEL);
    DCT.push(row);
  }
}

// ---------- YIN pitch ----------
function yin(x, start) {
  const W = YIN_WIN / 2; // integration window
  const tauMin = Math.floor(SR / F0_MAX), tauMax = Math.min(Math.floor(SR / F0_MIN), W);
  const d = new Float64Array(tauMax + 1);
  for (let tau = 1; tau <= tauMax; tau++) {
    let s = 0;
    for (let j = 0; j < W; j++) { const a = (x[start + j] || 0) - (x[start + j + tau] || 0); s += a * a; }
    d[tau] = s;
  }
  // cumulative mean normalized difference
  let run = 0; const cmnd = new Float64Array(tauMax + 1); cmnd[0] = 1;
  for (let tau = 1; tau <= tauMax; tau++) { run += d[tau]; cmnd[tau] = run > 0 ? d[tau] * tau / run : 1; }
  let best = -1;
  for (let tau = tauMin; tau <= tauMax; tau++) {
    if (cmnd[tau] < 0.15) {
      while (tau + 1 <= tauMax && cmnd[tau + 1] < cmnd[tau]) tau++;
      best = tau; break;
    }
  }
  if (best < 0) { // take global min
    let mv = 1e9;
    for (let tau = tauMin; tau <= tauMax; tau++) if (cmnd[tau] < mv) { mv = cmnd[tau]; best = tau; }
  }
  const ap = cmnd[best];
  // parabolic interpolation
  let t = best;
  if (best > tauMin && best < tauMax) {
    const a = cmnd[best - 1], b = cmnd[best], c = cmnd[best + 1];
    const den = a - 2 * b + c;
    if (Math.abs(den) > 1e-12) t = best + 0.5 * (a - c) / den;
  }
  return { f0: SR / t, ap };
}

// ---------- raw per-frame features ----------
// raw: [0] log energy dB, [1] f0 Hz (0 = unvoiced), [2] aperiodicity, [3] spectral flux,
//      [4] spectral centroid (Hz), [5] zcr, [6..17] mfcc c1..c12, [18] hf ratio
export function extractRaw(x) {
  initTables();
  const nFrames = Math.max(1, Math.floor(x.length / SR * FPS));
  const out = new Float32Array(nFrames * RAW_DIM);
  let prevMag = null;
  for (let i = 0; i < nFrames; i++) {
    const r = rawAt(x, Math.round(i * SR / FPS), prevMag);
    prevMag = r.mag;
    out.set(r.vals, i * RAW_DIM);
  }
  // median-of-3 cleanup of f0 octave jumps
  for (let i = 1; i < nFrames - 1; i++) {
    const a = out[(i - 1) * RAW_DIM + 1], b = out[i * RAW_DIM + 1], cc = out[(i + 1) * RAW_DIM + 1];
    if (b > 0 && a > 0 && cc > 0) { const med = [a, b, cc].sort((p, q) => p - q)[1]; if (Math.abs(Math.log2(b / med)) > 0.6) out[i * RAW_DIM + 1] = med; }
  }
  return { raw: out, nFrames };
}

// ---------- speaker statistics (for normalization) ----------
export function speakerStats(rawList) {
  // rawList: array of {raw, nFrames}
  const eAll = [], eV = [], f0 = [], mf = Array.from({ length: NMFCC }, () => []), cent = [], flux = [];
  for (const { raw, nFrames } of rawList) {
    for (let i = 0; i < nFrames; i++) {
      const o = i * RAW_DIM;
      eAll.push(raw[o]);
      if (raw[o + 1] > 0) { f0.push(Math.log2(raw[o + 1]) * 12); eV.push(raw[o]); }
    }
  }
  const eSorted = eAll.slice().sort((a, b) => a - b);
  const q = (arr, p) => arr.length ? arr[Math.min(arr.length - 1, Math.floor(p * arr.length))] : 0;
  // Speech level: measured on the voiced frames only, so that it does not depend on how much of the recording is
  // silence. (The first versions took the 90th percentile of all frames; in a recording where the speaker talks less
  // than a tenth of the time that percentile lands on the silence, and the silence then counts as speech.) The 70th
  // percentile of the voiced frames plus 0.5 dB gives the same level as the old rule on ordinary recordings.
  // Without any voiced frame the level is not allowed below -40 dB, so that noise alone is not taken for speech.
  const eHi = eV.length ? q(eV.sort((a, b) => a - b), 0.7) + 0.5 : Math.max(q(eSorted, 0.9), -40);
  const eThr = eHi - 25;
  for (const { raw, nFrames } of rawList) for (let i = 0; i < nFrames; i++) {
    const o = i * RAW_DIM;
    if (raw[o] > eThr) {
      for (let k = 0; k < NMFCC; k++) mf[k].push(raw[o + 6 + k]);
      cent.push(raw[o + 4]); flux.push(raw[o + 3]);
    }
  }
  const mean = (a) => a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0;
  const std = (a, m) => a.length > 1 ? Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) + 1e-6 : 1;
  const f0s = f0.slice().sort((a, b) => a - b);
  const st = {
    eSpeech: eHi,
    f0Median: f0s.length ? q(f0s, 0.5) : 12 * Math.log2(160),
    f0Spread: f0s.length > 10 ? Math.max(1.5, (q(f0s, 0.9) - q(f0s, 0.1)) / 2) : 3,
    mfMean: mf.map(mean), mfStd: null, centMean: mean(cent), centStd: 0, fluxMean: mean(flux), fluxStd: 0,
    nVoiced: f0.length, // how much of the voice the statistics rest on (frames)
  };
  st.mfStd = mf.map((a, k) => std(a, st.mfMean[k]));
  st.centStd = std(cent, st.centMean); st.fluxStd = std(flux, st.fluxMean);
  return st;
}

// What is assumed about a voice before any of it has been heard (live mode blends these with the measured values while
// the first seconds come in). Timbre and pitch range are the averages of the training speakers; the level is a guess
// that keeps room noise from counting as speech.
export const DEFAULT_STATS = {
  eSpeech: -25, f0Median: 90.2, f0Spread: 7.5,
  mfMean: [35.8, 11.4, 8.7, -0.3, -1.1, 0.2, -1.5, 0.3, -0.35, -0.3, -0.4, -1.65], mfStd: [14.2, 13.6, 8.5, 8.7, 6.3, 5.4, 4.8, 4.1, 3.9, 3.7, 3.6, 3.3],
  centMean: 506, centStd: 563, fluxMean: 1.38, fluxStd: 0.94, nVoiced: 0,
};

// ---------- recording-level inputs ("globals") ----------
// What the recording as a whole sounds like, in absolute terms. Everything in normalize() is relative to the recording
// itself, so in a short recording, or one that is all in a single emotion, the cues "higher / louder / brighter than
// this speaker usually is" are gone. These values give the network the absolute side: how high the voice is, how wide
// its pitch moves, how much speech the statistics rest on, how strongly the timbre varies. A model lists the ones it
// was trained with in meta.globals (indices into this vector). Centre and scale are rounded population values of the
// corpus (audio statistics only); train/simnorm.py holds the same numbers.
export const NG = 30;
const POP = {
  f0Median: 89.7, f0MedianS: 5, f0Spread: 7.5, f0SpreadS: 2, centMean: 520, centMeanS: 150, centStd: 620, centStdS: 230,
  mfMean: [35, 12, 8.5, 0, -0.5, 0, -1.5, 0.5, -0.5, -0.5, -0.5, -1.5], mfMeanS: [4, 5, 3, 3.5, 2.5, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5],
  mfStd: [14.5, 13.5, 8.5, 8.5, 6.5, 5.5, 5, 4, 4, 3.5, 3.5, 3.5], mfStdS: [2.5, 2.5, 1.5, 1.5, 1, 1, 1, 0.75, 0.75, 0.75, 0.75, 0.75],
  fluxStd: 0.95, fluxStdS: 0.15,
};
export function globalFeatures(st, out = new Float32Array(NG)) {
  const P = POP, c = (v) => Math.max(-4, Math.min(4, v));
  out[0] = c((st.f0Median - P.f0Median) / P.f0MedianS);
  out[1] = c((st.f0Spread - P.f0Spread) / P.f0SpreadS);
  out[2] = Math.min(1, Math.log2(1 + (st.nVoiced || 0) / 75) / 6);
  out[3] = c((st.centMean - P.centMean) / P.centMeanS);
  out[4] = c((st.centStd - P.centStd) / P.centStdS);
  for (let k = 0; k < NMFCC; k++) { out[5 + k] = c((st.mfMean[k] - P.mfMean[k]) / P.mfMeanS[k]); out[17 + k] = c((st.mfStd[k] - P.mfStd[k]) / P.mfStdS[k]); }
  out[29] = c((st.fluxStd - P.fluxStd) / P.fluxStdS);
  return out;
}

// ---------- normalized model input ----------
// [0] energy rel speech level /10, [1] voiced, [2] f0 st rel median / spread, [3] d f0, [4] d energy,
// [5] flux z, [6] centroid z, [7] zcr*10, [8..19] mfcc z
export function normalize(raw, nFrames, st) {
  const F = new Float32Array(nFrames * FEAT_DIM);
  let prevF0 = 0, prevE = 0, lastVoicedF0 = 0;
  for (let i = 0; i < nFrames; i++) {
    const o = i * RAW_DIM, p = i * FEAT_DIM;
    const e = (raw[o] - st.eSpeech) / 10;
    const v = raw[o + 1] > 0 ? 1 : 0;
    const f = v ? (Math.log2(raw[o + 1]) * 12 - st.f0Median) / st.f0Spread : 0;
    F[p] = Math.max(-4, Math.min(2, e));
    F[p + 1] = v;
    F[p + 2] = Math.max(-4, Math.min(4, f));
    F[p + 3] = v && lastVoicedF0 !== 0 ? Math.max(-2, Math.min(2, f - prevF0)) : 0;
    F[p + 4] = Math.max(-2, Math.min(2, F[p] - prevE));
    F[p + 5] = Math.max(-3, Math.min(3, (raw[o + 3] - st.fluxMean) / st.fluxStd));
    F[p + 6] = Math.max(-3, Math.min(3, (raw[o + 4] - st.centMean) / st.centStd));
    F[p + 7] = Math.min(3, raw[o + 5] * 10);
    const gate = F[p] > -2.5 ? 1 : 0.2; // de-emphasize MFCC in silence
    for (let k = 0; k < NMFCC; k++) F[p + 8 + k] = gate * Math.max(-3, Math.min(3, (raw[o + 6 + k] - st.mfMean[k]) / st.mfStd[k]));
    if (v) { prevF0 = f; lastVoicedF0 = 1; }
    prevE = F[p];
  }
  return F;
}

// ---------- streaming extractor for realtime mic ----------
export class StreamingFeatures {
  // opts.globals: indices of the recording-level inputs (globalFeatures) to append to every frame, as the model lists them
  constructor(opts = {}) {
    this.buf = new Float32Array(SR * 4);
    this.len = 0; // samples written total
    this.frameIdx = 0;
    this.stats = JSON.parse(JSON.stringify(DEFAULT_STATS));
    this.hist = [];
    this.prevF0 = 0; this.prevE = 0; this.lastVoiced = 0;
    this.calibration = 0; // 0..1: how well this speaker's own pitch and level are known yet
    this.globals = opts.globals || []; this._g = new Float32Array(NG); globalFeatures(this.stats, this._g);
  }
  // push mono 16k samples; returns array of new normalized frames (Float32Array FEAT_DIM, plus the model's globals)
  push(samples) {
    const out = [];
    for (let i = 0; i < samples.length; i++) { this.buf[this.len % this.buf.length] = samples[i]; this.len++; }
    while (true) {
      const c = Math.round(this.frameIdx * SR / FPS);
      if (c + Math.max(NFFT, YIN_WIN) > this.len) break;
      // gather window around c (need c - 400 .. c + 400 + tau)
      const s0 = c - YIN_WIN / 2, n = YIN_WIN + 420;
      const w = new Float32Array(n + NFFT);
      for (let j = 0; j < w.length; j++) { const idx = s0 + j; w[j] = idx >= 0 && idx < this.len && idx > this.len - this.buf.length ? this.buf[idx % this.buf.length] : 0; }
      const r = rawAt(w, YIN_WIN / 2, this._prevMag);
      this._prevMag = r.mag;
      out.push(this.pushRaw(r.vals));
    }
    return out;
  }
  // one raw frame in, one normalized frame out; the statistics of the session so far are updated on the way.
  // (Training replays recorded raw frames through this method, so the network learns from exactly what live mode sees.)
  pushRaw(vals) {
    this.hist.push(vals);
    if (this.hist.length > 30 * 300) this.hist.shift(); // the speaker's statistics come from the last 5 minutes
    if (this.frameIdx % (this.calibration < 1 ? 5 : 15) === 0 && this.hist.length > 20) this._updateStats();
    this.frameIdx++;
    return this._norm(vals);
  }
  _updateStats() {
    const n = this.hist.length;
    const raw = new Float32Array(n * RAW_DIM);
    this.hist.forEach((v, i) => raw.set(v, i * RAW_DIM));
    const s = speakerStats([{ raw, nFrames: n }]);
    // blend with the prior while little of the voice has been heard: about 2.5 s of voiced speech settle the speaker's
    // pitch and level (silence tells nothing about either, so only voiced frames count)
    const nv = s.nVoiced;
    const a = Math.min(1, nv / 75); this.calibration = a;
    const d = DEFAULT_STATS, m = (x, y) => x * a + y * (1 - a);
    this.stats = {
      eSpeech: m(s.eSpeech, d.eSpeech), f0Median: m(s.f0Median, d.f0Median), f0Spread: m(s.f0Spread, d.f0Spread),
      mfMean: s.mfMean.map((v, k) => m(v, d.mfMean[k])), mfStd: s.mfStd.map((v, k) => m(v, d.mfStd[k])),
      centMean: m(s.centMean, d.centMean), centStd: m(s.centStd, d.centStd), fluxMean: m(s.fluxMean, d.fluxMean), fluxStd: m(s.fluxStd, d.fluxStd),
      nVoiced: nv,
    };
    globalFeatures(this.stats, this._g);
  }
  _norm(vals) {
    const F0 = normalize(vals, 1, this.stats), G = this.globals;
    const F = G.length ? new Float32Array(FEAT_DIM + G.length) : F0;
    if (G.length) { F.set(F0); for (let j = 0; j < G.length; j++) F[FEAT_DIM + j] = this._g[G[j]]; }
    // fix deltas using history state
    const v = F[1];
    F[3] = v && this.lastVoiced ? Math.max(-2, Math.min(2, F[2] - this.prevF0)) : 0;
    F[4] = Math.max(-2, Math.min(2, F[0] - this.prevE));
    if (v) { this.prevF0 = F[2]; this.lastVoiced = 1; }
    this.prevE = F[0];
    return F;
  }
}

// single-frame raw computation used by the streaming path (same math as extractRaw)
function rawAt(x, c, prevMag) {
  initTables();
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  const s0 = c - NFFT / 2;
  let e = 0, zc = 0, prev = 0;
  for (let j = 0; j < NFFT; j++) {
    const v = x[s0 + j] || 0; e += v * v;
    if (j > 0 && (v >= 0) !== (prev >= 0)) zc++;
    prev = v; re[j] = v * HANN[j]; im[j] = 0;
  }
  const edb = 10 * Math.log10(e / NFFT + 1e-10);
  fft(re, im);
  const mag = new Float64Array(NFFT / 2 + 1);
  let num = 0, den = 0, hf = 0;
  for (let k = 0; k <= NFFT / 2; k++) { const p = re[k] * re[k] + im[k] * im[k]; mag[k] = Math.sqrt(p); num += k * SR / NFFT * p; den += p; if (k * SR / NFFT > 4000) hf += p; }
  let flux = 0;
  if (prevMag) for (let k = 0; k <= NFFT / 2; k++) { const d = mag[k] - prevMag[k]; if (d > 0) flux += d; }
  const vals = new Float32Array(RAW_DIM);
  vals[0] = edb;
  const { f0, ap } = yin(x, c - YIN_WIN / 2);
  vals[1] = ap < 0.3 && edb > -55 && f0 >= F0_MIN && f0 <= F0_MAX ? f0 : 0;
  vals[2] = ap; vals[3] = Math.log(1 + flux); vals[4] = den > 0 ? num / den : 0; vals[5] = zc / NFFT;
  const mel = new Float64Array(NMEL);
  for (let m = 0; m < NMEL; m++) { let t = 0; const f = MELFB[m]; for (let k = 0; k <= NFFT / 2; k++) if (f[k]) t += f[k] * mag[k] * mag[k]; mel[m] = Math.log(t + 1e-8); }
  for (let q = 0; q < NMFCC; q++) { let s = 0; for (let m = 0; m < NMEL; m++) s += DCT[q][m] * mel[m]; vals[6 + q] = s; }
  vals[18] = den > 0 ? hf / den : 0;
  return { vals, mag };
}

// ---------- resample helper (browser decoded audio -> 16 kHz) ----------
export function resampleTo16k(data, srcRate) {
  if (srcRate === SR) return data;
  const ratio = srcRate / SR, n = Math.floor(data.length / ratio);
  const out = new Float32Array(n);
  // simple windowed-average low-pass + linear interp
  const k = Math.max(1, Math.floor(ratio));
  for (let i = 0; i < n; i++) {
    const pos = i * ratio, i0 = Math.floor(pos);
    let s = 0, c = 0;
    for (let j = -k + 1; j < k; j++) { const v = data[i0 + j]; if (v !== undefined) { s += v; c++; } }
    out[i] = c ? s / c : 0;
  }
  return out;
}
