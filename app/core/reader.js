// Reading the feeling of a line with a speech encoder that was trained beforehand on a very large amount of speech in
// many languages (the first blocks of the encoder of OpenAI's Whisper, MIT licence), instead of the hand-made frames
// of features.js. Why: a reader trained on those frames only held for the sentences, rooms and language of the corpus
// it was trained on (train/SPLITS.md, "What the checks on CREMA-D could not show").
//
// This file holds the parts that do not depend on where the encoder runs: the encoder's input (log-mel spectrogram,
// exactly as Whisper computes it) and the cutting of a recording into the lines that are read.
import { SR, FPS, RAW_DIM, speakerStats } from './features.js';

const N_FFT = 400, HOP = 160, N_MELS = 80, NBIN = N_FFT / 2 + 1;
let FILT = null, WIN = null, COS = null, SIN = null;

function initMel() {
  if (FILT) return;
  // mel filterbank as librosa.filters.mel(sr=16000, n_fft=400, n_mels=80): Slaney scale, area-normalised triangles
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  const hz2mel = (f) => (f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fsp);
  const mel2hz = (m) => (m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m);
  const mmax = hz2mel(SR / 2), pts = Array.from({ length: N_MELS + 2 }, (_, i) => mel2hz((mmax * i) / (N_MELS + 1)));
  FILT = [];
  for (let m = 0; m < N_MELS; m++) {
    const lo = pts[m], ce = pts[m + 1], hi = pts[m + 2], norm = 2 / (hi - lo), w = [];
    let k0 = -1;
    for (let k = 0; k < NBIN; k++) {
      const f = (k * SR) / N_FFT, v = Math.max(0, Math.min((f - lo) / (ce - lo), (hi - f) / (hi - ce))) * norm;
      if (v > 0) { if (k0 < 0) k0 = k; w.push(v); } else if (k0 >= 0) break;
    }
    FILT.push({ k0: Math.max(0, k0), w: Float32Array.from(w) });
  }
  WIN = new Float32Array(N_FFT); for (let n = 0; n < N_FFT; n++) WIN[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT); // periodic Hann
  COS = new Float32Array(N_FFT); SIN = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) { COS[n] = Math.cos((2 * Math.PI * n) / N_FFT); SIN[n] = Math.sin((2 * Math.PI * n) / N_FFT); }
}

// x: Float32Array of samples at 16 kHz -> { data: Float32Array(80 * frames), frames } laid out [band][frame], the
// encoder's input (1, 80, frames). One frame per 10 ms. Same numbers as whisper/audio.py: log_mel_spectrogram.
export function logMel(x) {
  initMel();
  const n = Math.floor(x.length / HOP), pad = N_FFT / 2, L = x.length;
  const at = (i) => { let j = i - pad; if (j < 0) j = -j; else if (j >= L) j = 2 * (L - 1) - j; return x[Math.min(L - 1, Math.max(0, j))]; }; // reflect
  const out = new Float32Array(N_MELS * n), fr = new Float32Array(N_FFT), pw = new Float32Array(NBIN);
  let mx = -Infinity;
  for (let t = 0; t < n; t++) {
    const s = t * HOP;
    for (let i = 0; i < N_FFT; i++) fr[i] = at(s + i) * WIN[i];
    // 400-point transform, the 201 bins that are needed. Even and odd parts of the frame halve the work.
    for (let k = 0; k < NBIN; k++) {
      let re = fr[0] + (k & 1 ? -fr[200] : fr[200]), im = 0, idx = 0;
      for (let i = 1; i < 200; i++) { idx += k; if (idx >= N_FFT) idx -= N_FFT; re += (fr[i] + fr[N_FFT - i]) * COS[idx]; im += (fr[i] - fr[N_FFT - i]) * SIN[idx]; }
      pw[k] = re * re + im * im;
    }
    for (let m = 0; m < N_MELS; m++) {
      const f = FILT[m]; let e = 0;
      for (let j = 0; j < f.w.length; j++) e += f.w[j] * pw[f.k0 + j];
      const v = Math.log10(Math.max(e, 1e-10)); out[m * n + t] = v; if (v > mx) mx = v;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], mx - 8) + 4) / 4;
  return { data: out, frames: n };
}

// The lines of a recording that are read one by one: phrases (animate.js: analyzeSpeech) that follow each other with
// less than LINE_GAP of pause belong to one line; a line is at most LINE_MAX long and is cut at its longest pause when
// it would be longer. Each line is handed to the encoder with MARGIN of the recording on either side.
export const LINE_GAP = 1.0, LINE_MAX = 10, MARGIN = 0.25;
export function linesOf(phrases) {
  const out = [];
  const add = (ps) => {
    if ((ps[ps.length - 1].t1 - ps[0].t0) / FPS <= LINE_MAX || ps.length < 2) { out.push({ t0: ps[0].t0, t1: ps[ps.length - 1].t1, phrases: ps }); return; }
    let cut = 1, best = -1;
    for (let i = 1; i < ps.length; i++) { const g = ps[i].t0 - ps[i - 1].t1; if (g > best) { best = g; cut = i; } }
    add(ps.slice(0, cut)); add(ps.slice(cut));
  };
  let cur = [];
  for (const ph of phrases) {
    if (cur.length && (ph.t0 - cur[cur.length - 1].t1) / FPS >= LINE_GAP) { add(cur); cur = []; }
    cur.push(ph);
  }
  if (cur.length) add(cur);
  return out;
}
// Speech level of a line in dB (features.js: speakerStats on the frames of the line alone).
export function lineLevel(raw, line) {
  return speakerStats([{ raw: raw.subarray(line.t0 * RAW_DIM, line.t1 * RAW_DIM), nFrames: line.t1 - line.t0 }]).eSpeech;
}
// Samples of a line for the encoder: the line with its margins (never more than LINE_MAX + 2 s), brought to the same
// speech level whatever the recording level was. How loud a recording is says nothing about the feeling (it depends on
// the microphone and its gain, and differs by 25 dB between corpora), so the encoder never sees it; how loud a line is
// compared with the other lines of the same recording is passed to the reader as a number of its own (see SpeechReader).
export const LINE_LEVEL = -20;
export function lineSamples(x, raw, line) {
  const a = Math.max(0, Math.round((line.t0 / FPS - MARGIN) * SR)), b = Math.min(x.length, Math.round((line.t1 / FPS + MARGIN) * SR), a + (LINE_MAX + 2) * SR);
  const level = lineLevel(raw, line), g = Math.pow(10, Math.max(-30, Math.min(40, LINE_LEVEL - level)) / 20);
  const out = new Float32Array(Math.max(b - a, 1)); for (let i = 0; i < out.length; i++) out[i] = (x[a + i] || 0) * g;
  return { samples: out, level };
}
