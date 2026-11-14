// Serifu Motion — which vowel is being sounded, frame by frame, from the voice itself.
// The two lowest resonances of the vocal tract (formants F1, F2) tell the vowels apart in every language: F1 rises
// as the jaw opens (a), F2 rises as the tongue moves forward (i, e) and falls as the lips round (o, u). They are
// read from the spectral envelope (linear prediction), set against the speaker's own range (so a child, a woman and
// a man give the same answer for the same vowel), and turned into soft weights for the five Japanese vowels plus
// "closed" (m, b, p and the silences between syllables).
//   vowelTrack(x16k, T) -> Float32Array(T * 7): per frame [closed, a, i, u, e, o, loud] (weights sum to 1, loud 0..1)
import { SR, FPS } from './features.js';
export const VOWELS = ['X', 'A', 'I', 'U', 'E', 'O'];
// where each vowel sits in the speaker-normalized (F1, F2) plane (z-scores over the speaker's voiced frames);
// set from the usual Japanese vowel chart and checked on recordings with known text
const CENTER = { A: [1.35, 0.05], I: [-1.15, 1.35], U: [-0.95, -0.25], E: [0.15, 0.85], O: [0.25, -1.25] };
const WIN = 480, ORDER = 14, FMAX = 3600, DF = 20;

function lpc(w, p) { // autocorrelation method, Levinson-Durbin; returns a[0..p] with a[0] = 1
  const r = new Float64Array(p + 1);
  for (let k = 0; k <= p; k++) { let s = 0; for (let i = k; i < w.length; i++) s += w[i] * w[i - k]; r[k] = s; }
  if (r[0] <= 1e-12) return null;
  r[0] *= 1.0001; // a touch of white noise keeps the solution stable
  const a = new Float64Array(p + 1); a[0] = 1; let e = r[0];
  for (let i = 1; i <= p; i++) {
    let acc = r[i]; for (let j = 1; j < i; j++) acc += a[j] * r[i - j];
    const k = -acc / e, tmp = a.slice();
    for (let j = 1; j < i; j++) a[j] = tmp[j] + k * tmp[i - j];
    a[i] = k; e *= 1 - k * k; if (e <= 0) return null;
  }
  return a;
}
let COS = null, SIN = null;
function tables() {
  if (COS) return; const nF = Math.floor(FMAX / DF); COS = []; SIN = [];
  for (let f = 0; f < nF; f++) { const w = 2 * Math.PI * (f * DF) / SR, c = new Float64Array(ORDER + 1), s = new Float64Array(ORDER + 1); for (let k = 0; k <= ORDER; k++) { c[k] = Math.cos(w * k); s[k] = Math.sin(w * k); } COS.push(c); SIN.push(s); }
}
// F1, F2 of one frame (Hz), or null when the envelope shows no clear pair of peaks
function formants(x, c) {
  const w = new Float64Array(WIN); let prev = x[c - WIN / 2 - 1] || 0;
  for (let i = 0; i < WIN; i++) { const v = x[c - WIN / 2 + i] || 0; w[i] = (v - 0.97 * prev) * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (WIN - 1))); prev = v; }
  const a = lpc(w, ORDER); if (!a) return null;
  const env = new Float64Array(COS.length);
  for (let f = 0; f < COS.length; f++) { let re = 0, im = 0; const cc = COS[f], ss = SIN[f]; for (let k = 0; k <= ORDER; k++) { re += a[k] * cc[k]; im -= a[k] * ss[k]; } env[f] = -10 * Math.log10(re * re + im * im + 1e-12); }
  const peaks = []; for (let f = 1; f < env.length - 1; f++) if (env[f] > env[f - 1] && env[f] >= env[f + 1]) { const d = env[f - 1] - 2 * env[f] + env[f + 1], off = d < 0 ? 0.5 * (env[f - 1] - env[f + 1]) / d : 0; peaks.push({ hz: (f + off) * DF, db: env[f] }); }
  const p1 = peaks.find((p) => p.hz >= 220 && p.hz <= 1100); if (!p1) return null;
  const p2 = peaks.find((p) => p.hz >= Math.max(650, p1.hz + 250) && p.hz <= 3000); if (!p2) return null;
  return [p1.hz, p2.hz];
}

export function vowelTrack(x, T) {
  tables();
  const hop = SR / FPS, NV = VOWELS.length, D = NV + 1, out = new Float32Array(T * D);
  const db = new Float32Array(T), F = new Float32Array(T * 2), ok = new Uint8Array(T);
  for (let t = 0; t < T; t++) {
    const c = Math.round((t + 0.5) * hop); let s = 0; for (let i = c - 240; i < c + 240; i++) { const v = x[i] || 0; s += v * v; }
    db[t] = 10 * Math.log10(s / 480 + 1e-10);
  }
  const sorted = Array.from(db).sort((p, q) => q - p), top = sorted[Math.floor(T * 0.05)] ?? -20; // loud speech level of this recording
  for (let t = 0; t < T; t++) if (db[t] > top - 28) { const f = formants(x, Math.round((t + 0.5) * hop)); if (f) { F[2 * t] = f[0]; F[2 * t + 1] = f[1]; ok[t] = 1; } }
  // three-point median against single-frame errors
  const med = (t, j) => { const v = []; for (let u = t - 1; u <= t + 1; u++) if (u >= 0 && u < T && ok[u]) v.push(F[2 * u + j]); v.sort((p, q) => p - q); return v[v.length >> 1]; };
  const G = new Float32Array(T * 2); for (let t = 0; t < T; t++) if (ok[t]) { G[2 * t] = med(t, 0); G[2 * t + 1] = med(t, 1); }
  // the speaker's range (log Hz), from frames loud enough to be vowels
  const st = [0, 1].map((j) => { const v = []; for (let t = 0; t < T; t++) if (ok[t] && db[t] > top - 18) v.push(Math.log(G[2 * t + j])); if (v.length < 20) return null; const m = v.reduce((p, q) => p + q, 0) / v.length; const sd = Math.sqrt(v.reduce((p, q) => p + (q - m) ** 2, 0) / v.length) || 0.1; return [m, sd]; });
  for (let t = 0; t < T; t++) {
    const o = t * D, loud = Math.max(0, Math.min(1, (db[t] - (top - 30)) / 24)); out[o + NV] = loud;
    // closed: quiet frames inside speech (the lips meet for m/b/p, and between syllables)
    let wX = Math.max(0, Math.min(1, (top - 20 - db[t]) / 8));
    const w = new Array(NV).fill(0);
    if (ok[t] && st[0] && st[1]) {
      const z1 = (Math.log(G[2 * t]) - st[0][0]) / st[0][1], z2 = (Math.log(G[2 * t + 1]) - st[1][0]) / st[1][1];
      let s = 0; VOWELS.forEach((v, i) => { if (!i) return; const c = CENTER[v], d = (z1 - c[0]) ** 2 + (z2 - c[1]) ** 2; w[i] = Math.exp(-d / 0.5); s += w[i]; });
      if (s > 1e-6) for (let i = 1; i < NV; i++) w[i] = (w[i] / s) * (1 - wX); else wX = 1;
    } else wX = 1;
    w[0] = wX; for (let i = 0; i < NV; i++) out[o + i] = w[i];
  }
  // light smoothing over time (the mouth glides between vowels; weights stay normalized)
  const sm = new Float32Array(out.length); for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) { const a = out[Math.max(0, t - 1) * D + i], b = out[t * D + i], c = out[Math.min(T - 1, t + 1) * D + i]; sm[t * D + i] = 0.25 * a + 0.5 * b + 0.25 * c; }
  return sm;
}
