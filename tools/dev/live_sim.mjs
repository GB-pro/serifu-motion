// Live mode without a phone: the take's audio is pushed through the streaming path in small chunks, exactly as the
// microphone would deliver it; optionally the "camera" (the actor's tracked mouth and blink) is fed alongside.
// node tools/dev/live_sim.mjs <take.json> [cam]
import fs from 'fs';
import { LiveEmo } from '../../app/core/generator.js';
import { visemesFromCamera } from '../../app/core/animate.js';
import { IDS, IDX, K, KM, EMOS } from '../../app/core/au.js';
const takeP = process.argv[2], useCam = process.argv[3] === 'cam';
const tk = JSON.parse(fs.readFileSync(takeP)), base = takeP.slice(0, -5);
const rt = JSON.parse(fs.readFileSync(new URL('../../app/models/emo_realtime.json', import.meta.url)));
const profile = JSON.parse(fs.readFileSync(new URL('../../app/avatar/acting.json', import.meta.url)))[tk.avatar] || null;
const wav = fs.readFileSync(base + '.wav'); const pcm = new Int16Array(wav.buffer.slice(wav.byteOffset + 44, wav.byteOffset + wav.length - ((wav.length - 44) % 2)));
const x = Float32Array.from(pcm, (v) => v / 32768);
let cam = null; if (useCam) { const b = fs.readFileSync(base + '.cam.f32'); cam = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const live = new LiveEmo(rt, { seed: 11, style: 'anime', profile });
const LA = live.meta.lookahead || 0;
const passes = +(process.argv[4] || 1);   // 2 = the same take twice, scored on the second pass (the speaker's voice is known by then)
let frames = [], fi = 0; const t0 = performance.now(); let worst = 0, total = 0;
for (let pass = 0; pass < passes; pass++) { frames = []; fi = 0;
for (let o = 0; o < x.length; o += 800) {            // 50 ms chunks
  const chunk = x.subarray(o, Math.min(x.length, o + 800));
  let camVis = null;
  if (cam) { const f = Math.min(tk.T - 1, Math.max(0, fi - LA)); const au = {}; for (let k = 0; k < KM; k++) au[IDS[k]] = cam[f * KM + k]; camVis = visemesFromCamera(au, live.meta, {}); live.setAutoBlink(false); }
  const s = performance.now(); const out = live.push(chunk, camVis); worst = Math.max(worst, (performance.now() - s) / Math.max(1, out.length));
  for (const f of out) { const y = Float32Array.from(f.y); if (cam) { const g = Math.min(tk.T - 1, Math.max(0, fi - LA)); y[IDX.AU45] = Math.min(1, Math.max(0, (cam[g * KM + IDX.AU45] - 0.12) / 0.55)); } frames.push({ y, emo: f.emo, voiced: f.voiced, played: live.dir.emo, calib: live.feat.calibration }); fi++; total++; }
} }
const ms = (performance.now() - t0) / total;
const N = frames.length; let bad = 0, step = 0; const hi = ['headPitch', 'headYaw', 'headRoll'].map((n) => IDX[n]);
for (let i = 0; i < N; i++) { for (let k = 0; k < K; k++) if (!Number.isFinite(frames[i].y[k])) bad++; if (i) for (const k of hi) step = Math.max(step, Math.abs(frames[i].y[k] - frames[i - 1].y[k])); }
// silence well away from speech: is the mouth closed?
const sp = frames.map((f) => f.voiced); let closed = 0, nsil = 0;
for (let i = 0; i < N; i++) { let near = false; for (let j = -6; j <= 4; j++) if (sp[i + j]) near = true; if (!near) { nsil++; if (frames[i].y[IDX.AU26] < 0.12) closed++; } }
// the feeling shown in the middle of each line (the output lags the audio by the model's look-ahead)
const rows = tk.clips.map((c) => { const m = Math.min(N - 1, Math.round((c.f0 + c.f1) / 2) + LA + 8); const p = frames[m].emo; let b = 0; for (let e = 1; e < p.length; e++) if (p[e] > p[b]) b = e; return `${c.label}->${frames[m].played}${frames[m].played === c.label ? '' : ' X'} (heard ${EMOS[b]}, voice known ${frames[m].calib.toFixed(2)})`; });
const range = (id) => { let lo = 1e9, hi2 = -1e9; for (const f of frames) { lo = Math.min(lo, f.y[IDX[id]]); hi2 = Math.max(hi2, f.y[IDX[id]]); } return `${lo.toFixed(1)}..${hi2.toFixed(1)}`; };
let jawCorr = null; if (cam) { const a = [], b = []; for (let i = LA; i < N; i++) { a.push(frames[i].y[IDX.AU26]); b.push(cam[Math.min(tk.T - 1, i - LA) * KM + IDX.AU26]); } const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } jawCorr = sab / Math.sqrt(saa * sbb); }
console.log(JSON.stringify({ take: tk.name, camera: useCam, passes, frames: N, lookaheadFrames: LA, msPerFrame: +ms.toFixed(2), worstMsPerFrame: +worst.toFixed(2), nonFinite: bad, maxHeadStepDeg: +step.toFixed(2), jawClosedInSilence: +(closed / Math.max(1, nsil)).toFixed(3), silenceFrames: nsil,
  headPitch: range('headPitch'), headYaw: range('headYaw'), headRoll: range('headRoll'), torsoLean: range('torsoLean'), jawVsCameraJaw: jawCorr === null ? null : +jawCorr.toFixed(3), lines: rows }));
fs.writeFileSync(base + (useCam ? '.livecam.f32' : '.live.f32'), Buffer.concat(frames.map((f) => Buffer.from(f.y.buffer))));
