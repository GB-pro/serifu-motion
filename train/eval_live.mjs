// Live mode, measured without a phone: every line of the given actors is streamed through the realtime path
// (microphone-sized chunks -> streaming features -> causal network -> live director), lines joined into one session.
// Reports which feeling is being PLAYED at 0.5 s and 1 s into each line and at its end, plus artifacts.
// node train/eval_live.mjs <realtime.json[,more.json: committee]> <wavDir> <actors,csv> [avatar] [switch margins a,b]
import fs from 'fs'; import path from 'path';
import { LiveEmo } from '../app/core/generator.js';
import { IDX, K, EMOS } from '../app/core/au.js';
import { SR, FPS } from '../app/core/features.js';
const [modelPath, wavDir, actorsArg, avatar = 'hana', marginsArg = ''] = process.argv.slice(2);
const switchMargins = marginsArg ? marginsArg.split(',').map(Number) : undefined; // tuning runs only
const [rt, ...extra] = modelPath.split(',').map((m) => JSON.parse(fs.readFileSync(m)));
const profile = JSON.parse(fs.readFileSync(new URL('../app/avatar/acting.json', import.meta.url)))[avatar] || null;
function readWav(p) { const b = fs.readFileSync(p); let o = 12; while (o < b.length - 8) { const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4); if (id === 'data') { const x = new Float32Array(n >> 1); for (let i = 0; i < x.length; i++) x[i] = b.readInt16LE(o + 8 + i * 2) / 32768; return x; } o += 8 + n + (n & 1); } throw new Error('no data: ' + p); }
const hop = SR / FPS, GAP = Math.round(0.8 * SR / hop) * hop;
const at = { early: [0, 0], mid: [0, 0], end: [0, 0] }, heard = { early: [0, 0], mid: [0, 0], end: [0, 0] }, warm = { end: [0, 0] }; let step = 0, closed = 0, nsil = 0, lines = 0, frames = 0, bad = 0; const conf = EMOS.map(() => EMOS.map(() => 0));
const hi = ['headPitch', 'headYaw', 'headRoll'].map((n) => IDX[n]);
for (const a of actorsArg.split(',')) {
  const clips = fs.readdirSync(wavDir).filter((f) => f.startsWith(a + '_') && f.endsWith('.wav')).sort();
  const live = new LiveEmo(rt, { seed: +a, style: 'anime', profile, switchMargins, extra }), LA = live.meta.lookahead || 0;
  // diagnosis only: what if the speaker's voice statistics were known exactly from the start?
  if (process.env.ORACLE_STATS) { live.feat.stats = JSON.parse(fs.readFileSync(path.join(wavDir, a + '.stats.json'))); live.feat._updateStats = () => {}; live.feat.calibration = 1; }
  const out = []; const marks = []; let pos = 0;
  const feed = (x) => { for (let o = 0; o < x.length; o += 800) for (const f of live.push(x.subarray(o, Math.min(x.length, o + 800)))) out.push({ y: Float32Array.from(f.y), voiced: f.voiced, played: live.dir.emo, heard: EMOS[f.emo.indexOf(Math.max(...f.emo))], calib: live.feat.calibration }); };
  for (const c of clips) {
    const x = readWav(path.join(wavDir, c)); const n = Math.round(x.length / hop) * hop;
    marks.push({ emo: c.split('_')[2], f0: Math.round(pos / hop) + LA, f1: Math.round((pos + n) / hop) + LA });
    feed(x.subarray(0, Math.min(x.length, n))); if (x.length < n) feed(new Float32Array(n - x.length)); feed(new Float32Array(GAP)); pos += n + GAP;
  }
  const N = out.length; frames += N;
  for (let i = 0; i < N; i++) { for (let k = 0; k < K; k++) if (!Number.isFinite(out[i].y[k])) bad++; if (i) for (const k of hi) step = Math.max(step, Math.abs(out[i].y[k] - out[i - 1].y[k])); }
  for (let i = 0; i < N; i++) { let near = false; for (let j = -6; j <= 4; j++) if (out[i + j] && out[i + j].voiced) near = true; if (!near) { nsil++; if (out[i].y[IDX.AU26] < 0.12) closed++; } }
  for (const m of marks) {
    let s = -1, e = -1; for (let i = m.f0; i < Math.min(N, m.f1 + 6); i++) if (out[i].voiced) { if (s < 0) s = i; e = i; }
    if (s < 0) continue; lines++;
    const chk = (key, i) => { if (i <= e && i < N) { at[key][1]++; if (out[i].played === m.emo) at[key][0]++; heard[key][1]++; if (out[i].heard === m.emo) heard[key][0]++; } };
    chk('early', s + 15); chk('mid', s + 30); chk('end', e);
    if (marks.indexOf(m) >= 6) { warm.end[1]++; if (out[e].played === m.emo) warm.end[0]++; } conf[EMOS.indexOf(m.emo)][EMOS.indexOf(out[e].played)]++;
  }
}
const pct = (v) => (v[0] / Math.max(1, v[1])).toFixed(3);
console.log(JSON.stringify({ lines, frames, nonFinite: bad, played_right_after_0_5s: pct(at.early), after_1s: pct(at.mid), at_end_of_line: pct(at.end), at_end_after_the_first_6_lines: pct(warm.end), network_latest_answer_right: [pct(heard.early), pct(heard.mid), pct(heard.end)], maxHeadStepDeg: +step.toFixed(2), jawClosedInSilence: +(closed / Math.max(1, nsil)).toFixed(3) }));
console.log('played at the end of the line (rows = acted): ' + EMOS.join(' ')); conf.forEach((r, i) => console.log(EMOS[i], r.map((v) => String(v).padStart(4)).join(' ')));
