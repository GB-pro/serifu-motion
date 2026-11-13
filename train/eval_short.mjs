// How well is the emotion read when the recording is short? The app knows a speaker only from the recording it is
// given: pitch, level and timbre are measured relative to that recording's own statistics. The main evaluation joins
// all 82 lines of an actor (every emotion equally often), which is the easy case. This script re-reads the same
// held-out lines in smaller recordings, each normalised by its own statistics as the app would do.
// node train/eval_short.mjs <model[,model...]> <featDir> <actors,csv>
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM, RAW_DIM, speakerStats, normalize } from '../app/core/features.js';
import { EMOS } from '../app/core/au.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelArg, featDir, actorsArg] = process.argv.slice(2);
const models = modelArg.split(',').map((m) => JSON.parse(fs.readFileSync(m)));
const gen = new EmoGenerator(models[0], null, models.slice(1));
const GAP = 24;
function readSession(clips, seed) { // clips: [{name, raw, nFrames, emo}] -> emotion read for each line
  const st = speakerStats(clips);
  const parts = clips.map((c) => normalize(c.raw, c.nFrames, st));
  let T = 0; const offs = []; for (const c of clips) { offs.push(T + GAP); T += GAP + c.nFrames; } T += GAP;
  const F = new Float32Array(T * FEAT_DIM); for (let t = 0; t < T; t++) F[t * FEAT_DIM] = -4;
  parts.forEach((p, i) => F.set(p, offs[i] * FEAT_DIM));
  const net = gen.runNet(F, T), out = gen.direct(net, F, T, { seed, style: 'anime' });
  return clips.map((c, i) => {
    const f0 = offs[i], f1 = offs[i] + c.nFrames, p = new Array(EMOS.length).fill(0); let w = 0;
    for (const ph of out.plan.phrases) { const d = Math.min(ph.t1, f1) - Math.max(ph.t0, f0); if (d > 0) { for (let e = 0; e < p.length; e++) p[e] += ph.p[e] * d; w += d; } }
    let best = 0; for (let e = 1; e < p.length; e++) if (p[e] > p[best]) best = e;
    return w ? EMOS[best] : null;
  });
}
const tally = {}; const add = (k, ok, emo) => { const t = (tally[k] ||= { n: 0, ok: 0, per: {} }); t.n++; t.ok += ok ? 1 : 0; const q = (t.per[emo] ||= [0, 0]); q[0]++; q[1] += ok ? 1 : 0; };
for (const a of actorsArg.split(',')) {
  const clips = fs.readdirSync(featDir).filter((f) => f.startsWith(a + '_') && f.endsWith('.raw.f32')).sort().map((f) => { const raw = readF32(path.join(featDir, f)); return { name: f.replace('.raw.f32', ''), raw, nFrames: Math.floor(raw.length / RAW_DIM), emo: f.split('_')[2] }; });
  const run = (key, groups) => { for (const g of groups) { if (!g.length) continue; const r = readSession(g, +a); g.forEach((c, i) => add(key, r[i] === c.emo, c.emo)); } };
  run('all 82 lines of the speaker (as in the main evaluation)', [clips]);
  const bySent = {}; for (const c of clips) (bySent[c.name.split('_')[1]] ||= []).push(c);
  run('about 6 lines, one per emotion', Object.values(bySent));
  const three = []; for (const g of Object.values(bySent)) { for (let i = 0; i < g.length; i += 3) three.push(g.slice(i, i + 3)); }
  run('3 lines', three);
  run('1 line on its own', clips.map((c) => [c]));
  const byEmo = {}; for (const c of clips) (byEmo[c.emo] ||= []).push(c);
  run('about 13 lines, all in the same emotion', Object.values(byEmo));
  console.error(a, 'done');
}
for (const k in tally) { const t = tally[k]; console.log(k.padEnd(58), (t.ok / t.n).toFixed(3), 'n', t.n, ' ', EMOS.map((e) => e + ' ' + (t.per[e] ? (t.per[e][1] / t.per[e][0]).toFixed(2) : 'n/a')).join('  ')); }
