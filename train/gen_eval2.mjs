// Generate rig-space emo for every clip of the held-out actors with the app's own generator, three ways:
//   lin  = linear rendition of the learned motion (the first version of the app)
//   auto = animation layer driven by the automatic reading
//   fix  = animation layer with the acted emotion given (what a user gets after correcting a misread line)
// Each actor's clips are joined into one session with silence between them, like a real recording.
// node gen_eval2.mjs <model[,model...]> <featDir> <outDir> <actors,csv> [avatar]
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM } from '../app/core/features.js';
import { K, EMOS } from '../app/core/au.js';
import { analyzeSpeech } from '../app/core/animate.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelArg, featDir, outDir, actorsArg, avatar = 'hana'] = process.argv.slice(2);
const models = modelArg.split(',').map((m) => JSON.parse(fs.readFileSync(m)));
const gen = new EmoGenerator(models[0], null, models.slice(1));
const profile = JSON.parse(fs.readFileSync(new URL('../app/avatar/acting.json', import.meta.url)))[avatar] || null;
fs.mkdirSync(outDir, { recursive: true });
const index = [];
for (const a of actorsArg.split(',')) {
  const clips = fs.readdirSync(featDir).filter((f) => f.startsWith(a + '_') && f.endsWith('.feat.f32')).sort();
  const parts = clips.map((f) => readF32(path.join(featDir, f))).map((p) => p.subarray(0, Math.floor(p.length / FEAT_DIM) * FEAT_DIM));
  const GAP = 24; let T = 0; const offs = [];
  for (const p of parts) { offs.push(T + GAP); T += GAP + p.length / FEAT_DIM; } T += GAP;
  const F = new Float32Array(T * FEAT_DIM); for (let t = 0; t < T; t++) F[t * FEAT_DIM] = -4;
  parts.forEach((p, i) => F.set(p, offs[i] * FEAT_DIM));
  const net = gen.runNet(F, T), seed = +a;
  const an = analyzeSpeech(F, T, net.aux);
  const overrides = clips.map((f, i) => ({ start: (offs[i] - 4) / 30, end: (offs[i] + parts[i].length / FEAT_DIM + 4) / 30, emo: f.split('_')[2] }));
  const outs = { lin: gen.direct(net, F, T, { seed, style: 'natural' }), auto: gen.direct(net, F, T, { seed, style: 'anime', profile }), fix: gen.direct(net, F, T, { seed, style: 'anime', profile, overrides }) };
  // the whole session too (continuous motion across lines, for the head-motion statistics)
  for (const k in outs) fs.writeFileSync(path.join(outDir, `session_${a}.${k}.f32`), Buffer.from(outs[k].base.buffer, outs[k].base.byteOffset, outs[k].base.byteLength));
  fs.writeFileSync(path.join(outDir, `session_${a}.json`), JSON.stringify({ T, speech: Array.from(outs.auto.voiced), phrases: outs.auto.plan.phrases.map((ph) => ({ t0: ph.t0, t1: ph.t1, tp: ph.tp, emo: ph.emo })) }));
  clips.forEach((f, i) => {
    const n = parts[i].length / FEAT_DIM, name = f.replace('.feat.f32', ''), f0 = offs[i], f1 = offs[i] + n;
    for (const k in outs) fs.writeFileSync(path.join(outDir, `${name}.${k}.f32`), Buffer.from(outs[k].base.slice(f0 * K, f1 * K).buffer));
    const info = { name, n, nuclei: an.phrases.flatMap((ph) => ph.nuclei.filter((u) => u.t >= f0 && u.t < f1).map((u) => ({ t: u.t - f0, score: +u.score.toFixed(3) }))) };
    if (net.aux) fs.writeFileSync(path.join(outDir, `${name}.aux.f32`), Buffer.from(net.aux.slice(f0 * 3, f1 * 3).buffer));
    for (const k of ['auto', 'fix']) {
      const phs = outs[k].plan.phrases.filter((ph) => Math.min(ph.t1, f1) > Math.max(ph.t0, f0));
      const p = new Array(EMOS.length).fill(0); let w = 0; for (const ph of phs) { const d = Math.min(ph.t1, f1) - Math.max(ph.t0, f0); for (let e = 0; e < p.length; e++) p[e] += ph.p[e] * d; w += d; }
      let best = 0; for (let e = 1; e < p.length; e++) if (p[e] > p[best]) best = e;
      info[k] = { read: w ? EMOS[best] : null, phrases: phs.map((ph) => ({ t0: ph.t0 - f0, t1: ph.t1 - f0, tp: ph.tp - f0, emo: ph.emo, emo2: ph.emo2, inten: ph.inten, accents: ph.accents.map((x) => ({ t: x.t - f0, kind: x.kind, key: x.key })) })) };
    }
    index.push(info);
  });
  console.log(a, clips.length, 'clips', T, 'frames');
}
fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify(index));
