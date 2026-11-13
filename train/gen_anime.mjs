// Generate rig-space tracks for single clips: anime interpretation and the linear ("natural") rendition, same seed.
// node gen_anime.mjs <model.json> <featDir> <outDir> <clip,clip,...> [avatar]
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM } from '../app/core/features.js';
import { K } from '../app/core/au.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelPath, featDir, outDir, clipsArg, avatar = '', force = ''] = process.argv.slice(2); // force: 'label' (acted label of the clip) or an emotion code
const gen = new EmoGenerator(JSON.parse(fs.readFileSync(modelPath)));
let profile = null;
if (avatar) { const prof = JSON.parse(fs.readFileSync(new URL('../app/avatar/acting.json', import.meta.url))); profile = prof[avatar] || null; }
fs.mkdirSync(outDir, { recursive: true });
for (const name of clipsArg.split(',')) {
  const F = readF32(path.join(featDir, name + '.feat.f32')), T = F.length / FEAT_DIM;
  const seed = [...name].reduce((s, c) => (s * 31 + c.charCodeAt(0)) >>> 0, 7);
  const fe = force === 'label' ? name.split('_')[2] : force;
  const a = gen.generate(F, T, { seed, style: 'anime', profile, overrides: fe ? [{ start: 0, end: 1e9, emo: fe, intensity: 0.9 }] : [] });
  const l = gen.generate(F, T, { seed, style: 'natural' });
  fs.writeFileSync(path.join(outDir, name + (fe ? '.fix' : '.anime') + '.f32'), Buffer.from(a.base.buffer));
  fs.writeFileSync(path.join(outDir, name + '.linear.f32'), Buffer.from(l.base.buffer));
  fs.writeFileSync(path.join(outDir, name + '.plan.json'), JSON.stringify(a.plan));
  console.log(name, 'T', T, JSON.stringify(a.plan.phrases.map(p => ({ t: [p.t0, p.t1], emo: p.emo + (p.emo2 ? '+' + p.emo2 + p.w2 : ''), conf: p.conf, I: p.inten, pose: p.pose, q: p.question, acc: p.accents.map(x => x.t + ':' + x.kind + (x.key ? '*' : '')) }))));
}
