// Generate emo for held-out actors with the app's own generator (per-speaker track, clips joined by silence).
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM } from '../app/core/features.js';
// Node may hand back small files as views into a shared pool: copy exactly the file's bytes.
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelPath, featDir, outDir, actorsArg, mode = 'full'] = process.argv.slice(2);
const gen = new EmoGenerator(JSON.parse(fs.readFileSync(modelPath)));
const K = gen.meta.names.length;
const actors = actorsArg.split(',');
fs.mkdirSync(outDir, { recursive: true });
const opts = mode === 'net' ? { noise: 0, blinks: false, saccades: false, gain: false } : mode === 'gain' ? { noise: 0, blinks: false, saccades: false } : {};
for (const a of actors) {
  const clips = fs.readdirSync(featDir).filter(f => f.startsWith(a + '_') && f.endsWith('.feat.f32')).sort();
  const parts = clips.map(f => readF32(path.join(featDir, f)));
  const GAP = 24; let T = 0; const offs = [];
  for (const p of parts) { offs.push(T + GAP); T += GAP + p.length / FEAT_DIM; } T += GAP;
  const F = new Float32Array(T * FEAT_DIM); for (let t = 0; t < T; t++) F[t * FEAT_DIM] = -4;
  parts.forEach((p, i) => F.set(p, offs[i] * FEAT_DIM));
  const out = gen.generate(F, T, { seed: +a, ...opts });
  clips.forEach((f, i) => {
    const n = parts[i].length / FEAT_DIM;
    fs.writeFileSync(path.join(outDir, f.replace('.feat.f32', '.gen.f32')), Buffer.from(out.base.slice(offs[i] * K, (offs[i] + n) * K).buffer));
    fs.writeFileSync(path.join(outDir, f.replace('.feat.f32', '.emo.f32')), Buffer.from(out.emo.slice(offs[i] * 6, (offs[i] + n) * 6).buffer));
  });
  console.log(a, clips.length);
}
