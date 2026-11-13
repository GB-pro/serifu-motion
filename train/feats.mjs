// Compute normalized audio features for all CREMA-D clips with the app's own extractor.
import fs from 'fs'; import path from 'path';
import { extractRaw, speakerStats, normalize, FEAT_DIM } from '../app/core/features.js';
// Node may hand back small files as views into a shared pool: copy exactly the file's bytes.
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const dir = process.argv[2];
export function readWav(f) {
  const b = fs.readFileSync(f); let o = 12, data = null, sr = 16000, ch = 1;
  while (o < b.length) { const id = b.toString('ascii', o, o + 4), sz = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); }
    if (id === 'data') { data = b.subarray(o + 8, o + 8 + sz); break; } o += 8 + sz; }
  const n = data.length / 2 / ch, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2 * ch) / 32768; return x;
}
const wavs = fs.readdirSync(dir).filter(f => f.endsWith('.wav')).sort();
const byActor = {};
for (const w of wavs) (byActor[w.slice(0, 4)] ||= []).push(w);
let n = 0;
for (const [actor, list] of Object.entries(byActor)) {
  const raws = list.map(w => { const out = path.join(dir, w.replace('.wav', '.raw.f32'));
    if (fs.existsSync(out)) { const a = readF32(out); return { raw: a, nFrames: a.length / 19 }; }
    const r = extractRaw(readWav(path.join(dir, w))); fs.writeFileSync(out, Buffer.from(r.raw.buffer)); return r; });
  const st = speakerStats(raws);
  fs.writeFileSync(path.join(dir, actor + '.stats.json'), JSON.stringify(st));
  list.forEach((w, i) => { const F = normalize(raws[i].raw, raws[i].nFrames, st); fs.writeFileSync(path.join(dir, w.replace('.wav', '.feat.f32')), Buffer.from(F.buffer)); });
  n += list.length; console.log(actor, list.length, n, 'f0med', (2 ** (st.f0Median / 12)).toFixed(0));
}
