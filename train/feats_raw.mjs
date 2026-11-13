// Raw audio frames (app/core/features.js: extractRaw) for any folder of 16-bit PCM wav files, one .raw.f32 per file.
// Used for the corpora other than CREMA-D, where a recording is not "all lines of an actor": the normalisation is
// then done in Python (simnorm.py) for whatever recording the lines are put into.
//   node train/feats_raw.mjs <wavDir> <outDir> [shard] [shards]      (run several shards side by side)
import fs from 'fs'; import path from 'path';
import { extractRaw, resampleTo16k, SR } from '../app/core/features.js';

function readWav(f) {
  const b = fs.readFileSync(f); let o = 12, data = null, sr = SR, ch = 1, bits = 16, fmt = 1;
  while (o + 8 <= b.length) { const id = b.toString('ascii', o, o + 4), sz = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { fmt = b.readUInt16LE(o + 8); ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); bits = b.readUInt16LE(o + 22); }
    if (id === 'data') { data = b.subarray(o + 8, Math.min(b.length, o + 8 + sz)); break; } o += 8 + sz + (sz & 1); }
  if (!data || (fmt !== 1 && fmt !== 0xfffe) || bits !== 16) throw new Error(`unsupported wav (format ${fmt}, ${bits} bit)`);
  const n = Math.floor(data.length / 2 / ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < ch; c++) s += data.readInt16LE((i * ch + c) * 2); x[i] = s / ch / 32768; }
  return sr === SR ? x : resampleTo16k(x, sr);
}

const [wavDir, outDir, shard = '0', shards = '1'] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const wavs = fs.readdirSync(wavDir).filter(f => f.endsWith('.wav')).sort().filter((_, i) => i % +shards === +shard);
let done = 0, bad = 0, secs = 0; const t0 = Date.now();
for (const w of wavs) {
  const out = path.join(outDir, w.slice(0, -4) + '.raw.f32');
  if (fs.existsSync(out)) { done++; continue; }
  try { const x = readWav(path.join(wavDir, w)); const r = extractRaw(x); fs.writeFileSync(out, Buffer.from(r.raw.buffer, r.raw.byteOffset, r.raw.byteLength)); secs += x.length / SR; done++; }
  catch (e) { bad++; console.error(w, String(e.message || e)); }
  if (done % 500 === 0) console.log(`shard ${shard}: ${done}/${wavs.length} files, ${(secs / 3600).toFixed(2)} h of audio in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
console.log(`shard ${shard}: done ${done}, failed ${bad}, ${(secs / 3600).toFixed(2)} h of audio in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
