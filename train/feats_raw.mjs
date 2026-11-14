// Raw audio frames (app/core/features.js: extractRaw) for any folder of 16-bit PCM wav files, one .raw.f32 per file.
// Used for the corpora other than CREMA-D, where a recording is not "all lines of an actor": the normalisation is
// then done in Python (simnorm.py) for whatever recording the lines are put into.
//   node train/feats_raw.mjs <wavDir> <outDir> [shard] [shards]      (run several shards side by side)
import fs from 'fs'; import path from 'path';
import { extractRaw, SR } from '../app/core/features.js';
import { readWav } from './wav.mjs';

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
