// The lines of a set of recordings as the app's reader gets them (app/core/reader.js), and for every line what the
// speech encoder makes of it: mean and standard deviation over time of the output of each block.
//   node train/embed_lines.mjs <encoder.onnx> <manifest.tsv> <out prefix> [shard] [shards]
// manifest, one file per row: name, corpus, speaker, emotion (0..5), wav, raw frames (feats_raw.mjs), tab separated.
// Each file is treated as a recording of its own: its phrases (animate.js: analyzeSpeech) are grouped into lines.
// Writes <out prefix>.<shard>.tsv (one row per line: name, corpus, speaker, emotion, index of the line in the file,
// first and last frame, speech level in dB, number of encoder steps) and <out prefix>.<shard>.f32 (per line:
// blocks x width means).
import fs from 'fs';
import ort from 'onnxruntime-node';
import { RAW_DIM, speakerStats, normalize } from '../app/core/features.js';
import { analyzeSpeech } from '../app/core/animate.js';
import { logMel, linesOf, lineSamples } from '../app/core/reader.js';
import { readWav } from './wav.mjs';

const [model, manifest, out, shard = '0', shards = '1'] = process.argv.slice(2);
const sess = await ort.InferenceSession.create(model, { intraOpNumThreads: 1, interOpNumThreads: 1 });
const rows = fs.readFileSync(manifest, 'utf8').split('\n').filter(Boolean).map((l) => l.split('\t')).filter((_, i) => i % +shards === +shard);
const fd = fs.openSync(`${out}.${shard}.f32`, 'w'), idx = []; let bad = 0, none = 0; const t0 = Date.now();
for (let k = 0; k < rows.length; k++) {
  const [name, corpus, speaker, emo, wav, rawPath] = rows[k];
  try {
    const x = readWav(wav), b = fs.readFileSync(rawPath), raw = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)), T = Math.floor(raw.length / RAW_DIM);
    const st = speakerStats([{ raw, nFrames: T }]), F = normalize(raw, T, st), lines = linesOf(analyzeSpeech(F, T).phrases);
    if (!lines.length) none++;
    for (let i = 0; i < lines.length; i++) {
      const { samples, level } = lineSamples(x, raw, lines[i]), m = logMel(samples);
      const res = await sess.run({ mel: new ort.Tensor('float32', m.data, [1, 80, m.frames]) });
      const hs = sess.outputNames.map((n) => res[n]), n = hs[0].dims[1], D = hs[0].dims[2], v = new Float32Array(hs.length * D);
      hs.forEach((h, l) => {
        const d = h.data;
        for (let j = 0; j < D; j++) { let s = 0; for (let t = 0; t < n; t++) s += d[t * D + j]; v[l * D + j] = s / n; }
      });
      fs.writeSync(fd, Buffer.from(v.buffer)); idx.push([name, corpus, speaker, emo, i, lines[i].t0, lines[i].t1, level.toFixed(2), n].join('\t'));
    }
  } catch (e) { bad++; console.error(name, String(e.message || e)); }
  if ((k + 1) % 500 === 0) console.log(`shard ${shard}: ${k + 1}/${rows.length} files, ${idx.length} lines in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
fs.closeSync(fd); fs.writeFileSync(`${out}.${shard}.tsv`, idx.join('\n') + '\n');
console.log(`shard ${shard}: done ${rows.length} files, ${idx.length} lines, ${none} files without a phrase, failed ${bad}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
