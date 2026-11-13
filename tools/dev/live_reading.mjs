// How well can the causal (realtime) network read the feeling of a line after hearing only its first n speech frames?
// node tools/dev/live_reading.mjs <realtime.json> <featDir> <actors,csv>
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../../app/core/generator.js';
import { FEAT_DIM } from '../../app/core/features.js';
import { analyzeSpeech } from '../../app/core/animate.js';
import { softmax } from '../../app/core/util.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelPath, featDir, actorsArg] = process.argv.slice(2);
const gen = new EmoGenerator(JSON.parse(fs.readFileSync(modelPath)), null, []);
const EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS'], NS = [5, 10, 15, 20, 30, 45, 60, 999];
const ok = NS.map(() => 0), okFrame = NS.map(() => 0); let N = 0;
for (const a of actorsArg.split(',')) for (const f of fs.readdirSync(featDir).filter((f) => f.startsWith(a + '_') && f.endsWith('.feat.f32')).sort()) {
  const F = readF32(path.join(featDir, f)); if (F.length % FEAT_DIM) continue; const T = F.length / FEAT_DIM; if (T < 20) continue;
  const { logits, aux } = gen.runNet(F, T); const an = analyzeSpeech(F, T, aux); const truth = EMOS.indexOf(f.split('_')[2]);
  const p = new Array(6).fill(0); let n = 0, i = 0; N++; let last = null;
  for (let t = 0; t < T; t++) {
    if (!an.speech[t]) continue; const q = softmax(Array.from(logits.subarray(t * 6, t * 6 + 6))); for (let e = 0; e < 6; e++) p[e] += q[e]; n++; last = q;
    while (i < NS.length && n === NS[i]) { let b = 0, bq = 0; for (let e = 1; e < 6; e++) { if (p[e] > p[b]) b = e; if (q[e] > q[bq]) bq = e; } if (b === truth) ok[i]++; if (bq === truth) okFrame[i]++; i++; }
  }
  for (; i < NS.length; i++) { let b = 0, bq = 0; for (let e = 1; e < 6; e++) { if (p[e] > p[b]) b = e; if (last && last[e] > last[bq]) bq = e; } if (b === truth) ok[i]++; if (bq === truth) okFrame[i]++; }
}
console.log('clips', N); NS.forEach((n, i) => console.log(`after ${n === 999 ? 'the whole line' : n + ' speech frames'}: mean of what was heard so far ${(ok[i] / N).toFixed(3)} | the single latest frame ${(okFrame[i] / N).toFixed(3)}`));
