// Clip-level emotion reading of the network on held-out actors: accuracy, confusion, and how confidence relates to errors.
// node eval_emotion.mjs <model.json> <featDir> <actors,csv>
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM } from '../app/core/features.js';
import { analyzeSpeech } from '../app/core/animate.js';
import { softmax } from '../app/core/util.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelPath, featDir, actorsArg, outJson] = process.argv.slice(2);
const models = modelPath.split(',').map((m) => JSON.parse(fs.readFileSync(m)));
const gen = new EmoGenerator(models[0], null, models.slice(1)); // several models = committee
const EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS'];
const conf = EMOS.map(() => EMOS.map(() => 0)); const rows = [];
for (const a of actorsArg.split(',')) {
  for (const f of fs.readdirSync(featDir).filter(f => f.startsWith(a + '_') && f.endsWith('.feat.f32')).sort()) {
    const F = readF32(path.join(featDir, f)); if (F.length % FEAT_DIM) continue; const T = F.length / FEAT_DIM; if (T < 20) continue;
    const { logits, aux } = gen.runNet(F, T); const an = analyzeSpeech(F, T, aux);
    const p = new Array(6).fill(0); let n = 0;
    for (let t = 0; t < T; t++) { if (!an.speech[t]) continue; const q = softmax(Array.from(logits.subarray(t * 6, t * 6 + 6))); for (let e = 0; e < 6; e++) p[e] += q[e]; n++; }
    if (!n) continue; for (let e = 0; e < 6; e++) p[e] /= n;
    const truth = EMOS.indexOf(f.split('_')[2]); let best = 0; for (let e = 1; e < 6; e++) if (p[e] > p[best]) best = e;
    const s = p.slice().sort((x, y) => y - x);
    conf[truth][best]++; rows.push({ name: f.replace('.feat.f32', ''), truth, best, conf: s[0], margin: s[0] - s[1], p, lvl: f.split('_')[3].slice(0, 2), inten: aux ? (() => { let v = 0, c = 0; for (let t = 0; t < T; t++) if (an.speech[t]) { v += aux[t * 3 + 2]; c++; } return v / c; })() : 0 });
  }
}
const N = rows.length, acc = rows.filter(r => r.truth === r.best).length / N;
console.log('clips', N, 'accuracy', acc.toFixed(3));
console.log('confusion (rows = acted, cols = read) ' + EMOS.join(' '));
conf.forEach((r, i) => console.log(EMOS[i], r.map(v => String(v).padStart(4)).join(' '), ' recall', (r[i] / r.reduce((a, b) => a + b, 0)).toFixed(2)));
for (const thr of [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]) { const hi = rows.filter(r => r.conf >= thr); console.log('conf >=', thr, 'share', (hi.length / N).toFixed(2), 'accuracy', (hi.filter(r => r.truth === r.best).length / Math.max(1, hi.length)).toFixed(3)); }
// policy: leave neutral only with evidence
for (const [c, m] of [[0.4, 0.08], [0.42, 0.1], [0.45, 0.12], [0.5, 0.15]]) {
  let ok = 0, harmful = 0; for (const r of rows) { const pick = r.best !== 0 && (r.conf < c || r.margin < m) ? 0 : r.best; if (pick === r.truth) ok++; else if (pick !== 0) harmful++; }
  console.log('fallback to NEU when conf <', c, 'or margin <', m, ': accuracy', (ok / N).toFixed(3), 'wrong non-neutral reading', (harmful / N).toFixed(3));
}
{ let harmful = 0; for (const r of rows) if (r.best !== r.truth && r.best !== 0) harmful++; console.log('no fallback: wrong non-neutral reading', (harmful / N).toFixed(3)); }
const lv = {}; for (const r of rows) { (lv[r.lvl] = lv[r.lvl] || []).push(r.inten); } for (const k in lv) console.log('intensity output for level', k, (lv[k].reduce((a, b) => a + b, 0) / lv[k].length).toFixed(3), 'n', lv[k].length);
if (outJson) fs.writeFileSync(outJson, JSON.stringify({ N, acc, conf, rows }));
