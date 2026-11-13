// Fit an additive bias on the emotion logits with the validation actors (class-prior calibration), check on held-out actors.
// node calib_emotion.mjs <model.json> <featDir> <valActors> <testActors>
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { FEAT_DIM } from '../app/core/features.js';
import { analyzeSpeech } from '../app/core/animate.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [modelPath, featDir, valArg, testArg] = process.argv.slice(2);
const model = JSON.parse(fs.readFileSync(modelPath));
const gen = new EmoGenerator(model);
const EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS'];
function load(actors) {
  const out = [];
  for (const a of actors.split(',')) for (const f of fs.readdirSync(featDir).filter(f => f.startsWith(a + '_') && f.endsWith('.feat.f32')).sort()) {
    const F = readF32(path.join(featDir, f)); if (F.length % FEAT_DIM) continue; const T = F.length / FEAT_DIM; if (T < 20) continue;
    const { logits, aux } = gen.runNet(F, T); const an = analyzeSpeech(F, T, aux);
    const L = []; for (let t = 0; t < T; t++) if (an.speech[t]) L.push(Array.from(logits.subarray(t * 6, t * 6 + 6)));
    if (L.length) out.push({ L, y: EMOS.indexOf(f.split('_')[2]) });
  }
  return out;
}
function post(c, b) { const p = new Array(6).fill(0); for (const l of c.L) { let mx = -1e9; const z = l.map((v, e) => { const u = v + b[e]; if (u > mx) mx = u; return u; }); let s = 0; const ex = z.map(v => { const q = Math.exp(v - mx); s += q; return q; }); for (let e = 0; e < 6; e++) p[e] += ex[e] / s; } return p.map(v => v / c.L.length); }
function score(set, b) { let nll = 0, ok = 0; const cm = EMOS.map(() => EMOS.map(() => 0)); for (const c of set) { const p = post(c, b); nll -= Math.log(p[c.y] + 1e-9); let best = 0; for (let e = 1; e < 6; e++) if (p[e] > p[best]) best = e; if (best === c.y) ok++; cm[c.y][best]++; } return { nll: nll / set.length, acc: ok / set.length, cm }; }
const val = load(valArg), test = load(testArg);
let b = new Array(6).fill(0), best = score(val, b).nll;
for (const step of [0.6, 0.3, 0.15, 0.08, 0.04]) for (let it = 0; it < 6; it++) for (let e = 0; e < 6; e++) for (const d of [step, -step]) { const nb = b.slice(); nb[e] += d; const s = score(val, nb).nll; if (s < best - 1e-5) { best = s; b = nb; } }
const m = b.reduce((a, c) => a + c, 0) / 6; b = b.map(v => +(v - m).toFixed(3));
const z = new Array(6).fill(0);
console.log('bias', JSON.stringify(b));
for (const [n, set] of [['val', val], ['test', test]]) { const a = score(set, z), c = score(set, b); console.log(n, 'clips', set.length, 'acc', a.acc.toFixed(3), '->', c.acc.toFixed(3), 'nll', a.nll.toFixed(3), '->', c.nll.toFixed(3)); if (n === 'test') c.cm.forEach((r, i) => console.log(EMOS[i], r.map(v => String(v).padStart(4)).join(' '), ' recall', (r[i] / r.reduce((x, y) => x + y, 0)).toFixed(2))); }
if (process.argv[6] === 'write') { model.meta.emo_bias = b; fs.writeFileSync(modelPath, JSON.stringify(model)); console.log('written to', modelPath); }
