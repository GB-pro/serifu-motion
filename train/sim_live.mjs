// Live sessions for training and tuning the realtime network, without the microphone.
// The lines of one actor are replayed, raw frame by raw frame, through the app's own StreamingFeatures (pushRaw): the
// statistics of the voice start from the defaults and settle as the session goes on, exactly as in live mode. The
// realtime network is then trained on what live mode really sees at each moment of a session, instead of on features
// normalised with statistics of the whole recording, which live mode never has.
// node train/sim_live.mjs <featDir> <outDir> <actors,csv> <train|eval> [seed=1]
//   train: 10 sessions per actor of 6 to 82 lines in random order, half of them with a skewed mix of emotions,
//          random pauses and a random recording level
//   eval:  one session per actor, every line in the order and with the pauses of eval_live.mjs
// Writes <outDir>/<actor>_<k>.f32 (frames x (20 features + 30 recording-level inputs)) and .json (where each line is).
import fs from 'fs'; import path from 'path';
import { StreamingFeatures, RAW_DIM, FEAT_DIM, NG } from '../app/core/features.js';
import { rng } from '../app/core/util.js';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const [featDir, outDir, actorsArg, kind = 'train', seedArg = '1'] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS'], D = FEAT_DIM + NG, ALL = Array.from({ length: NG }, (_, i) => i);
const SILENCE = new Float32Array(RAW_DIM); SILENCE[0] = -100; SILENCE[2] = 1; // what the extractor gives for digital silence
for (const a of actorsArg.split(',')) {
  const r = rng((+a) * 7919 + (+seedArg));
  const clips = fs.readdirSync(featDir).filter((f) => f.startsWith(a + '_') && f.endsWith('.raw.f32')).sort().map((f) => { const raw = readF32(path.join(featDir, f)); return { name: f.replace('.raw.f32', ''), raw, n: Math.floor(raw.length / RAW_DIM), emo: EMOS.indexOf(f.split('_')[2]) }; });
  const sessions = [];
  if (kind === 'eval') sessions.push({ lines: clips, gap: () => 24, gain: 0, lead: 0 });
  else for (const [k, L] of [82, 82, 48, 48, 24, 24, 12, 12, 6, 6].entries()) {
    // odd sessions lean towards some emotions (a cheerful stream, an angry scene): weights like a Dirichlet(0.4) draw
    const w = EMOS.map(() => (k % 2 ? Math.pow(-Math.log(1 - r()), 2.5) + 0.02 : 1));
    const pool = clips.slice(), lines = [];
    while (lines.length < L && pool.length) { let s = 0; for (const c of pool) s += w[c.emo]; let u = r() * s, i = 0; for (; i < pool.length - 1; i++) { u -= w[pool[i].emo]; if (u <= 0) break; } lines.push(pool.splice(i, 1)[0]); }
    sessions.push({ lines, gap: () => (r() < 0.1 ? 60 + Math.floor(r() * 40) : 10 + Math.floor(r() * 36)), gain: (r() * 2 - 1) * 8, lead: Math.floor(r() * 30) });
  }
  sessions.forEach((s, k) => {
    const sf = new StreamingFeatures({ globals: ALL }), frames = [], marks = [];
    const push = (vals) => frames.push(sf.pushRaw(vals));
    for (let i = 0; i < s.lead; i++) push(SILENCE);
    for (const c of s.lines) {
      marks.push({ name: c.name, t0: frames.length, n: c.n });
      for (let i = 0; i < c.n; i++) { const v = c.raw.slice(i * RAW_DIM, (i + 1) * RAW_DIM); v[0] += s.gain; push(v); }
      for (let g = s.gap(), i = 0; i < g; i++) push(SILENCE);
    }
    const X = new Float32Array(frames.length * D); frames.forEach((f, i) => X.set(f, i * D));
    fs.writeFileSync(path.join(outDir, `${a}_${k}.f32`), Buffer.from(X.buffer));
    fs.writeFileSync(path.join(outDir, `${a}_${k}.json`), JSON.stringify({ actor: +a, T: frames.length, dim: D, gain: +s.gain.toFixed(2), lines: marks }));
  });
  console.log(a, sessions.length, 'sessions');
}
