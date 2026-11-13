// Build one comparison "take": several recorded lines of one speaker joined into a monologue, then generated three ways
// with the app's own code path (features -> network -> linear rendition / animation layer).
// node tools/make_compare.mjs <model.json[,model2.json...]> <wavDir> <outDir> <name> <avatar> <clip1,clip2,...> [gapSec]
// Writes <name>.wav, <name>.linear.f32, <name>.auto.f32, <name>.fix.f32 (emotion given per line), <name>.json
import fs from 'fs'; import path from 'path';
import { EmoGenerator } from '../app/core/generator.js';
import { extractRaw, speakerStats, normalize, SR, FPS } from '../app/core/features.js';
import { K, EMOS } from '../app/core/au.js';
const [modelArg, wavDir, outDir, name, avatar, clipsArg, gapArg = '0.45'] = process.argv.slice(2);
function readWav(p) {
  const b = fs.readFileSync(p); let o = 12, fmt = null;
  while (o < b.length - 8) {
    const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = { ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) };
    if (id === 'data') { if (!fmt || fmt.sr !== SR || fmt.ch !== 1 || fmt.bits !== 16) throw new Error('expected 16 kHz mono 16-bit: ' + p); const x = new Float32Array(n / 2); for (let i = 0; i < x.length; i++) x[i] = b.readInt16LE(o + 8 + i * 2) / 32768; return x; }
    o += 8 + n + (n & 1);
  }
  throw new Error('no data chunk: ' + p);
}
function writeWav(p, x) {
  const b = Buffer.alloc(44 + x.length * 2); b.write('RIFF', 0); b.writeUInt32LE(36 + x.length * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(x.length * 2, 40);
  for (let i = 0; i < x.length; i++) b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x[i] * 32768))), 44 + i * 2);
  fs.writeFileSync(p, b);
}
const models = modelArg.split(',').map((m) => JSON.parse(fs.readFileSync(m)));
const gen = new EmoGenerator(models[0], null, models.slice(1));
const profile = JSON.parse(fs.readFileSync(new URL('../app/avatar/acting.json', import.meta.url)))[avatar] || null;
const clips = clipsArg.split(','), gap = Math.round(parseFloat(gapArg) * SR);
const parts = clips.map((c) => readWav(path.join(wavDir, c + '.wav')));
// each line is rounded to whole video frames so the recorded video can be laid next to it frame by frame
const hop = SR / FPS, info = []; let n = 0;
for (let i = 0; i < parts.length; i++) { const len = Math.round(parts[i].length / hop) * hop; info.push({ name: clips[i], label: clips[i].split('_')[2], s0: n, s1: n + len }); n += len + (i < parts.length - 1 ? Math.round(gap / hop) * hop : 0); }
const audio = new Float32Array(n);
parts.forEach((p, i) => audio.set(p.subarray(0, Math.min(p.length, info[i].s1 - info[i].s0)), info[i].s0));
fs.mkdirSync(outDir, { recursive: true });
writeWav(path.join(outDir, name + '.wav'), audio);
// the app's path for an imported file: features normalized by this speaker's own statistics
const r = extractRaw(audio), st = speakerStats([r]), F = normalize(r.raw, r.nFrames, st), T = r.nFrames;
const seed = [...name].reduce((s, c) => (s * 31 + c.charCodeAt(0)) >>> 0, 7);
const lin = gen.generate(F, T, { seed, style: 'natural' });
const auto = gen.generate(F, T, { seed, style: 'anime', profile });
const overrides = info.map((c) => ({ start: c.s0 / SR - 0.1, end: c.s1 / SR + 0.1, emo: c.label }));
const fix = gen.generate(F, T, { seed, style: 'anime', profile, overrides });
for (const [k, o] of [['linear', lin], ['auto', auto], ['fix', fix]]) fs.writeFileSync(path.join(outDir, `${name}.${k}.f32`), Buffer.from(o.base.buffer, o.base.byteOffset, o.base.byteLength));
for (const c of info) {
  c.f0 = Math.round(c.s0 / hop); c.f1 = Math.round(c.s1 / hop);
  // the automatic reading of the line: phrases inside it, weighted by length
  const p = new Array(EMOS.length).fill(0); let w = 0;
  for (const ph of auto.plan.phrases) { const a = Math.max(ph.t0, c.f0), b = Math.min(ph.t1, c.f1); if (b > a) { for (let e = 0; e < p.length; e++) p[e] += ph.p[e] * (b - a); w += b - a; } }
  c.p = p.map((v) => +(v / Math.max(1, w)).toFixed(3)); let best = 0; for (let e = 1; e < p.length; e++) if (p[e] > p[best]) best = e;
  c.read = EMOS[best]; c.conf = c.p[best]; c.ok = c.read === c.label;
  c.played = auto.plan.phrases.filter((ph) => Math.min(ph.t1, c.f1) > Math.max(ph.t0, c.f0)).map((ph) => ph.emo + (ph.emo2 ? '+' + ph.emo2 : ''));
}
fs.writeFileSync(path.join(outDir, name + '.json'), JSON.stringify({ name, avatar, T, K, fps: FPS, clips: info, plan: auto.plan, planFix: fix.plan }, null, 1));
console.log(name, avatar, 'T', T, info.map((c) => `${c.label}->${c.played.join('/')}(${c.conf})${c.ok ? '' : ' X'}`).join('  '));
