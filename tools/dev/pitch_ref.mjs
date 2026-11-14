// node tools/dev/pitch_ref.mjs <wav 16k> : who-speaks reference for a woman/man dialogue from pitch (10 fps): 0 = low voice, 1 = high voice, only clear frames
import { readWav } from '../../train/wav.mjs'; import { extractRaw } from '../../app/core/features.js';
const x = readWav(process.argv[2]), r = extractRaw(x), D = 19, T = r.nFrames, out = [];
for (let t = 0; t + 3 <= T; t += 3) { const f = []; for (let u = Math.max(0, t - 6); u < Math.min(T, t + 9); u++) { const v = r.raw[u * D + 1]; if (v > 0) f.push(v); } f.sort((a, b) => a - b); const m = f.length >= 6 ? f[f.length >> 1] : 0; out.push(m === 0 ? -1 : m < 150 ? 0 : m > 175 ? 1 : -1); }
console.log(JSON.stringify(out));
