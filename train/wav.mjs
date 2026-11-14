// Reading wav files (16/24/32-bit PCM, 32-bit float; any number of channels, mixed down) -> Float32Array at 16 kHz.
import fs from 'fs';
import { resampleTo16k, SR } from '../app/core/features.js';

export function readWav(f) {
  const b = fs.readFileSync(f); let o = 12, data = null, sr = SR, ch = 1, bits = 16, fmt = 1;
  while (o + 8 <= b.length) { const id = b.toString('ascii', o, o + 4), sz = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { fmt = b.readUInt16LE(o + 8); ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); bits = b.readUInt16LE(o + 22); }
    if (id === 'data') { data = b.subarray(o + 8, Math.min(b.length, o + 8 + sz)); break; } o += 8 + sz + (sz & 1); }
  const pcm = fmt === 1 || fmt === 0xfffe, by = bits / 8;
  const get = pcm && bits === 16 ? (p) => data.readInt16LE(p) / 32768 : pcm && bits === 24 ? (p) => data.readIntLE(p, 3) / 8388608
    : pcm && bits === 32 ? (p) => data.readInt32LE(p) / 2147483648 : fmt === 3 && bits === 32 ? (p) => data.readFloatLE(p) : null;
  if (!data || !get) throw new Error(`unsupported wav (format ${fmt}, ${bits} bit)`);
  const n = Math.floor(data.length / by / ch), x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < ch; c++) s += get((i * ch + c) * by); x[i] = s / ch; }
  return sr === SR ? x : resampleTo16k(x, sr);  // the app's own resampler, as for a file dropped into the browser
}
