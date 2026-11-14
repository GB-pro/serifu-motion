// node tools/dev/vowel_seq.mjs <wav 16k mono> [from s] [to s] : the strongest vowel per frame from vowels.js, as a string (X = closed, lower case = unsure)
import { vowelTrack, VOWELS } from '../../app/core/vowels.js';
import { readWav } from '../../train/wav.mjs';
const [wav, a = 0, b = 9999] = process.argv.slice(2), x = readWav(wav), T = Math.floor(x.length / 16000 * 30), v = vowelTrack(x, T), D = 7;
let s = '', last = '';
for (let t = Math.round(a * 30); t < Math.min(T, b * 30); t++) { let best = 0; for (let i = 1; i < 6; i++) if (v[t * D + i] > v[t * D + best]) best = i; const c = v[t * D + best] > 0.55 ? VOWELS[best] : VOWELS[best].toLowerCase(); s += c; }
console.log(s.replace(/(.)\1+/g, '$1'));
