import fs from 'fs'; import { EmoNet } from '../app/core/model.js';
// Node may hand back small files as views into a shared pool: copy exactly the file's bytes.
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const net = new EmoNet(JSON.parse(fs.readFileSync(process.argv[2])));
const X = readF32(process.argv[3]); const T = X.length / 20;
const Y = net.run(X, T); console.log(JSON.stringify(Array.from(Y.slice(0, 8)).map(v=>+v.toFixed(5))), JSON.stringify(Array.from(Y.slice((T-1)*net.Dout, (T-1)*net.Dout+4)).map(v=>+v.toFixed(5))));
