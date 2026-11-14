// Serifu Motion — speech to text in a worker, so the page stays responsive while the model reads a line.
// Messages in:  { type: 'load', base } then { type: 'line', id, samples (Float32Array 16 kHz), lang }
// Messages out: { type: 'progress', loaded, total } while the model files arrive, { type: 'ready' },
//               { type: 'line', id, text, lang, conf }, { type: 'error', id, message }
import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { Transcriber, parseTokens } from './asr.js';
import { SpeakerSplitter } from './speakers.js';

const FILES = [['base-encoder.int8.onnx', 29120534], ['base-decoder.int8.onnx.part0', 66000000], ['base-decoder.int8.onnx.part1', 64672026], ['base-tokens.txt', 816730]];
const TOTAL = FILES.reduce((s, f) => s + f[1], 0);
let tr = null, loading = null, splitter = null;
// the voiceprint network for telling speakers apart (3D-Speaker CAM++, 29.6 MB), loaded the first time it is needed
async function getSplitter(base) {
  if (splitter) return splitter;
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  let n = 0; const b = await get(base, 'campplus_en_voxceleb.onnx', (k) => { n += k; postMessage({ type: 'progress', what: 'speakers', loaded: n, total: 29596978 }); });
  splitter = new SpeakerSplitter(ort, await ort.InferenceSession.create(b, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }));
  return splitter;
}

// One model file: from the browser's cache when it is there, otherwise from the site (and then kept for next time).
async function get(base, name, done) {
  const url = base + name; let cache = null, res = null;
  try { cache = await caches.open('serifu-whisper-base-1'); res = await cache.match(url); } catch { /* no cache storage here */ }
  if (res) { const b = new Uint8Array(await res.arrayBuffer()); done(b.length); return b; }
  res = await fetch(url); if (!res.ok) throw new Error(`${name}: ${res.status}`);
  const chunks = [], reader = res.body.getReader(); let n = 0;
  for (;;) { const { done: end, value } = await reader.read(); if (end) break; chunks.push(value); n += value.length; done(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  if (cache) cache.put(url, new Response(out.slice().buffer, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
  return out;
}
async function load(base) {
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  let loaded = 0; const done = (k) => { loaded += k; postMessage({ type: 'progress', loaded, total: TOTAL }); };
  const [enc, d0, d1, tok] = await Promise.all(FILES.map((f) => get(base, f[0], done)));
  const dec = new Uint8Array(d0.length + d1.length); dec.set(d0); dec.set(d1, d0.length);
  const o = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  tr = new Transcriber(ort, await ort.InferenceSession.create(enc, o), await ort.InferenceSession.create(dec, o), parseTokens(new TextDecoder().decode(tok)));
}
let queue = Promise.resolve();
onmessage = (e) => {
  const m = e.data;
  queue = queue.then(async () => {
    try {
      if (m.type === 'load') { loading = loading || load(m.base); await loading; postMessage({ type: 'ready' }); }
      else if (m.type === 'line') { await loading; const r = await tr.line(m.samples, { lang: m.lang || undefined }); postMessage({ type: 'line', id: m.id, ...r }); }
      else if (m.type === 'forget') { if (tr) tr.lang = null; }
      else if (m.type === 'speakers') { const s = await getSplitter(m.base); const r = await s.run(m.samples, m.k, (u) => postMessage({ type: 'progress', what: 'splitting', id: m.id, u })); postMessage({ type: 'speakers', id: m.id, segments: r.segments }); }
    } catch (err) { if (m.type === 'load') loading = null; postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) }); }
  });
};
