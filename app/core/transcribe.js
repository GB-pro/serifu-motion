// Serifu Motion — the page's side of speech to text (asr_worker.js does the work).
//   const stt = new SpeechToText(); await stt.load((loaded, total) => …); const { text, lang, conf } = await stt.line(samples, lang);
import { FPS, SR } from './features.js';
import { analyzeSpeech } from './animate.js';
import { linesOf } from './reader.js';

export class SpeechToText {
  constructor() { this.worker = null; this.ready = null; this.wait = new Map(); this.n = 0; }
  // the worker, started once; speech to text and the speaker split share it
  start() {
    if (this.worker) return this.worker;
    const w = this.worker = new Worker(new URL('./asr_worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => this._msg(e.data); w.onerror = (e) => { for (const p of this.wait.values()) p.no(new Error(e.message || 'worker failed')); this.wait.clear(); if (this._loadNo) this._loadNo(new Error(e.message || 'speech to text could not start')); this.ready = null; };
    return w;
  }
  _msg(m) {
    if (m.type === 'progress') { if (m.what === 'splitting' || m.what === 'speakers') this.onSplit && this.onSplit(m); else this.onProgress && this.onProgress(m.loaded, m.total); }
    else if (m.type === 'ready') { this._loadOk && this._loadOk(); }
    else if (m.type === 'line' || m.type === 'speakers') { const p = this.wait.get(m.id); this.wait.delete(m.id); p && p.ok(m); }
    else if (m.type === 'error') { if (m.id == null) { this._loadNo && this._loadNo(new Error(m.message)); this.ready = null; } else { const p = this.wait.get(m.id); this.wait.delete(m.id); p && p.no(new Error(m.message)); } }
  }
  // who speaks when in a mixed recording: [{ start, end, speaker }]; onProgress({ what: 'speakers', loaded, total } | { what: 'splitting', u })
  speakers(samples, k, onProgress) {
    this.start(); this.onSplit = onProgress; const id = ++this.n;
    return new Promise((ok, no) => { this.wait.set(id, { ok: (m) => ok(m.segments), no }); this.worker.postMessage({ type: 'speakers', id, samples, k, base: new URL('../models/speaker/', import.meta.url).href }); });
  }
  load(onProgress) {
    if (this.ready) return this.ready;
    this.onProgress = onProgress;
    this.ready = new Promise((ok, no) => { this._loadOk = ok; this._loadNo = no; this.start().postMessage({ type: 'load', base: new URL('../models/whisper/', import.meta.url).href }); });
    return this.ready;
  }
  line(samples, lang) { this.start(); const id = ++this.n; return new Promise((ok, no) => { this.wait.set(id, { ok, no }); this.worker.postMessage({ type: 'line', id, samples, lang }); }); }
  forget() { this.worker && this.worker.postMessage({ type: 'forget' }); } // a new recording may be in another language
}

// The lines of a speaker's recording (start and end in seconds) with the samples the model is given for each.
export function speechLines(sp) {
  if (!sp.features) return [];
  return linesOf(analyzeSpeech(sp.features, sp.T).phrases).map((l) => {
    const a = Math.max(0, Math.round((l.t0 / FPS - 0.15) * SR)), b = Math.min(sp.audio16k.length, Math.round((l.t1 / FPS + 0.15) * SR));
    return { start: l.t0 / FPS, end: l.t1 / FPS, samples: sp.audio16k.slice(a, b) };
  });
}
