// Serifu Motion — the page's side of speech to text (asr_worker.js does the work).
//   const stt = new SpeechToText(); await stt.load((loaded, total) => …); const { text, lang, conf } = await stt.line(samples, lang);
import { FPS, SR } from './features.js';
import { analyzeSpeech } from './animate.js';
import { linesOf } from './reader.js';

export class SpeechToText {
  constructor() { this.worker = null; this.ready = null; this.wait = new Map(); this.n = 0; }
  load(onProgress) {
    if (this.ready) return this.ready;
    this.ready = new Promise((ok, no) => {
      const w = this.worker = new Worker(new URL('./asr_worker.js', import.meta.url), { type: 'module' });
      w.onerror = (e) => { no(new Error(e.message || 'speech to text could not start')); this.ready = null; };
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'progress') onProgress && onProgress(m.loaded, m.total);
        else if (m.type === 'ready') ok();
        else if (m.type === 'line') { const p = this.wait.get(m.id); this.wait.delete(m.id); p && p.ok(m); }
        else if (m.type === 'error') { if (m.id == null) { no(new Error(m.message)); this.ready = null; w.terminate(); } else { const p = this.wait.get(m.id); this.wait.delete(m.id); p && p.no(new Error(m.message)); } }
      };
      w.postMessage({ type: 'load', base: new URL('../models/whisper/', import.meta.url).href });
    });
    return this.ready;
  }
  line(samples, lang) { const id = ++this.n; return new Promise((ok, no) => { this.wait.set(id, { ok, no }); this.worker.postMessage({ type: 'line', id, samples, lang }); }); }
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
