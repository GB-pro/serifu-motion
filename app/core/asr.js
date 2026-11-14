// Serifu Motion — the words of a line: speech to text with the open Whisper model (MIT), run with ONNX Runtime.
//
// The same code runs in the page (onnxruntime-web) and in Node (onnxruntime-node): the caller hands in `ort` and the two
// sessions. A line is at most 30 seconds. Decoding is greedy and without time stamps: where a line is in time is known
// from the phrases of the recording, the model only has to say what was said.
import { logMel } from './reader.js';

const SOT = 50258, EOT = 50257, TRANSCRIBE = 50359, NO_TIMESTAMPS = 50363, LANG0 = 50259, N_LANG = 99, SR = 16000;
export const LANGS = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'];

// tokens file of the model: one "<base64 of the bytes> <id>" per line -> bytes by id
export function parseTokens(text) {
  const out = [], dec = typeof atob === 'function' ? (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)) : (s) => Uint8Array.from(Buffer.from(s, 'base64'));
  for (const raw of text.split('\n')) { const line = raw.trim(), i = line.lastIndexOf(' '); if (i <= 0) continue; try { out[+line.slice(i + 1)] = dec(line.slice(0, i)); } catch (e) { /* a token that is not text */ } }
  return out;
}

export class Transcriber {
  // ort: the runtime; encoder, decoder: its sessions; tokens: parseTokens(...)
  constructor(ort, encoder, decoder, tokens) { this.ort = ort; this.enc = encoder; this.dec = decoder; this.tokens = tokens; this.lang = null; }
  // x: samples of one line at 16 kHz. opts.lang: language code, or none to let the model hear it (and keep it for the lines that follow).
  // Returns { text, lang, conf } — conf: mean probability of the chosen tokens (a low one means the model was guessing)
  async line(x, opts = {}) {
    const ort = this.ort, n = Math.min(x.length, 30 * SR), a = new Float32Array(n + SR); a.set(x.subarray(0, n)); // a second of silence after the line, so the model hears that it ends
    const m = logMel(a), enc = await this.enc.run({ mel: new ort.Tensor('float32', m.data, [1, 80, m.frames]) });
    const ck = enc.n_layer_cross_k, cv = enc.n_layer_cross_v, L = ck.dims[0], W = ck.dims[3];
    let kc = new ort.Tensor('float32', new Float32Array(L * 448 * W), [L, 1, 448, W]), vc = new ort.Tensor('float32', new Float32Array(L * 448 * W), [L, 1, 448, W]);
    const step = async (toks, offset) => {
      const r = await this.dec.run({ tokens: new ort.Tensor('int64', BigInt64Array.from(toks.map(BigInt)), [1, toks.length]), in_n_layer_self_k_cache: kc, in_n_layer_self_v_cache: vc, n_layer_cross_k: ck, n_layer_cross_v: cv, offset: new ort.Tensor('int64', BigInt64Array.from([BigInt(offset)]), [1]) });
      kc = r.out_n_layer_self_k_cache; vc = r.out_n_layer_self_v_cache; const V = r.logits.dims[2]; return r.logits.data.subarray((toks.length - 1) * V, toks.length * V);
    };
    let lang = opts.lang || this.lang;
    if (!lang) { // which language: the token the model would put after the start
      const lg = await step([SOT], 0); let best = 0; for (let i = 1; i < N_LANG; i++) if (lg[LANG0 + i] > lg[LANG0 + best]) best = i;
      lang = LANGS[best]; if (opts.keep !== false && n > 1.5 * SR) this.lang = lang;
      kc = new ort.Tensor('float32', new Float32Array(L * 448 * W), [L, 1, 448, W]); vc = new ort.Tensor('float32', new Float32Array(L * 448 * W), [L, 1, 448, W]);
    }
    const li = Math.max(0, LANGS.indexOf(lang)), start = [SOT, LANG0 + li, TRANSCRIBE, NO_TIMESTAMPS], ids = [];
    let lg = await step(start, 0), off = start.length, sum = 0;
    const most = Math.min(200, Math.round((n / SR) * 12) + 12); // nobody says more than this in the time
    for (let k = 0; k < most; k++) {
      let best = 0, mx = -Infinity; for (let i = 0; i <= EOT; i++) if (lg[i] > mx) { mx = lg[i]; best = i; } // text and the end only
      let z = 0; for (let i = 0; i <= EOT; i++) z += Math.exp(lg[i] - mx); sum += 1 / z;
      if (best === EOT) break;
      ids.push(best);
      if (ids.length >= 12) { const t = ids.slice(-4).join(','), u = ids.slice(-8, -4).join(','), v = ids.slice(-12, -8).join(','); if (t === u && u === v) { ids.length -= 8; break; } } // going round in circles
      { let rep = 0; for (let n = 5; n <= 40 && 2 * n <= ids.length; n++) { let same = true; for (let j = 1; j <= n; j++) if (ids[ids.length - j] !== ids[ids.length - j - n]) { same = false; break; } if (same) { rep = n; break; } } if (rep) { ids.length -= rep; break; } } // the same sentence a second time
      lg = await step([best], off++);
    }
    const bytes = []; for (const id of ids) { const b = this.tokens[id]; if (b) for (const c of b) bytes.push(c); }
    const text = new TextDecoder('utf-8').decode(Uint8Array.from(bytes)).replace(/�/g, '').replace(/\([^)]*\)|\[[^\]]*\]|（[^）]*）|[♪♫]/g, '').trim(); // without the model's notes on sounds that are not speech, like (music)
    return { text, lang, conf: ids.length ? sum / (ids.length + 1) : 0 };
  }
}
