// Serifu Motion — understanding what is said. The lines of the dialogue (text from asr.js, with their times and
// speakers) go in; for every line come back the phrases it falls into and, for each phrase, the feeling the words carry
// in their context and how strong it is. The result is written into the speakers' emotion marks (emoOverrides), which
// the director (animate.js: directOffline) plays instead of what the tone of voice alone suggested.
//
// Two readers: an external AI (the Claude API, with the key the user typed into the settings of the app; the key and
// the text of the lines go from the browser straight to the API and nowhere else), and a small reader on the device
// that knows a few hundred words and is used when there is no key or the request fails.
import { EMOS } from './au.js';

export const API_URL = 'https://api.anthropic.com/v1/messages';
export const DEFAULT_MODEL = 'claude-opus-5-5';
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

const SYSTEM = `You direct the facial acting of animated characters. You get the lines of a dialogue as JSON: for each line its index "i", the speaker, start and end time in seconds, and the text. The text comes from automatic speech recognition, so it can contain wrong or missing words; read through such errors and judge from the context of the whole dialogue what was meant.

For every line decide what the speaker feels while saying it, from the meaning of the words in their context (not from single keywords: irony, questions, politeness, quoting someone else and negation all matter). When the feeling changes inside a line, split the line into parts in the order they are spoken; otherwise give one part.

Each part has:
- "share": the fraction of the line's duration this part takes (the shares of a line add up to 1)
- "emo": one of NEU (calm, matter-of-fact), HAP (joy, warmth, amusement, pride), SAD (sadness, regret, disappointment, loneliness), ANG (anger, irritation, strong determination), FEA (fear, worry, shock, startled surprise), DIS (disgust, contempt, dislike)
- "intensity": 0.2 (a hint) to 1.3 (overflowing); ordinary friendly talk is about 0.5 to 0.7
- "emo2" (optional): a second feeling mixed in, same codes
- "emphasis" (optional): the one word or short phrase of this part that carries the most weight, copied from the text

Explanations, lively narration and polite cheerful talk are HAP at a moderate intensity rather than NEU; keep NEU for lines that really are flat.

Answer with JSON only, no other text, in this form:
{"lines":[{"i":0,"parts":[{"share":1,"emo":"HAP","intensity":0.6,"emphasis":"..."}]}]}`;

// The request for a dialogue. lines: [{ speaker, start, end, text }] in the order they are spoken.
export function buildRequest(lines, { model = DEFAULT_MODEL } = {}) {
  const payload = lines.map((l, i) => ({ i, speaker: l.speaker, start: +l.start.toFixed(2), end: +l.end.toFixed(2), text: l.text }));
  return { model, max_tokens: 16000, system: SYSTEM, messages: [{ role: 'user', content: JSON.stringify({ lines: payload }) }] };
}

// The first JSON object in a text (the answer may be wrapped in a code fence or a sentence).
export function parseJson(text) {
  const a = text.indexOf('{'); if (a < 0) throw new Error('no JSON in the answer');
  let depth = 0, str = false, esc = false;
  for (let i = a; i < text.length; i++) {
    const c = text[i];
    if (str) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') str = false; continue; }
    if (c === '"') str = true; else if (c === '{') depth++; else if (c === '}' && --depth === 0) return JSON.parse(text.slice(a, i + 1));
  }
  throw new Error('the JSON of the answer is cut off');
}

// Makes whatever came back safe to use: one entry per line, parts with known feelings and shares that add up to 1.
export function tidy(result, n) {
  const out = new Array(n).fill(null);
  for (const l of (result && Array.isArray(result.lines) ? result.lines : [])) {
    if (!Number.isInteger(l.i) || l.i < 0 || l.i >= n || !Array.isArray(l.parts)) continue;
    const parts = l.parts.filter((p) => p && EMOS.includes(p.emo)).map((p) => ({
      share: Number.isFinite(+p.share) && +p.share > 0 ? +p.share : 1, emo: p.emo, intensity: clamp(Number.isFinite(+p.intensity) ? +p.intensity : 0.6, 0.2, 1.3),
      ...(EMOS.includes(p.emo2) && p.emo2 !== p.emo ? { emo2: p.emo2 } : {}), ...(typeof p.emphasis === 'string' && p.emphasis ? { emphasis: p.emphasis.slice(0, 60) } : {}) }));
    if (!parts.length) continue;
    const sum = parts.reduce((s, p) => s + p.share, 0); for (const p of parts) p.share /= sum;
    out[l.i] = parts;
  }
  return out;
}

// Asks the external AI. Returns an array with the parts of each line (null where the answer had nothing usable).
// Throws with a short reason (code: 'key', 'rate', 'network', 'refused', 'answer', 'http') when it did not work.
export async function understandRemote(lines, { key, model = DEFAULT_MODEL, fetchFn = globalThis.fetch, signal } = {}) {
  const fail = (code, msg) => Object.assign(new Error(msg), { code });
  if (!key) throw fail('key', 'no API key');
  let res;
  try {
    res = await fetchFn(API_URL, { method: 'POST', signal, headers: { 'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2026-10-21', 'anthropic-dangerous-direct-browser-access': 'true' }, body: JSON.stringify(buildRequest(lines, { model })) });
  } catch (e) { throw fail('network', String(e && e.message || e)); }
  if (!res.ok) {
    let msg = ''; try { msg = (await res.json()).error?.message || ''; } catch { /* no body */ }
    throw fail(res.status === 401 || res.status === 403 ? 'key' : res.status === 429 || res.status === 529 ? 'rate' : 'http', `${res.status} ${msg}`.trim());
  }
  const data = await res.json();
  if (data.stop_reason === 'refusal') throw fail('refused', 'the AI declined to read this dialogue');
  const text = (data.content || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
  let parsed; try { parsed = parseJson(text); } catch (e) { throw fail('answer', e.message + (data.stop_reason === 'max_tokens' ? ' (too long)' : '')); }
  const out = tidy(parsed, lines.length);
  if (!out.some(Boolean)) throw fail('answer', 'the answer had no usable lines');
  return out;
}

// ---------- the small reader on the device ----------
// Words that usually carry a feeling, per language (matched as substrings, lower case). It cannot follow context or
// irony; it exists so that the app still does something sensible with the words when no external AI is available.
const LEX = {
  ja: {
    HAP: ['嬉し', 'うれし', '楽し', 'たのし', 'ありがと', '大好き', '好き', '最高', 'やった', '素敵', 'すてき', 'おいし', '美味し', '幸せ', 'しあわせ', '面白', 'おもしろ', 'すごい', '凄い', 'よかった', '良かった', 'おめでと', 'わーい', '可愛', 'かわい', '笑', 'ワクワク', 'わくわく', '楽しみ', 'いいね', 'ようこそ', 'みたいと思います', 'ましょう'],
    SAD: ['悲し', 'かなし', '寂し', 'さびし', 'さみし', '残念', 'ごめん', 'すみません', '申し訳', '泣', '涙', 'つらい', '辛い', '切な', 'もう会えな', 'さよなら', 'さようなら', '失っ', '亡くな', '後悔', 'ひとりぼっち', 'がっかり', '落ち込'],
    ANG: ['怒', 'ふざけ', '許さ', '許せ', 'むかつ', 'ムカつ', 'うるさい', 'うるせ', 'バカ', '馬鹿', 'ばか', 'やめろ', 'やめて', 'いい加減', '何やって', 'なんだと', '黙れ', 'だまれ', '最低', 'しつこい', '邪魔', 'イライラ', 'くそ', 'クソ'],
    FEA: ['怖', 'こわ', '恐', 'やばい', 'ヤバい', '助けて', '危な', 'あぶな', 'まさか', '嘘', 'うそ', 'びっくり', '不安', '心配', 'どうしよう', '逃げ', '大変', 'しまった', 'えっ', 'ええっ'],
    DIS: ['気持ち悪', 'きもち悪', 'キモ', 'まずい', '不味', '汚', 'きたな', '臭', 'くさい', 'うわ', 'げっ', '最悪', '嫌い', 'きらい', '嫌だ', 'いやだ', 'うんざり', '吐き'],
    NEG: ['ない', 'ません', 'なかった'],
  },
  en: {
    HAP: ['happy', 'glad', 'love', 'great', 'wonderful', 'awesome', 'thank', 'yay', 'fantastic', 'amazing', 'delicious', 'fun', 'excited', 'beautiful', 'congrat', 'welcome', 'haha', 'perfect', 'brilliant', 'nice', 'good news', "can't wait"],
    SAD: ['sad', 'sorry', 'miss you', 'cry', 'tears', 'lonely', 'alone', 'lost', 'gone', 'died', 'dead', 'regret', 'unfortunately', 'goodbye', 'hurts', 'heartbroken', 'disappoint', 'never see'],
    ANG: ['angry', 'furious', 'shut up', 'how dare', 'stop it', 'hate you', 'idiot', 'stupid', 'damn', 'get out', 'enough', 'what the hell', 'ridiculous', 'unacceptable', 'fed up', 'liar'],
    FEA: ['afraid', 'scared', 'scary', 'terrified', 'help me', 'oh no', 'danger', 'watch out', 'run!', 'worried', 'nervous', 'what was that', 'oh my god', 'no way', 'careful', 'panic'],
    DIS: ['disgust', 'gross', 'ew', 'yuck', 'nasty', 'stinks', 'smells', 'filthy', 'revolting', 'awful', 'sick of', 'creepy'],
    NEG: ["n't", ' not ', ' no ', 'never'],
  },
};
const ORDER = ['HAP', 'SAD', 'ANG', 'FEA', 'DIS'];
// Returns the parts of each line, or null for a line in which no word was recognised (its voice reading then stays).
export function understandLocal(lines) {
  return lines.map((l) => {
    const lex = LEX[l.lang] || LEX[/[぀-ヿ一-鿿]/.test(l.text) ? 'ja' : 'en'], text = ' ' + l.text.toLowerCase() + ' ';
    const score = ORDER.map((e) => lex[e].reduce((s, w) => s + (text.includes(w) ? 1 : 0), 0));
    let best = 0; for (let i = 1; i < score.length; i++) if (score[i] > score[best]) best = i;
    if (!score[best]) return null;
    let emo = ORDER[best];
    // "not happy", 「楽しくない」: a denied good feeling is played as a sad one; other denials are left as they are
    if (emo === 'HAP' && lex.HAP.some((w) => { const k = text.indexOf(w); return k >= 0 && lex.NEG.some((n) => { const j = text.indexOf(n, l.lang === 'en' ? Math.max(0, k - 8) : k); return j >= 0 && Math.abs(j - k) < 10; }); })) emo = 'SAD';
    const marks = (l.text.match(/[!！]/g) || []).length;
    return [{ share: 1, emo, intensity: clamp(0.55 + 0.1 * (score[best] - 1) + 0.12 * Math.min(marks, 3), 0.2, 1.1) }];
  });
}

// ---------- into the timeline ----------
// Emotion marks for one speaker from the parts of their lines. Marks the understanding wrote earlier (ai: true) are
// replaced; marks the user set by hand stay, and win where they overlap a new one.
export function toOverrides(lines, parts, existing = []) {
  const manual = existing.filter((o) => !o.ai), out = manual.slice();
  lines.forEach((l, i) => {
    if (!parts[i]) return;
    let t = l.start; const dur = l.end - l.start;
    for (const p of parts[i]) {
      const a = t, b = t + dur * p.share; t = b;
      if (manual.some((m) => m.start < b && m.end > a)) continue;
      out.push({ start: +a.toFixed(3), end: +b.toFixed(3), emo: p.emo, intensity: +p.intensity.toFixed(2), ai: true, ...(p.emo2 ? { emo2: p.emo2 } : {}), ...(p.emphasis ? { emphasis: p.emphasis } : {}) });
    }
  });
  return out.sort((x, y) => x.start - y.start);
}
