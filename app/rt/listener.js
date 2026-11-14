// Serifu Motion — live: hears what is being said a few seconds before the character plays it.
// The microphone's sound goes two ways. The performer gets it late (LiveSession holds it back by the chosen delay).
// This listener gets it at once: it cuts the speech into pieces at pauses (or every few seconds while someone keeps
// talking), has each piece turned into text together with what was said of the line before it, asks what the words
// mean, and hands back the feeling for the stretch of sound the piece covers, in time for the performer to play it.
const SR = 16000, HOP = 320; // 20 ms
export const PIECE = 2.5, LINE_PAUSE = 0.5, LINE_KEEP = 5; // seconds

export class LineListener {
  // transcribe(samples) -> { text, lang }, understand(lines) -> parts per line (understand.js), onMark({ start, end, emo, intensity, text }) with start/end in samples of the stream
  // maxAge (seconds): a piece that waited longer than this for its turn has been played already and is dropped
  constructor({ transcribe, understand, onMark, onText, maxAge = 6 }) {
    this.maxAge = maxAge;
    this.transcribe = transcribe; this.understand = understand; this.onMark = onMark; this.onText = onText || (() => {});
    this.n = 0; this.buf = []; this.bufN = 0; this.rem = new Float32Array(0);
    this.floor = 0.004; this.speaking = false; this.quiet = 0; this.lineStart = 0; this.pieceStart = 0; this.line = []; // the samples of the line so far
    this.history = []; this.busy = Promise.resolve(); this.lineNo = 0;
  }
  push(x) {
    const a = new Float32Array(this.rem.length + x.length); a.set(this.rem); a.set(x, this.rem.length);
    let i = 0;
    for (; i + HOP <= a.length; i += HOP) {
      let s = 0; for (let j = i; j < i + HOP; j++) s += a[j] * a[j]; const rms = Math.sqrt(s / HOP);
      this.floor = rms < this.floor ? rms * 0.5 + this.floor * 0.5 : this.floor * 1.002; // follows the room's noise up slowly, down fast
      const loud = rms > Math.max(0.01, this.floor * 3.5), at = this.n + i;
      if (loud) {
        if (!this.speaking) { this.speaking = true; if (this.quiet * HOP / SR >= LINE_PAUSE || !this.line.length) { this.line = []; this.lineLen = 0; this.lineStart = at; this.lineNo++; } this.pieceStart = at; }
        this.quiet = 0;
      } else this.quiet++;
      if (this.speaking || this.quiet * HOP / SR < LINE_PAUSE) { this.line.push(a.subarray(i, i + HOP)); this.lineLen = (this.lineLen || 0) + HOP; }
      if (this.speaking) {
        const pieceLen = (at + HOP - this.pieceStart) / SR;
        if (this.quiet * HOP / SR >= 0.3 || pieceLen >= PIECE) { if (pieceLen - this.quiet * HOP / SR >= 0.3) this._piece(this.pieceStart, at + HOP, this.quiet > 0); if (this.quiet > 0) this.speaking = false; else this.pieceStart = at + HOP; }
      }
    }
    this.rem = a.slice(i); this.n += i;
  }
  _piece(start, end, atPause) {
    // the line so far (at most the last LINE_KEEP seconds of it), so the words are heard with the ones before them
    const keep = Math.min(this.lineLen, LINE_KEEP * SR), all = new Float32Array(keep); { let o = keep; for (let k = this.line.length - 1; k >= 0 && o > 0; k--) { o -= HOP; all.set(this.line[k], o); } }
    const lineNo = this.lineNo, lineStart = this.lineStart;
    this.busy = this.busy.then(async () => {
      try {
        if ((this.n - end) / SR > this.maxAge) return;
        const r = await this.transcribe(all); if (!r.text) return;
        this.onText(r.text);
        const cur = { speaker: 'live', start: Math.max(lineStart, end - keep) / SR, end: end / SR, text: r.text, lang: r.lang, lineNo, piece: start };
        const ctx = this.history.filter((h) => h.lineNo !== lineNo || end - h.piece > keep).slice(-6), parts = await this.understand([...ctx, cur]), p = parts && parts[parts.length - 1];
        this.history = [...this.history.filter((h) => h.lineNo !== lineNo || end - h.piece > keep), cur].slice(-8); // earlier pieces of the same line stay as context once they are no longer inside what was just heard
        if (p && p.length) { const last = p[p.length - 1]; this.onMark({ start, end, emo: last.emo, intensity: last.intensity, text: r.text, open: !atPause }); } // the newest words are at the end of the line
      } catch (e) { console.warn('live understanding', e); }
    });
  }
}
