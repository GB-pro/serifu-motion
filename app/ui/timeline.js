// Serifu Motion — multi-speaker timeline editor (canvas). Tracks per speaker: audio, acting plan (emotion per phrase,
// accents), AU/head/torso curves.
import { IDS, K, IDX, EMOS, CHANNELS } from '../core/au.js';
import { FPS } from '../core/features.js';
import { t as tr, chName } from './i18n.js';
import { isDirected } from '../core/project.js';

export const EMO_COLORS = { NEU: '#9aa5b1', HAP: '#ffb300', SAD: '#4f7cff', ANG: '#ff4d4d', FEA: '#a05cff', DIS: '#2bb673' };
const HEAD_W = 168, RULER = 26, SP_H = 26, WAVE_H = 38, EMO_H = 32, CH_H = 40;

export class Timeline {
  constructor(el, app) {
    this.el = el; this.app = app;
    this.canvas = document.createElement('canvas'); el.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.pxPerSec = 120; this.scrollX = 0; this.time = 0; this.vz = 1; // vz: height of the lanes (vertical zoom)
    this.sel = null; this.tool = 'select'; // sel: { sp, a, b } the selected band of an acting track; tool: 'select' | 'cut'
    this.drag = null; this.layout = [];
    this._bind();
    new ResizeObserver(() => this.draw()).observe(el);
  }
  get project() { return this.app.project; }
  x2t(x) { return (x - HEAD_W) / this.pxPerSec + this.scrollX; }
  t2x(t) { return HEAD_W + (t - this.scrollX) * this.pxPerSec; }
  duration() { return Math.max(1, ...this.project.speakers.map((s) => s.T / FPS)); }

  computeLayout() {
    const L = []; let y = RULER;
    for (const sp of this.project.speakers) {
      L.push({ type: 'speaker', sp, y, h: SP_H }); y += SP_H;
      const wh = Math.round(WAVE_H * this.vz), eh = Math.round(EMO_H * Math.sqrt(this.vz)), ch = Math.round(CH_H * this.vz);
      L.push({ type: 'wave', sp, y, h: wh }); y += wh;
      L.push({ type: 'emo', sp, y, h: eh }); y += eh;
      for (const id of sp.visible) { L.push({ type: 'ch', sp, id, k: IDX[id], y, h: ch }); y += ch; }
    }
    this.layout = L; this.contentH = y + 8;
  }
  laneAt(y) { return this.layout.find((l) => y >= l.y && y < l.y + l.h); }
  chRange(id) { const c = CHANNELS[IDX[id]]; return c.range; }
  v2y(l, v) { const [a, b] = this.chRange(l.id); return l.y + l.h - 4 - (v - a) / (b - a) * (l.h - 8); }
  y2v(l, y) { const [a, b] = this.chRange(l.id); return a + (l.y + l.h - 4 - y) / (l.h - 8) * (b - a); }

  _bind() {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this._down(e));
    c.addEventListener('pointermove', (e) => this._move(e));
    c.addEventListener('pointerup', (e) => this._up(e));
    c.addEventListener('pointercancel', () => { this.drag = null; });
    c.addEventListener('dblclick', (e) => this._dbl(e));
    c.addEventListener('wheel', (e) => {
      if (e.altKey) { e.preventDefault(); this.vzoom(e.deltaY < 0 ? 1.15 : 1 / 1.15); return; } // Alt + wheel: lane height
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); const t0 = this.x2t(e.offsetX); this.pxPerSec = Math.max(20, Math.min(800, this.pxPerSec * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); this.scrollX = Math.max(0, t0 - (e.offsetX - HEAD_W) / this.pxPerSec); this.draw(); }
      else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { e.preventDefault(); this.scrollX = Math.max(0, this.scrollX + (e.deltaX || e.deltaY) / this.pxPerSec); this.draw(); }
    }, { passive: false });
  }
  _pos(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  _down(e) {
    const [x, y] = this._pos(e);
    this.canvas.setPointerCapture(e.pointerId);
    if (y < RULER || (x > HEAD_W && this.laneAt(y)?.type === 'wave')) { this.drag = { type: 'seek' }; this.app.seek(Math.max(0, this.x2t(x))); return; }
    const l = this.laneAt(y); if (!l) return;
    if (x < HEAD_W) { // header click
      if (l.type === 'speaker') this.app.selectSpeaker(l.sp);
      if (l.type === 'ch') this.app.selectSpeaker(l.sp);
      return;
    }
    if (l.type === 'emo') { const t0 = this.x2t(x); if (this.tool === 'cut') { this.app.cutBand(l.sp, t0); return; } this.drag = { type: 'range', l, t0, t1: t0 }; return; }
    if (l.type === 'ch') {
      const sp = l.sp, t = this.x2t(x);
      const keys = sp.edits[l.id] || (sp.edits[l.id] = []);
      // hit test existing key
      let hit = -1;
      keys.forEach((k, i) => { const kx = this.t2x(k.t), ky = this.v2y(l, this.finalAt(sp, l.k, k.t)); if (Math.hypot(kx - x, ky - y) < 9) hit = i; });
      this.app.pushUndo();
      if (hit < 0) {
        const cur = this.finalAt(sp, l.k, t);
        const off = this.offsetAt(keys, t);
        const target = this.y2v(l, y);
        keys.push({ t: +t.toFixed(3), v: off + (target - cur) });
        if (!keys.some((k) => k.t < t && t - k.t < 0.45)) keys.push({ t: +(t - 0.4).toFixed(3), v: 0 });
        if (!keys.some((k) => k.t > t && k.t - t < 0.45)) keys.push({ t: +(t + 0.4).toFixed(3), v: 0 });
        keys.sort((a, b) => a.t - b.t);
        hit = keys.findIndex((k) => Math.abs(k.t - t) < 0.0015);
      }
      this.drag = { type: 'key', l, key: keys[hit], startY: y, startV: keys[hit].v, startX: x, startT: keys[hit].t };
      this.app.onEdit(sp);
    }
  }
  offsetAt(keys, t) { // current additive offset
    if (!keys.length) return 0;
    let v = 0; for (let i = 0; i < keys.length - 1; i++) if (t >= keys[i].t && t <= keys[i + 1].t) { const u = (t - keys[i].t) / (keys[i + 1].t - keys[i].t || 1); v = keys[i].v + (keys[i + 1].v - keys[i].v) * u * u * (3 - 2 * u); }
    return v;
  }
  finalAt(sp, k, t) { const f = sp.final; if (!f) return 0; const i = Math.max(0, Math.min(sp.T - 1, Math.round(t * FPS))); return f[i * K + k]; }
  _move(e) {
    const [x, y] = this._pos(e);
    if (!this.drag) { const l = this.laneAt(y); this.canvas.style.cursor = y < RULER ? 'col-resize' : l?.type === 'ch' && x > HEAD_W ? 'crosshair' : l?.type === 'emo' && x > HEAD_W ? (this.tool === 'cut' ? 'cell' : 'pointer') : 'default'; return; }
    const d = this.drag;
    if (d.type === 'seek') this.app.seek(Math.max(0, this.x2t(x)));
    else if (d.type === 'range') { d.t1 = this.x2t(x); this.draw(); }
    else if (d.type === 'key') {
      const [a, b] = this.chRange(d.l.id);
      d.key.v = d.startV + (d.startY - y) / (d.l.h - 8) * (b - a);
      if (Math.abs(x - d.startX) > 4) d.key.t = Math.max(0, +(d.startT + (x - d.startX) / this.pxPerSec).toFixed(3));
      const keys = d.l.sp.edits[d.l.id]; keys.sort((p, q) => p.t - q.t);
      this.app.onEdit(d.l.sp);
    }
  }
  _up() {
    const d = this.drag; this.drag = null;
    if (d?.type === 'range') {
      const a = Math.min(d.t0, d.t1), b = Math.max(d.t0, d.t1), sp = d.l.sp;
      if (b - a > 0.1) this.app.openEmoDialog(sp, a, b, null); // dragged: a new emotion for that stretch
      else { const band = this.bandAt(sp, a); this.select(band ? { sp, ...band } : null); } // clicked: select the band under the pointer
      this.draw();
    }
  }
  _dbl(e) {
    const [x, y] = this._pos(e); const l = this.laneAt(y); if (!l || l.type !== 'ch') return;
    const keys = l.sp.edits[l.id] || [];
    const i = keys.findIndex((k) => Math.hypot(this.t2x(k.t) - x, this.v2y(l, this.finalAt(l.sp, l.k, k.t)) - y) < 10);
    if (i >= 0) { this.app.pushUndo(); keys.splice(i, 1); this.app.onEdit(l.sp); }
  }

  setTime(t) {
    this.time = t;
    const w = this.canvas.width / devicePixelRatio;
    const x = this.t2x(t);
    if (x > w - 40) this.scrollX = t - (w - HEAD_W) * 0.25 / this.pxPerSec;
    if (x < HEAD_W) this.scrollX = Math.max(0, t - 0.5);
    this.draw();
  }
  zoom(f) { this.pxPerSec = Math.max(20, Math.min(800, this.pxPerSec * f)); this.draw(); }
  vzoom(f) { this.vz = Math.max(0.5, Math.min(4, this.vz * f)); this.draw(); }
  // The bands of a speaker's acting track, in time order: the phrases of the acting plan, or the emotion marks of a
  // take the director cannot re-plan. a, b: seconds; emo, inten: what is played there.
  bands(sp) {
    if (isDirected(sp)) return sp.plan.phrases.map((q) => ({ a: Math.max(0, q.tp / FPS), b: q.t1 / FPS, emo: q.emo, inten: q.inten }));
    return sp.emoOverrides.slice().sort((p, q) => p.start - q.start).map((o) => ({ a: o.start, b: o.end, emo: o.emo, inten: o.intensity ?? 0.8 }));
  }
  bandAt(sp, t) { const B = this.bands(sp); return B.find((x) => t >= x.a && t <= x.b) || null; }
  select(sel) {
    this.sel = sel; this.app.onSelectBand(sel); this.draw();
    if (sel) { const w = this.el.clientWidth; if (this.t2x(sel.a) < HEAD_W || this.t2x(sel.b) > w) { this.scrollX = Math.max(0, sel.a - 0.5); } }
  }
  // after a change the bands are rebuilt: keep the selection on the band at the same place
  reselect() { if (!this.sel) return; const sp = this.sel.sp; if (!this.project.speakers.includes(sp)) { this.select(null); return; } const band = this.bandAt(sp, (this.sel.a + this.sel.b) / 2) || this.bandAt(sp, this.sel.a + 0.05); this.sel = band ? { sp, ...band } : null; this.app.onSelectBand(this.sel); }
  step(dir) { // the next (1) or previous (-1) band of the same speaker
    if (!this.sel) { const sp = this.app.selected || this.project.speakers[0]; if (!sp) return; const B = this.bands(sp); const b = B.find((x) => x.b >= this.time) || B[B.length - 1]; if (b) this.select({ sp, ...b }); return; }
    const B = this.bands(this.sel.sp), mid = (this.sel.a + this.sel.b) / 2; let i = B.findIndex((x) => mid >= x.a && mid <= x.b); if (i < 0) i = B.findIndex((x) => x.a > mid) - (dir > 0 ? 1 : 0);
    const j = Math.max(0, Math.min(B.length - 1, i + dir)); if (B[j]) { this.select({ sp: this.sel.sp, ...B[j] }); this.app.seek(B[j].a); }
  }

  draw() {
    this.computeLayout();
    const dpr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth, H = Math.max(this.el.clientHeight, this.contentH);
    if (this.canvas.width !== W * dpr || this.canvas.height !== H * dpr) { this.canvas.width = W * dpr; this.canvas.height = H * dpr; this.canvas.style.width = W + 'px'; this.canvas.style.height = H + 'px'; }
    const g = this.ctx; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const C = (n) => css.getPropertyValue(n).trim();
    g.fillStyle = C('--tl-bg'); g.fillRect(0, 0, W, H);
    // ruler
    g.fillStyle = C('--tl-ruler'); g.fillRect(0, 0, W, RULER);
    g.font = '11px system-ui, sans-serif'; g.textBaseline = 'middle';
    const step = this.pxPerSec > 300 ? 0.25 : this.pxPerSec > 120 ? 0.5 : this.pxPerSec > 50 ? 1 : 2;
    const t0 = Math.floor(this.scrollX / step) * step;
    for (let t = t0; this.t2x(t) < W; t += step) {
      const x = this.t2x(t); if (x < HEAD_W) continue;
      const major = Math.abs(t / (step * 2) - Math.round(t / (step * 2))) < 1e-6;
      g.strokeStyle = C('--tl-grid'); g.beginPath(); g.moveTo(x + 0.5, major ? 12 : 18); g.lineTo(x + 0.5, H); g.stroke();
      if (major) { g.fillStyle = C('--tl-text-dim'); g.fillText(fmt(t), x + 3, 9); }
    }
    for (const l of this.layout) this._lane(g, l, W, C);
    // the selected band
    if (this.sel) { const l = this.layout.find((q) => q.type === 'emo' && q.sp === this.sel.sp); if (l) { const a = Math.max(HEAD_W, this.t2x(this.sel.a)), b = this.t2x(this.sel.b); if (b > HEAD_W) { g.strokeStyle = C('--accent'); g.lineWidth = 3; g.strokeRect(a + 1.5, l.y + 1.5, b - a - 3, l.h - 4); g.lineWidth = 1; } } }
    // range selection
    if (this.drag?.type === 'range') { const l = this.drag.l, a = this.t2x(Math.min(this.drag.t0, this.drag.t1)), b = this.t2x(Math.max(this.drag.t0, this.drag.t1)); g.fillStyle = 'rgba(255,255,255,0.25)'; g.fillRect(a, l.y + 2, b - a, l.h - 4); }
    // playhead
    const px = this.t2x(this.time);
    if (px >= HEAD_W) { g.strokeStyle = C('--accent'); g.lineWidth = 2; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke(); g.lineWidth = 1; g.fillStyle = C('--accent'); g.beginPath(); g.moveTo(px - 6, 0); g.lineTo(px + 6, 0); g.lineTo(px, 9); g.fill(); }
    if (!this.project.speakers.length) { g.fillStyle = C('--tl-text-dim'); g.font = '13px system-ui, sans-serif'; g.fillText(tr('emptyHint'), 20, RULER + 30); }
  }
  _lane(g, l, W, C) {
    const sp = l.sp;
    g.fillStyle = l.type === 'speaker' ? C('--tl-sp') : C('--tl-lane');
    g.fillRect(0, l.y, W, l.h - 1);
    g.fillStyle = C('--tl-head'); g.fillRect(0, l.y, HEAD_W, l.h - 1);
    g.fillStyle = C('--tl-text'); g.font = '12px system-ui, sans-serif';
    if (l.type === 'speaker') {
      g.fillStyle = sp.color; g.fillRect(0, l.y, 4, l.h - 1);
      g.fillStyle = C('--tl-text'); g.font = '600 12px system-ui, sans-serif';
      g.fillText(sp.name + (this.app.selected === sp ? '  ●' : ''), 12, l.y + l.h / 2);
      return;
    }
    g.fillStyle = sp.color; g.fillRect(0, l.y, 4, l.h - 1);
    g.fillStyle = C('--tl-text-dim'); g.font = '11px system-ui, sans-serif';
    const T = sp.T, x0 = Math.max(HEAD_W, this.t2x(0)), x1 = Math.min(W, this.t2x(T / FPS));
    if (l.type === 'wave') {
      g.fillText(tr('waveform'), 12, l.y + l.h / 2);
      if (!sp.audio16k) return;
      const a = sp.audio16k, sps = 16000 / this.pxPerSec, mid = l.y + l.h / 2;
      g.fillStyle = sp.color + 'aa';
      for (let x = Math.floor(x0); x < x1; x++) {
        const s = Math.floor(this.x2t(x) * 16000); let m = 0;
        for (let j = 0; j < sps; j += 4) { const v = Math.abs(a[s + j] || 0); if (v > m) m = v; }
        const h = Math.min(l.h / 2 - 2, m * l.h * 0.9); g.fillRect(x, mid - h, 1, h * 2);
      }
      // what was heard in each line (speech to text), written over the waveform
      g.font = '11px system-ui, sans-serif'; g.textBaseline = 'top';
      for (const ln of sp.script || []) {
        const xa = Math.max(HEAD_W, this.t2x(ln.start)), xb = Math.min(W, this.t2x(ln.end)); if (xb - xa < 30) continue;
        g.save(); g.beginPath(); g.rect(xa, l.y, xb - xa, l.h); g.clip();
        const w = g.measureText(ln.text).width + 8; g.fillStyle = C('--tl-bg') || '#fff'; g.globalAlpha = 0.8; g.fillRect(xa, l.y + 1, Math.min(w, xb - xa), 14); g.globalAlpha = 1;
        g.fillStyle = C('--tl-text'); g.fillText(ln.text, xa + 4, l.y + 2); g.restore();
      }
      g.textBaseline = 'middle';
      return;
    }
    if (l.type === 'emo') {
      g.fillText(tr(isDirected(sp) ? 'acting' : 'emotion'), 12, l.y + l.h / 2);
      if (!sp.emo) return;
      if (isDirected(sp)) {
        // the acting plan: one block per phrase (emotion played, strength), a stripe for a blended second emotion,
        // ticks for the accents; a white frame marks emotions set by the user
        for (const ph of sp.plan.phrases) {
          const a = this.t2x(ph.tp / FPS), b = this.t2x(ph.t1 / FPS); if (b < HEAD_W || a > W) continue;
          const xa = Math.max(HEAD_W, a), top = l.y + 3, hh = l.h - 12;
          g.globalAlpha = ph.fixed ? 1 : 0.45 + 0.55 * Math.min(1, ph.conf * 1.5); g.fillStyle = EMO_COLORS[ph.emo]; g.fillRect(xa, top, b - xa, hh);
          if (ph.emo2) { g.fillStyle = EMO_COLORS[ph.emo2]; g.fillRect(xa, top + hh - 5, b - xa, 5); }
          g.globalAlpha = 1;
          if (ph.fixed) { g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.strokeRect(xa + 0.75, top + 0.75, b - xa - 1.5, hh - 1.5); g.lineWidth = 1; }
          if (b - xa > 44) { g.fillStyle = '#14161a'; g.font = '600 11px system-ui, sans-serif'; g.fillText(trunc(`${tr(ph.emo)}${ph.emo2 ? '+' + tr(ph.emo2) : ''} ${Math.round(ph.inten * 100)}%`, Math.floor((b - xa) / 7)), xa + 5, top + hh / 2); }
          for (const ac of ph.accents) { const xc = this.t2x(ac.t / FPS); if (xc < HEAD_W) continue; const r = ac.key ? 4 : 3; g.fillStyle = ac.key ? C('--tl-text') : C('--tl-text-dim'); g.beginPath(); g.moveTo(xc - r, l.y + l.h - 2); g.lineTo(xc + r, l.y + l.h - 2); g.lineTo(xc, l.y + l.h - 9); g.fill(); }
        }
        return;
      }
      for (let x = Math.floor(x0); x < x1; x += 2) {
        const t = Math.round(this.x2t(x) * FPS); if (t < 0 || t >= T) continue;
        let bi = 0, bv = 0; for (let e = 0; e < 6; e++) { const v = sp.emo[t * 6 + e]; if (v > bv) { bv = v; bi = e; } }
        g.globalAlpha = 0.25 + 0.75 * Math.min(1, (bv - 0.17) * 2.2); g.fillStyle = EMO_COLORS[EMOS[bi]]; g.fillRect(x, l.y + 3, 2, l.h - 7);
      }
      g.globalAlpha = 1;
      for (const o of sp.emoOverrides) {
        const a = this.t2x(o.start), b = this.t2x(o.end); if (b < HEAD_W) continue;
        g.fillStyle = EMO_COLORS[o.emo]; g.fillRect(Math.max(HEAD_W, a), l.y + 2, b - Math.max(HEAD_W, a), l.h - 5);
        g.strokeStyle = '#fff'; g.strokeRect(Math.max(HEAD_W, a) + 0.5, l.y + 2.5, b - Math.max(HEAD_W, a) - 1, l.h - 6);
        g.fillStyle = '#fff'; g.font = '600 11px system-ui, sans-serif'; g.fillText(`${tr(o.emo)} ${Math.round(o.intensity * 100)}%`, Math.max(HEAD_W, a) + 6, l.y + l.h / 2);
      }
      return;
    }
    // channel curve
    const ch = CHANNELS[l.k];
    g.fillStyle = C('--tl-text'); g.font = '600 11px system-ui, sans-serif'; g.fillText(ch.id.startsWith('AU') ? ch.id : '', 12, l.y + 12);
    g.fillStyle = C('--tl-text-dim'); g.font = '10px system-ui, sans-serif'; g.fillText(trunc(chName(ch), 24), 12, l.y + 27);
    const [ra, rb] = ch.range;
    if (ra < 0) { g.strokeStyle = C('--tl-grid'); g.beginPath(); const yz = this.v2y(l, 0); g.moveTo(HEAD_W, yz + 0.5); g.lineTo(W, yz + 0.5); g.stroke(); }
    if (!sp.final) return;
    // base (dashed) and final
    for (const [arr, style, lw] of [[sp.base, C('--tl-base'), 1], [sp.final, sp.color, 1.8]]) {
      if (!arr) continue;
      g.strokeStyle = style; g.lineWidth = lw; g.beginPath(); let first = true;
      for (let x = Math.floor(x0); x < x1; x += 1) {
        const ft = this.x2t(x) * FPS, t = Math.floor(ft); if (t < 0 || t >= T) continue;
        const t2 = Math.min(T - 1, t + 1), v = arr[t * K + l.k] + (arr[t2 * K + l.k] - arr[t * K + l.k]) * (ft - t); // between two frames the curve runs straight, as the stage plays it
        const y = this.v2y(l, Math.max(ra, Math.min(rb, v)));
        if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y);
      }
      g.stroke();
    }
    g.lineWidth = 1;
    const keys = sp.edits[l.id] || [];
    for (const k of keys) {
      const x = this.t2x(k.t); if (x < HEAD_W || x > W) continue;
      const y = this.v2y(l, Math.max(ra, Math.min(rb, this.finalAt(sp, l.k, k.t))));
      g.fillStyle = '#fff'; g.strokeStyle = sp.color; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.fill(); g.stroke(); g.lineWidth = 1;
    }
  }
}
const fmt = (t) => { const m = Math.floor(t / 60), s = t - m * 60; return `${m}:${s < 10 ? '0' : ''}${s.toFixed(s % 1 ? 1 : 0)}`; };
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
