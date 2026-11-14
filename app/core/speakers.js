// Serifu Motion — who is speaking when, in a recording of several people (speaker diarization).
// A speaker-embedding network (3D-Speaker CAM++, trained to tell voices apart) turns every short window of speech
// into a "voiceprint". Windows are grouped into the asked number of speakers by how alike their voiceprints are,
// and every moment of speech is given to the speaker whose windows around it it resembles most; short flips are
// smoothed away. The voice's pitch alone (the first versions) cannot tell two people of the same sex apart.
//   const d = new SpeakerSplitter(ort, session); const { segments } = await d.run(x16k, k, onProgress)
//   segments: [{ start, end, speaker }] (seconds)

// ---------- log mel filter bank, as Kaldi computes it with the settings sherpa-onnx uses for these networks ----------
// (80 bins from 20 Hz to 7600 Hz, povey window, frames not snipped at the edges; checked against sherpa-onnx's own embeddings)
const FR = 400, SH = 160, NFFT = 512, NMEL = 80;
let MEL = null, WINDOW = null;
function tables() {
  if (MEL) return;
  WINDOW = new Float32Array(FR); for (let i = 0; i < FR; i++) WINDOW[i] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FR - 1)), 0.85); // povey
  const mel = (f) => 1127 * Math.log(1 + f / 700), lo = mel(20), hi = mel(7600), d = (hi - lo) / (NMEL + 1);
  MEL = [];
  for (let m = 0; m < NMEL; m++) {
    const l = lo + m * d, c = l + d, r = c + d, w = new Float32Array(NFFT / 2);
    for (let i = 0; i < NFFT / 2; i++) { const z = mel(i * 16000 / NFFT); w[i] = z > l && z < r ? (z <= c ? (z - l) / d : (r - z) / d) : 0; }
    let a = 0; while (a < w.length && !w[a]) a++; let b = w.length; while (b > a && !w[b - 1]) b--;
    MEL.push({ a, w: w.slice(a, b) });
  }
}
function fft(re, im) { // in place, length a power of two
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let j = 0; j < len / 2; j++) { const ar = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci, ai = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr; re[i + j + len / 2] = re[i + j] - ar; im[i + j + len / 2] = im[i + j] - ai; re[i + j] += ar; im[i + j] += ai; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
// x: samples in -1..1 at 16 kHz (what the network was trained on: see scale). Returns { data: Float32Array(n * 80), n } (one frame per 10 ms)
// scale: 1 for networks trained on samples in -1..1 (3D-Speaker), 32768 for those trained on 16-bit values (WeSpeaker)
export function fbank(x, scale = 1) {
  tables();
  const N = x.length, n = Math.floor((N + SH / 2) / SH), out = new Float32Array(n * NMEL);
  const re = new Float32Array(NFFT), im = new Float32Array(NFFT), fr = new Float32Array(FR);
  for (let t = 0; t < n; t++) {
    const o = t * SH + SH / 2 - FR / 2; let mean = 0; // frames centred on every 10 ms, the edges mirrored (Kaldi without snip-edges)
    for (let i = 0; i < FR; i++) { let k = o + i; if (k < 0) k = -k - 1; if (k >= N) k = 2 * N - 1 - k; fr[i] = (x[Math.max(0, Math.min(N - 1, k))] || 0) * scale; mean += fr[i]; } mean /= FR;
    for (let i = 0; i < FR; i++) fr[i] -= mean;
    for (let i = FR - 1; i > 0; i--) fr[i] -= 0.97 * fr[i - 1]; fr[0] -= 0.97 * fr[0];
    re.fill(0); im.fill(0); for (let i = 0; i < FR; i++) re[i] = fr[i] * WINDOW[i];
    fft(re, im);
    for (let m = 0; m < NMEL; m++) { const { a, w } = MEL[m]; let s = 0; for (let i = 0; i < w.length; i++) { const k = a + i; s += w[i] * (re[k] * re[k] + im[k] * im[k]); } out[t * NMEL + m] = Math.log(Math.max(s, 1.1920929e-7)); }
  }
  return { data: out, n };
}

// ---------- the pipeline ----------
const WIN_S = 2.0, HOP_S = 0.5; // 2 s windows (the length sherpa-onnx gives these networks; the most stable grouping in our checks)
const cos = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const unit = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };

// voice activity: 10 ms frames whose level is within reach of the loud speech of the recording
function activity(x) {
  const n = Math.floor(x.length / SH), db = new Float32Array(n);
  for (let t = 0; t < n; t++) { let s = 0; for (let i = t * SH; i < (t + 1) * SH; i++) s += x[i] * x[i]; db[t] = 10 * Math.log10(s / SH + 1e-10); }
  const s = Array.from(db).sort((a, b) => a - b), hi = s[Math.floor(n * 0.95)], lo = s[Math.floor(n * 0.1)], thr = Math.max(lo + 6, hi - 30);
  const act = new Uint8Array(n); for (let t = 0; t < n; t++) act[t] = db[t] > thr ? 1 : 0;
  // fill pauses shorter than 0.3 s, drop blips shorter than 0.15 s
  for (let t = 0; t < n;) { if (!act[t]) { let e = t; while (e < n && !act[e]) e++; if (t > 0 && e < n && e - t < 30) act.fill(1, t, e); t = e; } else t++; }
  for (let t = 0; t < n;) { if (act[t]) { let e = t; while (e < n && act[e]) e++; if (e - t < 15) act.fill(0, t, e); t = e; } else t++; }
  return act;
}

export class SpeakerSplitter {
  // scale: see fbank (1 for 3D-Speaker networks, 32768 for WeSpeaker ones)
  constructor(ort, session, scale = 1) { this.ort = ort; this.s = session; this.scale = scale; this.input = session.inputNames[0]; this.output = session.outputNames[0]; }
  async embed(x) {
    const f = fbank(x, this.scale), mean = new Float32Array(NMEL);
    for (let t = 0; t < f.n; t++) for (let m = 0; m < NMEL; m++) mean[m] += f.data[t * NMEL + m] / f.n;
    for (let t = 0; t < f.n; t++) for (let m = 0; m < NMEL; m++) f.data[t * NMEL + m] -= mean[m];
    const r = await this.s.run({ [this.input]: new this.ort.Tensor('float32', f.data, [1, f.n, NMEL]) });
    return unit(Array.from(r[this.output].data));
  }
  async run(x, k = 2, onProgress = null) {
    const act = activity(x), n = act.length, W = Math.round(WIN_S * 100), H = Math.round(HOP_S * 100);
    // windows over speech: every HOP inside a stretch of speech; a short stretch gets one window of its own length
    const wins = [];
    for (let t = 0; t < n;) {
      if (!act[t]) { t++; continue; } let e = t; while (e < n && act[e]) e++;
      if (e - t <= W) wins.push([t, e]); else { for (let a = t; a + W <= e; a += H) wins.push([a, a + W]); if ((e - t - W) % H > H / 2) wins.push([e - W, e]); }
      t = e;
    }
    const emb = [];
    for (let i = 0; i < wins.length; i++) {
      const [a, b] = wins[i], seg = x.subarray(a * SH, Math.min(x.length, b * SH + FR));
      emb.push(seg.length >= 0.4 * 16000 ? await this.embed(seg) : null);
      if (onProgress && i % 8 === 0) { onProgress((i + 1) / wins.length); await new Promise((r) => setTimeout(r, 0)); }
    }
    const idx = emb.map((e, i) => (e ? i : -1)).filter((i) => i >= 0), E = idx.map((i) => emb[i]), Wt = idx.map((i) => Math.min(1, (wins[i][1] - wins[i][0]) / W));
    const lab = cluster(E, Wt, Math.min(k, E.length));
    const C = []; for (let j = 0; j < k; j++) { const v = new Array(E[0]?.length || 0).fill(0); E.forEach((e, i) => { if (lab[i] === j) e.forEach((z, d) => (v[d] += z)); }); C.push(unit(v)); }
    // every 10 ms of speech: which speaker the windows covering it resemble most (windows nearer their centre count more)
    const score = new Float32Array(n * k), cover = new Float32Array(n);
    idx.forEach((wi, i) => { const [a, b] = wins[wi], s = C.map((c) => cos(E[i], c)); for (let t = a; t < b; t++) { const u = 1 - Math.abs((t - a) / Math.max(1, b - a) - 0.5) * 1.2; for (let j = 0; j < k; j++) score[t * k + j] += u * s[j]; cover[t] += u; } });
    let lbl = new Int8Array(n).fill(-1);
    for (let t = 0; t < n; t++) if (act[t]) { let bj = 0; for (let j = 1; j < k; j++) if (score[t * k + j] > score[t * k + bj]) bj = j; lbl[t] = cover[t] > 0 ? bj : -2; }
    // stretches no window covered take the speaker before them; turns shorter than 0.4 s join their neighbours
    for (let t = 0; t < n; t++) if (lbl[t] === -2) lbl[t] = t > 0 && lbl[t - 1] >= 0 ? lbl[t - 1] : 0;
    for (let pass = 0; pass < 2; pass++) for (let t = 0; t < n;) {
      if (lbl[t] < 0) { t++; continue; } let e = t; while (e < n && lbl[e] === lbl[t]) e++;
      if (e - t < 40) { const prev = t > 0 ? lbl[t - 1] : -1, next = e < n ? lbl[e] : -1, to = prev >= 0 ? prev : next; if (to >= 0 && to !== lbl[t] && (prev === next || prev < 0 || next < 0 || pass)) lbl.fill(to, t, e); }
      t = e;
    }
    const segments = []; for (let t = 0; t < n;) { if (lbl[t] < 0) { t++; continue; } let e = t; while (e < n && lbl[e] === lbl[t]) e++; segments.push({ start: t / 100, end: e / 100, speaker: lbl[t] }); t = e; }
    return { segments };
  }
}

// Agglomerative grouping by average cosine similarity down to k groups, then a few rounds of reassigning every
// window to its nearest group centre (spherical k-means) to clean up early merges.
export function cluster(E, W, k) {
  const N = E.length; if (!N) return [];
  if (N <= k) return E.map((_, i) => i);
  let groups = E.map((_, i) => [i]);
  const S = E.map((a) => E.map((b) => cos(a, b)));
  // work on a manageable number of groups: start from consecutive pairs when there are many windows
  const avg = (g, h) => { let s = 0; for (const i of g) for (const j of h) s += S[i][j]; return s / (g.length * h.length); };
  const M = groups.map((g, i) => groups.map((h, j) => (i === j ? -Infinity : avg(g, h))));
  // Group down to a few more groups than speakers first: a laugh, a cough or a jingle forms a small group of its own
  // instead of taking a speaker's place (which would merge two real voices). The small groups then join the nearest
  // of the k groups that hold the most speech.
  const K0 = Math.min(N, Math.max(k + 2, k * 3));
  while (groups.length > K0) {
    let bi = 0, bj = 1, bv = -Infinity;
    for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) if (M[i][j] > bv) { bv = M[i][j]; bi = i; bj = j; }
    const gi = groups[bi], gj = groups[bj], ni = gi.length, nj = gj.length;
    for (let q = 0; q < groups.length; q++) if (q !== bi && q !== bj) { const v = (M[bi][q] * ni + M[bj][q] * nj) / (ni + nj); M[bi][q] = M[q][bi] = v; }
    groups[bi] = gi.concat(gj); groups.splice(bj, 1); M.splice(bj, 1); for (const row of M) row.splice(bj, 1);
  }
  { const dur = groups.map((g) => g.reduce((t, i) => t + W[i], 0)), order = groups.map((_, j) => j).sort((p, q) => dur[q] - dur[p]), keep = order.slice(0, k);
    const merged = keep.map((j) => groups[j].slice());
    for (const j of order.slice(k)) { let best = 0, bv = -Infinity; keep.forEach((q, m) => { const v = M[j][q]; if (v > bv) { bv = v; best = m; } }); merged[best].push(...groups[j]); }
    groups = merged; }
  const lab = new Array(N); groups.forEach((g, j) => g.forEach((i) => (lab[i] = j)));
  for (let it = 0; it < 10; it++) {
    const C = []; for (let j = 0; j < k; j++) { const v = new Array(E[0].length).fill(0); E.forEach((e, i) => { if (lab[i] === j) e.forEach((z, d) => (v[d] += z * W[i])); }); C.push(unit(v)); }
    let moved = 0; E.forEach((e, i) => { let bj = 0, bv = -Infinity; C.forEach((c, j) => { const v = cos(e, c); if (v > bv) { bv = v; bj = j; } }); if (bj !== lab[i]) { lab[i] = bj; moved++; } });
    if (!moved) break;
  }
  // number the speakers in the order they first speak
  const order = []; lab.forEach((l) => { if (!order.includes(l)) order.push(l); });
  return lab.map((l) => order.indexOf(l));
}
