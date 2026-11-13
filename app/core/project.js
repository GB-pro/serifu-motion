// Serifu Motion — project model: speakers, generated base emo, user edits, composition and export.
// All emo tracks are in rig space (K channels of au.js): what the avatar plays is what the timeline shows and exports.
import { IDS, K, KM, IDX, EMOS, CHANNELS } from './au.js';
import { extractRaw, speakerStats, normalize, RAW_DIM, SR, FPS } from './features.js';
import { POSES } from './animate.js';
import { linearToRig } from './generator.js';

export const SPEAKER_COLORS = ['#ff7a59', '#4f8cff', '#2bb673', '#b061ff', '#f5b700', '#ff4f9a'];
export const STYLES = ['anime', 'natural'];
const KEEP_SYNC = new Set(['AU45', 'AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28']); // blink and mouth shapes are never rescaled

export function newSpeaker(i, name, avatar) {
  return { id: 's' + Date.now().toString(36) + i, name, avatar, color: SPEAKER_COLORS[i % SPEAKER_COLORS.length], expressiveness: 1.0, style: 'anime', seed: 7 + i * 101,
    audio16k: null, T: 0, base: null, emo: null, voiced: null, plan: null, raw: null, net: null, edits: {}, emoOverrides: [], visible: ['AU12', 'AU4', 'AU1', 'AU26', 'headPitch', 'headRoll', 'torsoLean'] };
}

// ---------- analysis ----------
export function analyzeSpeaker(sp) {
  const r = extractRaw(sp.audio16k);
  sp.raw = r; sp.T = r.nFrames;
  // speaker normalization uses only this speaker's voiced material, so any language / voice works
  const st = speakerStats([r]);
  sp.stats = st;
  sp.features = normalize(r.raw, r.nFrames, st);
  sp.net = null;
  return sp;
}
// the network's reading of the speech (emotion, accents, mouth): the slow step, done once per recording
export function readSpeaker(sp, gen) { if (!sp.features) analyzeSpeaker(sp); sp.net = gen.runNet(sp.features, sp.T); return sp; }
// the performance built from that reading: fast, repeated after every change of emotion, avatar, style or strength
export function directSpeaker(sp, gen, acting = null) {
  if (!sp.net) readSpeaker(sp, gen);
  const out = gen.direct(sp.net, sp.features, sp.T, { seed: sp.seed, style: sp.style, profile: acting ? acting[sp.avatar] : null, overrides: sp.emoOverrides, exaggeration: sp.expressiveness });
  sp.base = out.base; sp.emo = out.emo; sp.voiced = out.voiced; sp.plan = out.plan;
  return sp;
}
export function generateSpeaker(sp, gen, acting = null) { return directSpeaker(sp, gen, acting); }
// takes the director can re-plan (generated from audio in the anime style); the others get additive corrections
export const isDirected = (sp) => !sp.recorded && sp.style !== 'natural' && !!sp.plan;
// projects saved by the first version kept tracks in tracker space (30 channels)
export function upgradeTrack(base, T, meta) {
  if (base.length === T * K) return base;
  const out = new Float32Array(T * K), rig = new Float32Array(K);
  for (let t = 0; t < T; t++) out.set(linearToRig(base.subarray(t * KM, (t + 1) * KM), meta, rig), t * K);
  return out;
}

// ---------- simple diarization for a mixed dialogue file ----------
export function diarize(x16k, k = 2) {
  const { raw, nFrames } = extractRaw(x16k);
  const E = []; for (let i = 0; i < nFrames; i++) E.push(raw[i * RAW_DIM]);
  const sorted = E.slice().sort((a, b) => a - b);
  const hi = sorted[Math.floor(sorted.length * 0.95)], lo = sorted[Math.floor(sorted.length * 0.1)];
  const thr = Math.max(lo + 8, hi - 28);
  // voice activity with hangover
  const act = new Uint8Array(nFrames);
  for (let i = 0; i < nFrames; i++) act[i] = E[i] > thr ? 1 : 0;
  let segs = [], s = -1, gap = 0;
  for (let i = 0; i <= nFrames; i++) {
    const a = i < nFrames && act[i];
    if (a) { if (s < 0) s = i; gap = 0; } else if (s >= 0) { gap++; if (gap > 9 || i === nFrames) { segs.push([s, i - gap + 1]); s = -1; gap = 0; } }
  }
  segs = segs.filter(([a, b]) => b - a >= 8);
  // split long segments at energy dips so turn changes inside a segment can still be separated
  const fin = [];
  for (const [a, b] of segs) { if (b - a < 90) { fin.push([a, b]); continue; } let st = a; for (let i = a + 45; i < b - 30; i++) { if (E[i] < thr + 4 && i - st > 45) { fin.push([st, i]); st = i; } } fin.push([st, b]); }
  // embedding: mean/std of mfcc + median pitch
  const emb = fin.map(([a, b]) => {
    const v = []; const f0 = [];
    for (let q = 0; q < 12; q++) { let m = 0, c = 0; for (let i = a; i < b; i++) if (E[i] > thr) { m += raw[i * RAW_DIM + 6 + q]; c++; } v.push(c ? m / c : 0); }
    for (let i = a; i < b; i++) { const f = raw[i * RAW_DIM + 1]; if (f > 0) f0.push(12 * Math.log2(f)); }
    f0.sort((p, q) => p - q);
    v.push(f0.length ? f0[Math.floor(f0.length / 2)] * 1.5 : 0);
    let c = 0, m = 0; for (let i = a; i < b; i++) { m += raw[i * RAW_DIM + 4]; c++; } v.push(m / c / 300);
    return v;
  });
  const D = emb[0]?.length || 0;
  const mu = new Array(D).fill(0), sd = new Array(D).fill(0);
  emb.forEach((v) => v.forEach((x, j) => (mu[j] += x / emb.length)));
  emb.forEach((v) => v.forEach((x, j) => (sd[j] += (x - mu[j]) ** 2 / emb.length)));
  const Z = emb.map((v) => v.map((x, j) => (x - mu[j]) / (Math.sqrt(sd[j]) + 1e-6)));
  const W = fin.map(([a, b]) => b - a);
  const lab = kmeans(Z, W, Math.min(k, Z.length));
  return { segments: fin.map(([a, b], i) => ({ start: a / FPS, end: b / FPS, speaker: lab[i] })), nFrames };
}
function kmeans(Z, W, k) {
  if (!Z.length) return [];
  let best = null, bestCost = Infinity;
  for (let rep = 0; rep < 8; rep++) {
    const C = [Z[(rep * 7919) % Z.length].slice()];
    while (C.length < k) { // k-means++ deterministic-ish
      let far = 0, fd = -1; Z.forEach((z, i) => { const d = Math.min(...C.map((c) => dist(z, c))) * W[i]; if (d > fd) { fd = d; far = i; } }); C.push(Z[far].slice());
    }
    let lab = new Array(Z.length).fill(0);
    for (let it = 0; it < 30; it++) {
      lab = Z.map((z) => { let bi = 0, bd = Infinity; C.forEach((c, j) => { const d = dist(z, c); if (d < bd) { bd = d; bi = j; } }); return bi; });
      for (let j = 0; j < k; j++) { const idx = lab.map((l, i) => (l === j ? i : -1)).filter((i) => i >= 0); if (!idx.length) continue; const tw = idx.reduce((s, i) => s + W[i], 0); C[j] = Z[0].map((_, d) => idx.reduce((s, i) => s + Z[i][d] * W[i], 0) / tw); }
    }
    const cost = Z.reduce((s, z, i) => s + dist(z, C[lab[i]]) * W[i], 0);
    if (cost < bestCost) { bestCost = cost; best = lab; }
  }
  return best;
}
const dist = (a, b) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);

// ---------- composition ----------
export function evalKeys(keys, t) {
  if (!keys || !keys.length) return 0;
  if (t <= keys[0].t || t >= keys[keys.length - 1].t) return t === keys[0].t ? keys[0].v : (t === keys[keys.length - 1].t ? keys[keys.length - 1].v : 0);
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i], b = keys[i + 1];
    if (t >= a.t && t <= b.t) { const u = (t - a.t) / (b.t - a.t || 1); const s = u * u * (3 - 2 * u); return a.v + (b.v - a.v) * s; }
  }
  return 0;
}

export class Composer {
  constructor(meta) { this.meta = meta; }
  // emotion ranges on takes the director cannot re-plan (linear style, live recordings): the key pose of the emotion is laid over the take
  emoDelta(sp, t, out) {
    const time = t / FPS;
    for (const ov of sp.emoOverrides) {
      if (time < ov.start - 0.3 || time > ov.end + 0.3) continue;
      const ramp = Math.min(1, (time - (ov.start - 0.3)) / 0.3, ((ov.end + 0.3) - time) / 0.3), u = ramp * ramp * (3 - 2 * ramp);
      const pose = POSES[ov.emo] && POSES[ov.emo][0]; if (!pose) continue;
      for (const id in pose) { const k = IDX[id]; if (k !== undefined) out[k] += u * (ov.intensity ?? 0.8) * pose[id]; }
    }
  }
  // returns Float32Array(K): the final rig-space emo of a speaker at frame t
  frame(sp, t, out = new Float32Array(K)) {
    t = Math.max(0, Math.min(sp.T - 1, t | 0));
    if (!sp.base) { out.fill(0); return out; }
    out.set(sp.base.subarray(t * K, (t + 1) * K));
    if (!isDirected(sp)) {
      const ex = sp.expressiveness ?? 1;
      if (ex !== 1) for (let k = 0; k < K; k++) if (!KEEP_SYNC.has(IDS[k])) out[k] *= ex;
      if (sp.emoOverrides.length) this.emoDelta(sp, t, out);
    }
    const time = t / FPS;
    for (const id in sp.edits) { const k = IDX[id]; if (k !== undefined) out[k] += evalKeys(sp.edits[id], time); }
    for (let k = 0; k < K; k++) { const [lo, hi] = CHANNELS[k].range; out[k] = Math.max(lo < 0 ? lo * 1.3 : 0, Math.min(hi * 1.3, out[k])); }
    return out;
  }
}

// ---------- gestures (inserted as additive key sets) ----------
export const GESTURES = {
  nod: [['headPitch', [[0, 0], [0.1, -3], [0.24, 10], [0.42, -2], [0.58, 5], [0.85, 0]]], ['squash', [[0, 0], [0.12, 0.15], [0.26, -0.25], [0.45, 0.08], [0.7, 0]]]],
  shake: [['headYaw', [[0, 0], [0.15, 12], [0.4, -12], [0.65, 8], [0.9, 0]]], ['torsoYaw', [[0, 0], [0.2, 3], [0.45, -3], [0.7, 2], [0.95, 0]]]],
  browFlash: [['AU1', [[0, 0], [0.1, 0.55], [0.45, 0.5], [0.7, 0]]], ['AU2', [[0, 0], [0.1, 0.65], [0.45, 0.6], [0.7, 0]]], ['AU5', [[0, 0], [0.1, 0.3], [0.4, 0.2], [0.65, 0]]]],
  smile: [['AU12', [[0, 0], [0.3, 0.75], [1.4, 0.7], [1.9, 0]]], ['AU6', [[0, 0], [0.3, 0.5], [1.4, 0.45], [1.9, 0]]], ['headRoll', [[0, 0], [0.35, 4], [1.4, 4], [1.9, 0]]]],
  tilt: [['headRoll', [[0, 0], [0.12, -2], [0.35, 11], [1.3, 10], [1.8, 0]]], ['browAsym', [[0, 0], [0.3, 0.35], [1.3, 0.3], [1.8, 0]]]],
  shrug: [['torsoShrug', [[0, 0], [0.2, -8], [0.6, -7], [0.9, 0]]], ['AU1', [[0, 0], [0.2, 0.35], [0.6, 0.3], [0.9, 0]]], ['AU2', [[0, 0], [0.2, 0.3], [0.6, 0.25], [0.9, 0]]], ['headRoll', [[0, 0], [0.2, 6], [0.6, 5], [0.9, 0]]]],
};
export function insertGesture(sp, name, t0) {
  for (const [id, pts] of GESTURES[name]) {
    const keys = sp.edits[id] || (sp.edits[id] = []);
    for (const [dt, v] of pts) {
      const t = +(t0 + dt).toFixed(3);
      const ex = keys.find((k) => Math.abs(k.t - t) < 0.02);
      if (ex) ex.v += v; else keys.push({ t, v });
    }
    keys.sort((a, b) => a.t - b.t);
  }
}

// ---------- export ----------
export function exportEmo(project, composer) {
  const T = Math.max(...project.speakers.map((s) => s.T), 0);
  return {
    format: 'serifu-motion-emo/2', fps: FPS, channels: IDS, emotions: EMOS,
    channelInfo: CHANNELS.map((c) => ({ id: c.id, group: c.group, range: c.range, name: c.name.en })),
    speakers: project.speakers.map((sp) => {
      const frames = []; const f = new Float32Array(K);
      for (let t = 0; t < sp.T; t++) { composer.frame(sp, t, f); frames.push(Array.from(f, (v) => +v.toFixed(4))); }
      // the acting plan: per phrase the emotion that was played, its strength, the key pose and the accents
      const plan = sp.plan ? sp.plan.phrases.map((ph) => ({ start: +(ph.tp / FPS).toFixed(3), end: +(ph.t1 / FPS).toFixed(3), emotion: ph.emo, blend: ph.emo2, intensity: ph.inten, pose: ph.pose, setByUser: !!ph.fixed, accents: ph.accents.map((a) => ({ time: +(a.t / FPS).toFixed(3), kind: a.kind, key: a.key })) })) : null;
      return { name: sp.name, avatar: sp.avatar, style: sp.recorded ? 'live' : sp.style, frames, plan, emotionOverrides: sp.emoOverrides };
    }),
    duration: T / FPS,
  };
}
export function exportCSV(project, composer) {
  const rows = ['speaker,frame,time,' + IDS.join(',')]; const f = new Float32Array(K);
  for (const sp of project.speakers) for (let t = 0; t < sp.T; t++) { composer.frame(sp, t, f); rows.push(`${sp.name},${t},${(t / FPS).toFixed(3)},` + Array.from(f, (v) => v.toFixed(4)).join(',')); }
  return rows.join('\n');
}
export { SR, FPS };
