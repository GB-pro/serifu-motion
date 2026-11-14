// Serifu Motion — the sculpted head of a character.
//
// The head is the "ICT Face Model Light" of the USC Institute for Creative Technologies (MIT licence, see
// head/LICENSE-ICT-FaceKit.txt): skin of the face, back of the head with neck, inside of the mouth, gums and tongue,
// teeth, eyelashes, and 55 expression shapes (the set that phone face tracking uses: jawOpen, mouthSmile_L, ...).
// tools/avatar/build_base.py packs it into head/base.bin. Here it is given the proportions of a character (the same
// change for the neutral head and for every expression shape), eyes, brows and materials, and it is moved by the
// weights of the expression shapes.
import * as THREE from 'three';
import { Field, polygonise } from './sculpt.js';
import { bakeOcclusion, smoothOcclusion, occluded } from './bake.js';
export { Field };

let BASE = null;
export async function loadHeadBase(dir = new URL('./head/', import.meta.url).href) {
  if (BASE) return BASE;
  const meta = await (await fetch(dir + 'base.json')).json(), buf = await (await fetch(dir + 'base.bin')).arrayBuffer();
  BASE = parseBase(meta, buf); return BASE;
}
export function parseBase(meta, buf) {
  const n = meta.vertices, t = meta.triangles; let o = 0;
  const take = (C, len) => { const a = new C(buf.slice(o, o + len * C.BYTES_PER_ELEMENT)); o += len * C.BYTES_PER_ELEMENT; return a; };
  const pos = take(Float32Array, n * 3), tri = take(Uint32Array, t * 3), region = take(Uint8Array, n), rings = take(Uint8Array, n * 3);
  const shapes = meta.shapes.map((s) => { const idx = take(Uint32Array, s.count), q = take(Int16Array, s.count * 3), d = new Float32Array(s.count * 3), k = s.scale / 32767; for (let i = 0; i < d.length; i++) d[i] = q[i] * k; return { name: s.name, idx, d }; });
  return { meta, n, pos, tri, region, rings, shapes, index: Object.fromEntries(shapes.map((s, i) => [s.name, i])) };
}
export function setHeadBase(b) { BASE = b; }

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const smooth5 = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * t * (t * (t * 6 - 15) + 10); }; // without a change of curvature at the ends
const ramp = (u, a = 1) => 0.5 * (u + Math.sqrt(u * u + a * a)); // u for u >> 0, 0 for u << 0, smooth in between
// Weight of a local scaling by s about a centre, as a function of the distance d from it: 1 up to r0, 0 from r1 on, and
// in between such that the distance after the scaling still grows with d (the skin is pressed together evenly, never
// folded). The distance after the scaling is s d up to r0 and d from r1 on; in the ring between, its slope goes over
// from s to one even value and from that to 1, without a jump of slope or of curvature (a change of curvature alone
// already shows as a line in the shading of the skin). A ring too narrow to take up the growth is made wider.
function radial(d, s, r0, r1) {
  if (d <= r0) return 1;
  if (Math.abs(s - 1) < 0.02) { if (d >= r1) return 0; const u = (d - r0) / (r1 - r0); return 1 - u * u * u * (u * (u * 6 - 15) + 10); }
  const al = 0.25, mMin = 0.3 * (1 - al) + (al * (s + 1)) / 2; // the even slope in the ring is never below 0.3
  if (s > 1) r1 = Math.max(r1, (r0 * (s - mMin)) / (1 - mMin));
  if (d >= r1) return 0;
  const h = r1 - r0, m = (r1 - s * r0) / h, p = (m - (al * (s + 1)) / 2) / (1 - al), t = (d - r0) / h, I = (u) => u * u * u * u * (u * u - 3 * u + 2.5);
  const F = t < al ? s * t + (p - s) * al * I(t / al) : t < 1 - al ? s * al + (p - s) * al * 0.5 + p * (t - al) : s * al + (p - s) * al * 0.5 + p * (t - al) + (1 - p) * al * I((t - 1 + al) / al);
  return Math.min(1, Math.max(0, ((s * r0 + h * F) / d - 1) / (s - 1)));
}

// Proportions of a character: a smooth change of place for every point of the published head (centimetres; x to the
// character's left, y up, z forward), the sum of a few local changes. The neutral head, every expression shape and the
// eyeballs go through the same change, so every expression keeps working: a larger eye still closes fully.
export const DEFAULT_SHAPE = {
  eye: 1, eyeTall: 1, eyeDepth: 1, eyeApart: 0, eyeDown: 0, eyeIn: 0, eyeZone: 3.2, // eyes with their lids: size, height, depth, place (cm); eyeZone: how far the skin around makes room (heads without a form)
  eyeBox: [1.45, 1.7, 0.95, 0.9], eyeReach: [8.2, -1.0, 9.5], // (heads with a form) the box around an eye that grows as a whole: cm from its middle towards the nose, outwards, down, up; and where the skin stops making room: x at the side, y below, y above
  eyeOut: 0, eyeOutR: 2.4, // the eye with its lids brought forward out of its socket (cm), and how wide the piece is that comes along
  skull: 1, skullWide: 1, skullTall: 1, skullDeep: 1, // head above the cheeks
  feature: 1, featureUp: 0, featureEdge: 2.0, // the face below the eyes (nose, mouth, chin, jaw) as one piece: size (it moves towards the eyes as it shrinks), shift (cm)
  bridge: 0, // the bridge of the nose between the eyes set back (cm)
  wide: 1, jaw: 1, chin: 1, cheek: 0, jowl: 0, // width of the outline of the head, of the jaw; chin; fullness of the cheeks in front and of the outline beside the mouth (cm)
  nose: 1, noseWide: 1, mouth: 1, ear: 1, neck: 1, neckTop: -7, // neckTop: the height (cm) below which the neck gets its width
  nostril: 0, // passes that close the nostrils: the skin is drawn flat across them
  neckSmooth: 0, // ironing passes on the neck (sinews, the hollow of the throat)
  reliefSmooth: 0, // passes that even out the relief of the face between its features (only with a form)
  form: null, relief: 0.6, noseOut: 1, // form: a sculpted field (sculpt.js) that the head takes as its smooth form; relief: how much of the height of the face's relief is kept on it; noseOut: how far the nose stands out
  noseForm: null, // a nose of the character's own, sculpted on the form in place of the published one: { r: [wide, tall, deep] of the tip, out, k, bridge: { up, r: [at the tip, at the top], in, k } } (cm)
  mouthCurve: 0, mouthFlat: 0, mouthIn: 0, mouthRelief: 1, // the corners of the mouth higher, and further out of the face (cm: the mouth unwrapped from the muzzle); the mouth set back into the form (cm); how much of the relief of the lips is kept
  calm: 0, // ironing passes on the movement of every expression shape: no folds beside the nose when the character smiles
  smooth: 0, noseSmooth: 0, relax: 0, // ironing passes: fine relief of the published head, the nose, ripples after the change of proportions
};
// ---- the head as a whole: a ball with the face in relief ----
// A drawn face is a simple round form with eyes, nose and mouth standing on it in low relief; a real head is not. So the
// published head is taken apart into its own smooth form (how far the skin is from the middle of the head in every
// direction, without the features) and the relief on top of it. The character gets a form of its own, sculpted from
// soft primitives like the hair, and the relief goes back on, lower than it was.

// a function of direction kept as a table (the poles of the table lie at the ears), read with smooth interpolation
class DirTable {
  constructor(fn, nT = 72, nP = 144) {
    this.nT = nT; this.nP = nP; const v = (this.v = new Float32Array(nT * nP));
    for (let i = 0; i < nT; i++) { const th = (Math.PI * i) / (nT - 1), ct = Math.cos(th), st = Math.sin(th); for (let j = 0; j < nP; j++) { const ph = (2 * Math.PI * j) / nP; v[i * nP + j] = fn(ct, st * Math.sin(ph), st * Math.cos(ph)); } }
  }
  at(ux, uy, uz) {
    const { nT, nP, v } = this, ft = (Math.acos(Math.min(1, Math.max(-1, ux))) / Math.PI) * (nT - 1); let fp = (Math.atan2(uy, uz) / (2 * Math.PI)) * nP; if (fp < 0) fp += nP;
    const i = Math.min(nT - 2, Math.floor(ft)), j = Math.floor(fp) % nP, t = ft - i, u = fp - Math.floor(fp);
    // Catmull-Rom in both directions
    const wt0 = ((-t + 2) * t - 1) * t * 0.5, wt1 = ((3 * t - 5) * t * t + 2) * 0.5, wt2 = ((-3 * t + 4) * t + 1) * t * 0.5, wt3 = (t - 1) * t * t * 0.5;
    const wu0 = ((-u + 2) * u - 1) * u * 0.5, wu1 = ((3 * u - 5) * u * u + 2) * 0.5, wu2 = ((-3 * u + 4) * u + 1) * u * 0.5, wu3 = (u - 1) * u * u * 0.5;
    let r = 0;
    for (let a = -1; a <= 2; a++) {
      const ii = Math.min(nT - 1, Math.max(0, i + a)), o = ii * nP, wt = a === -1 ? wt0 : a === 0 ? wt1 : a === 1 ? wt2 : wt3;
      r += wt * (wu0 * v[o + ((j + nP - 1) % nP)] + wu1 * v[o + j] + wu2 * v[o + ((j + 1) % nP)] + wu3 * v[o + ((j + 2) % nP)]);
    }
    return r;
  }
}
const BALL_C = [0, 2.5, 1.5]; // the middle of the head
const headPlane = (y, z) => y + 4.5 + 0.41 * (z + 5.7); // above 0: head; below: neck (a slanted plane under the chin and through the nape)
// the smooth form of the published head: a low polynomial in the direction, fitted to the skin (ears left out)
function publishedForm(base) {
  if (base._form) return base._form;
  const { n, pos, tri, region } = base, R = base.meta.regions, T = [], deg = 6;
  for (let c = 0; c <= 1; c++) for (let a = 0; a + c <= deg; a += 2) for (let b = 0; a + b + c <= deg; b++) T.push([a, b, c]);
  const m = T.length, M = new Float64Array(m * m), v = new Float64Array(m), f = new Float64Array(m), area = new Float64Array(n);
  for (let t = 0; t < tri.length; t += 3) { const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3, ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2], ar = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 6; area[tri[t]] += ar; area[tri[t + 1]] += ar; area[tri[t + 2]] += ar; }
  const basis = (ux, uy, uz) => { for (let k = 0; k < m; k++) f[k] = ux ** T[k][0] * uy ** T[k][1] * (T[k][2] ? uz : 1); };
  const add = (ux, uy, uz, r, w) => { basis(ux, uy, uz); for (let a = 0; a < m; a++) { v[a] += w * f[a] * r; for (let b = 0; b < m; b++) M[a * m + b] += w * f[a] * f[b]; } };
  const seen = new Uint8Array(24 * 48), cell = (ux, uy, uz) => Math.min(23, Math.floor((Math.acos(Math.max(-1, Math.min(1, uy))) / Math.PI) * 24)) * 48 + Math.min(47, Math.floor(((Math.atan2(ux, uz) + Math.PI) / (2 * Math.PI)) * 48));
  let wsum = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    const r0 = R[region[i]]; if (r0 !== 'face' && r0 !== 'head') continue;
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2]; if (headPlane(y, z) < -2) continue; // with the skin under the chin
    if (Math.abs(x) > 7.1 && y > -2.6 && y < 5 && z < 4.6 && z > -2.4) continue; // ears
    const qx = x - BALL_C[0], qy = y - BALL_C[1], qz = z - BALL_C[2], r = Math.hypot(qx, qy, qz), w = area[i] / (r * r);
    add(qx / r, qy / r, qz / r, r, w); seen[cell(qx / r, qy / r, qz / r)] = 1; wsum += w; cnt++;
  }
  // where there is no skin (downwards, into the neck): an egg, so that the fit stays calm there
  const N = 1500, ga = Math.PI * (3 - Math.sqrt(5)), wp = (wsum / cnt) * 2.5;
  for (let k = 0; k < N; k++) { const uy = 1 - (2 * (k + 0.5)) / N, rr = Math.sqrt(1 - uy * uy), ux = Math.cos(ga * k) * rr, uz = Math.sin(ga * k) * rr; if (seen[cell(ux, uy, uz)]) continue; add(ux, uy, uz, 1 / Math.hypot(ux / 7.6, uy / 10.5, uz / 9.2), wp); }
  for (let a = 0; a < m; a++) M[a * m + a] *= 1 + 1e-10;
  // solve M c = v (Gauss with pivoting)
  const c = Float64Array.from(v);
  for (let k = 0; k < m; k++) {
    let piv = k; for (let i = k + 1; i < m; i++) if (Math.abs(M[i * m + k]) > Math.abs(M[piv * m + k])) piv = i;
    if (piv !== k) { for (let j = 0; j < m; j++) { const t = M[k * m + j]; M[k * m + j] = M[piv * m + j]; M[piv * m + j] = t; } const t = c[k]; c[k] = c[piv]; c[piv] = t; }
    for (let i = k + 1; i < m; i++) { const g = M[i * m + k] / M[k * m + k]; if (!g) continue; for (let j = k; j < m; j++) M[i * m + j] -= g * M[k * m + j]; c[i] -= g * c[k]; }
  }
  for (let k = m - 1; k >= 0; k--) { let t = c[k]; for (let j = k + 1; j < m; j++) t -= M[k * m + j] * c[j]; c[k] = t / M[k * m + k]; }
  const coef = Float64Array.from(c);
  return (base._form = new DirTable((ux, uy, uz) => { let r = 0; for (let k = 0; k < m; k++) r += coef[k] * ux ** T[k][0] * uy ** T[k][1] * (T[k][2] ? uz : 1); return r; }));
}
// the form of a character: how far the surface of a sculpted field is from the middle of the head, in every direction
function fieldForm(field, nT = 72, nP = 144) {
  return new DirTable((ux, uy, uz) => { let lo = 1, hi = 26; for (let k = 0; k < 17; k++) { const t = (lo + hi) / 2; if (field.value(BALL_C[0] + ux * t, BALL_C[1] + uy * t, BALL_C[2] + uz * t) < 0) lo = t; else hi = t; } return (lo + hi) / 2; }, nT, nP);
}

// A change of place along one line that never folds: through the given points [from, to] (both rising), straight in
// between, the corners rounded so that neither the slope nor the curvature jumps. Before the first point and after
// the last one the line only shifts (by what that point shifts).
function along(x, K) {
  const n = K.length; let j = 0; while (j < n && x > K[j][0]) j++; // x lies between the points j - 1 and j
  const sl = (i) => (i <= 0 || i >= n ? 1 : (K[i][1] - K[i - 1][1]) / (K[i][0] - K[i - 1][0])); // the slope before point i
  let X = j === 0 ? x + K[0][1] - K[0][0] : j === n ? x + K[n - 1][1] - K[n - 1][0] : K[j - 1][1] + sl(j) * (x - K[j - 1][0]);
  for (let i = Math.max(0, j - 1); i <= Math.min(n - 1, j); i++) {
    const before = i > 0 ? K[i][0] - K[i - 1][0] : 1e9, after = i < n - 1 ? K[i + 1][0] - K[i][0] : 1e9, dl = Math.min(1, 0.45 * Math.min(before, after)), u = (x - K[i][0] + dl) / (2 * dl);
    if (u > 0 && u < 1) X += (sl(i + 1) - sl(i)) * (2 * dl * u * u * u * u * (u * u - 3 * u + 2.5) - Math.max(0, x - K[i][0]));
  }
  return X;
}

export function makeWarp(base, sh) {
  if (!sh.form) return plainWarp(base, sh);
  const C = BALL_C, ball = { R0: publishedForm(base), R1: fieldForm(sh.form) }, keep0 = sh.relief ?? 0.6;
  // Where things go is changed in steps, one on top of the other. No step folds the skin, so all of them together never
  // do, however much they press it together. A step moves p = [x, y, z] and keeps count of what happens to the relief
  // there: s.g (times), s.h (plus, cm), s.keep (how much of the height of the face's relief is kept).
  const steps = [];
  const step = (mark, make) => { const c = mark.slice(), s = { g: 1, h: 0, keep: 1, deep: 1 }; for (const f of steps) f(c, s); steps.push(make(c)); }; // mark: the published place a step is about, taken along by the steps before it
  // the face below the eyes as one piece (nose, mouth, chin): its size, and how far it sits up towards the eyes
  step([0, 2.2, 9.5], (c) => (p, s) => {
    const wf = radial(Math.hypot((p[0] - c[0]) / 6.5, (p[1] - c[1] + 5.8) / 5.6, (p[2] - c[2]) / 9), sh.feature, 1, sh.featureEdge), k = wf * (sh.feature - 1), up = wf * sh.featureUp * (1 - smooth5(0, 3.2, p[1]));
    p[0] += k * (p[0] - c[0]); p[1] += k * (p[1] - c[1]) + up; p[2] += k * (p[2] - c[2]); s.g *= 1 + k;
  });
  step([0, -7.6, 8.5], (c) => (p, s) => { const ax = p[0] - c[0], ay = p[1] - c[1], az = p[2] - c[2], k = radial(Math.hypot(ax, ay, az), sh.chin, 1.8, 4.2) * (sh.chin - 1); p[0] += k * ax; p[1] += k * ay; p[2] += k * az; s.g *= 1 + k; });
  step([0, -0.2, 11], (c) => (p, s) => {
    const ax = p[0] - c[0], ay = p[1] - c[1], az = p[2] - c[2], w = radial(Math.hypot(ax, ay, az), sh.nose, 2.3, 4.4), k = w * (sh.nose - 1);
    p[0] += k * ax + w * (sh.noseWide - 1) * sh.nose * ax; p[1] += k * ay; p[2] += k * az; if (!sh.noseForm) s.g *= 1 + w * (sh.nose * (sh.noseOut ?? 1) - 1);
  });
  // The published mouth lies around a muzzle: on a rounder form its corners would hang (mouthCurve lifts them) and lie
  // deep in the face (mouthFlat brings everything at a corner out together: lips, teeth and the lining behind).
  step([0, -3.4, 9.8], (c) => (p, s) => {
    const ax = p[0] - c[0], ay = p[1] - c[1], az = p[2] - c[2], w = radial(Math.hypot(ax / 1.3, ay, az / 1.2), sh.mouth, 2.2, 4), k = w * (sh.mouth - 1), t = Math.abs(ax) / 2.4;
    p[0] += k * ax; p[1] += k * ay + w * sh.mouthCurve * Math.min(t, 1.3) ** 2; p[2] += k * az; s.g *= 1 + k;
    s.h += w * (sh.mouthFlat * t * t * (1 - smooth5(1, 1.9, t)) - sh.mouthIn); s.keep *= 1 + w * (sh.mouthRelief - 1); s.deep = 1 - 0.7 * w;
  });
  for (const sg of [-1, 1]) step([sg * 8.1, 1.3, 0.8], (c) => (p, s) => { const ax = p[0] - c[0], ay = p[1] - c[1], az = p[2] - c[2], k = radial(Math.hypot(ax, ay, az), sh.ear, 3, 4.6) * (sh.ear - 1); p[0] += k * ax; p[1] += k * ay; p[2] += k * az; s.g *= 1 + k; });
  // The eyes: the box around each eye with its lids grows as a whole (wider by eye, taller by eye x eyeTall) and moves
  // (eyeApart, eyeDown). The skin between the two boxes, beside, above and below them is pressed together evenly, as
  // far as eyeReach. First along x (the same for both sides), then along y.
  { const e = base.meta.eyes.eyeL.centre, cx = e[0], cy = e[1], B = sh.eyeBox, Rr = sh.eyeReach, sx = sh.eye, sy = sh.eye * sh.eyeTall, ex = cx + sh.eyeApart, ey = cy - sh.eyeDown;
    const X1 = ex - sx * B[0], X2 = ex + sx * B[1], KX = [[-Rr[0], -Rr[0]], [-cx - B[1], -X2], [-cx + B[0], -X1], [cx - B[0], X1], [cx + B[1], X2], [Rr[0], Rr[0]]];
    const KY = [[Rr[1], Rr[1]], [cy - B[2], ey - sy * B[2]], [cy + B[3], ey + sy * B[3]], [Rr[2], Rr[2]]];
    steps.push((p) => { const dy = p[1] - cy; p[0] += (along(p[0], KX) - p[0]) * (1 - smooth5(B[3] + 0.3, Rr[2] - cy, dy)) * (1 - smooth5(B[2] + 0.3, cy - Rr[1], -dy)) * smooth5(-1.5, 2.5, p[2]); });
    steps.push((p) => { const ax = Math.abs(p[0]); p[1] += (along(p[1], KY) - p[1]) * smooth5(0.35 * X1, 0.95 * X1, ax) * (1 - smooth5(X2 + 0.4, Rr[0] - 0.3, ax)) * smooth5(-1.5, 2.5, p[2]); });
  }
  const P = [0, 0, 0], S = { g: 1, h: 0, keep: 1, deep: 1 };
  const W = (x, y, z, o) => {
    P[0] = x; P[1] = y; P[2] = z; S.g = 1; S.h = 0; S.keep = keep0; S.deep = 1; for (const f of steps) f(P, S);
    const wn = smooth5(sh.neckTop, sh.neckTop - (sh.neckFade ?? 3.5), y) * (1 - smooth5(4.5, 8.5, z)), nx = wn * (sh.neck - 1) * x, nz = wn * (sh.neck - 1) * (z + 1); // neck
    // with a nose of the character's own, the published nose gives up its relief (wings, nostrils, bridge) and only keeps its place
    const zN = sh.noseForm ? noseZone(x, y, z) : 0;
    const m = smooth5(-3.4, -0.8, headPlane(y, z)); // 1 on the head, 0 on the neck
    let X = P[0] + nx, Y = P[1], Z = P[2] + nz;
    if (m > 0) {
      // On the head a point's new place only says in which direction from the middle of the head it ends up: things
      // slide over the form. How far out it lies is the form there plus its own relief.
      const qx = x - C[0], qy = y - C[1], qz = z - C[2], r = Math.hypot(qx, qy, qz), ux = qx / r, uy = qy / r, uz = qz / r, b0 = ball.R0.at(ux, uy, uz), d = r - b0;
      const px = P[0] - C[0], py = P[1] - C[1], pz = P[2] - C[2], r2 = Math.hypot(px, py, pz), vx = px / r2, vy = py / r2, vz = pz / r2, b1 = ball.R1.at(vx, vy, vz);
      // the relief of the face is made lower (not what lies deep inside or stands far out, and not the back of the head
      // or the ears). At the mouth what lies behind the lips (S.deep: from how deep on) keeps its distance from them.
      const face = smooth5(0.05, 0.5, uz) * smooth5(0.8, 0.4, uy), mid = d - ramp(d - 1.8, 0.4) + ramp(-d - S.deep, 0.4 * S.deep), fd = d - face * (1 - S.keep) * mid;
      const gN = 1 - zN * smooth5(-0.6, -0.25, d); // (the skin, not the gums and teeth that lie behind it)
      const rr = b1 + (fd * S.g * gN + S.h) * (b1 / b0);
      X += m * (C[0] + vx * rr + nx - X); Y += m * (C[1] + vy * rr - Y); Z += m * (C[2] + vz * rr + nz - Z);
    }
    o[0] = X; o[1] = Y; o[2] = Z; return o;
  };
  W.ball = ball;
  if (sh.noseForm) {
    // the tip of the published nose says where the character's nose goes: there the form gets a rounded tip and a bridge
    // that runs out towards the brow
    const nf = sh.noseForm, o = W(0, 0.1, 13.1, [0, 0, 0]), l = Math.hypot(o[1] - C[1], o[2] - C[2]), a0 = Math.atan2(o[1] - C[1], o[2] - C[2]), r1 = ball.R1.at(0, (o[1] - C[1]) / l, (o[2] - C[2]) / l);
    const at = (up, off) => { const a = a0 + up / r1, sy = Math.sin(a), cz = Math.cos(a), r = ball.R1.at(0, sy, cz) + off; return [0, C[1] + sy * r, C[2] + cz * r]; }; // `up` cm above the tip along the form, `off` cm off it
    const f = new Field(), r = nf.r || [1.7, 1.25, 1.45], out = nf.out ?? 1.2, br = nf.bridge; f.ops = sh.form.ops.slice();
    const tip = at(nf.up ?? 0, out - r[2]);
    if (br !== false) { const b = br || {}, rb = b.r || [0.95, 0.85]; f.cone(at((nf.up ?? 0) + 0.3, out - r[2]), at(b.up ?? 3.2, -(b.in ?? 0.75)), rb[0], rb[1], { k: b.k ?? 1.3 }); }
    f.ell(tip, r, { rot: [-a0 + (nf.tilt ?? 0), 0, 0], k: nf.k ?? 1.0 });
    ball.R1 = fieldForm(f, 144, 288); W.nose = tip;
  }
  return W;
}
// the part of the published head that is its nose (1 inside, 0 outside): from the bridge between the eyes to the wings, not the lip below
const noseZone = (x, y, z) => (1 - smooth5(0.8, 1.15, Math.hypot(x / 2.6, (y - 1.0) / 3.4, (z - 11.6) / 3.2))) * smooth5(-2.2, -1.4, y);

// (heads without a form of their own) every change is a shift of its own and the shifts are added up
function plainWarp(base, sh) {
  const eyes = ['eyeL', 'eyeR'].map((k) => { const e = base.meta.eyes[k]; return { x: e.centre[0], y: e.centre[1], z: e.centre[2] + e.radius, sg: Math.sign(e.centre[0]) }; });
  const sxE = sh.eye - 1, syE = sh.eye * sh.eyeTall - 1, szE = sh.eyeDepth - 1, skx = sh.skull * sh.skullWide - 1, sky = sh.skull * sh.skullTall - 1, skz = sh.skull * sh.skullDeep - 1;
  return (x, y, z, o) => {
    let dx = 0, dy = 0, dz = 0, w;
    for (const e of eyes) { // the eye with its lids grows about the front of the eyeball; the skin around makes room over a wide area
      const ex = x - e.x, ey = y - e.y, ez = z - e.z, u2 = (ex * ex + ey * ey + 0.3 * ez * ez) / (sh.eyeZone * sh.eyeZone); w = 1 / ((1 + u2) * (1 + u2));
      dx += w * (sxE * ex + e.sg * sh.eyeApart); dy += w * (syE * ey - sh.eyeDown); dz += w * (szE * ez - sh.eyeIn);
      if (sh.eyeOut) dz += sh.eyeOut * Math.exp(-(ex * ex + ey * ey + 0.5 * ez * ez) / (sh.eyeOutR * sh.eyeOutR));
    }
    w = smooth(1, 9, y); dx += w * skx * x; dy += w * sky * (y - 3); dz += w * skz * (z - 0.5); // skull
    // the face below the eyes as one piece, the same in every direction so that its forms and the expressions keep their
    // look; the piece hangs from the bridge of the nose, the skin around it stretches
    const wf = radial(Math.hypot(x / 6.5, (y + 3.6) / 5.6, (z - 9.5) / 9), sh.feature, 1, sh.featureEdge);
    { const k = wf * (sh.feature - 1); dx += k * x; dy += k * (y - 2.2) + wf * sh.featureUp; dz += k * (z - 9.5); }
    if (sh.bridge) dz -= sh.bridge * Math.exp(-(x * x * 1.6 + (y - 3.0) * (y - 3.0) * 0.5 + (z - 10.8) * (z - 10.8) * 0.4) / 2.2);
    const below = 1 - smooth(-8.5, -12, y); // not the base of the neck
    dx += below * (1 - wf) * (sh.wide - 1) * x; // the outline of the head wider, the features in the middle as they are
    w = smooth(1.5, -6.5, y) * (1 - smooth(-8.5, -11.5, y)) * (1 - wf); dx += w * (sh.jaw - 1) * x; // jaw
    if (sh.jowl) { const cy = y + 3.6, cz = z - 4.5, r = (cy * cy) / 14 + (cz * cz) / 34, gl = Math.exp(-((x - 6) * (x - 6)) / 9 - r), gr = Math.exp(-((x + 6) * (x + 6)) / 9 - r); dx += (gl - gr) * sh.jowl; dy -= (gl + gr) * sh.jowl * 0.15; }
    { const cx = x, cy = y + 7.6, cz = z - 8.5; const k = radial(Math.hypot(cx, cy, cz), sh.chin, 1.8, 4.2) * (sh.chin - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
    { const cx = x, cy = y + 0.2, cz = z - 11; w = radial(Math.hypot(cx, cy, cz), sh.nose, 2.3, 4.4); const k = w * (sh.nose - 1); dx += k * cx + w * (sh.noseWide - 1) * sh.nose * cx; dy += k * cy; dz += k * cz; }
    { const cx = x, cy = y + 3.4, cz = z - 9.8; const k = radial(Math.hypot(cx / 1.3, cy, cz / 1.2), sh.mouth, 2.2, 4) * (sh.mouth - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
    if (sh.cheek) { const cy = y + 1.2, cz = z - 8.8, r = cy * cy + cz * cz, gl = Math.exp(-((x - 4.3) * (x - 4.3) + r) / 11.5) * sh.cheek, gr = Math.exp(-((x + 4.3) * (x + 4.3) + r) / 11.5) * sh.cheek; dx += (gl - gr) * 0.8; dz += (gl + gr) * 0.6; }
    const wn = smooth5(sh.neckTop, sh.neckTop - (sh.neckFade ?? 3.5), y) * (1 - smooth5(4.5, 8.5, z)); dx += wn * (sh.neck - 1) * x; dz += wn * (sh.neck - 1) * (z + 1); // neck
    for (const sg of [-1, 1]) { const cx = x - sg * 8.1, cy = y - 1.3, cz = z - 0.8, k = radial(Math.hypot(cx, cy, cz), sh.ear, 3, 4.6) * (sh.ear - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
    o[0] = x + dx; o[1] = y + dy; o[2] = z + dz; return o;
  };
}

// The skin of the head (face and back of the head) as a surface on which things can be placed: brows, hair, a beard.
// A thing is described on the published head; what is stored for each of its points is the triangle of the skin it
// sits on and how far it stands off. It then follows the skin through the character's proportions and every expression.
class Skin {
  constructor(P, tri, t0, t1) {
    this.P = P; this.tri = tri; const C = 0.5, X0 = -14, Y0 = -21, NX = 56, NY = 74;
    this.g2 = { C, X0, Y0, NX, NY, cells: Array.from({ length: NX * NY }, () => []) };
    const D = 1.5, Z0 = -11, MX = 19, MY = 25, MZ = 17; this.g3 = { D, X0, Y0, Z0, MX, MY, MZ, cells: new Map() };
    for (let t = t0; t < t1; t++) {
      const a = tri[t * 3] * 3, b = tri[t * 3 + 1] * 3, c = tri[t * 3 + 2] * 3;
      const lox = Math.min(P[a], P[b], P[c]), hix = Math.max(P[a], P[b], P[c]), loy = Math.min(P[a + 1], P[b + 1], P[c + 1]), hiy = Math.max(P[a + 1], P[b + 1], P[c + 1]), loz = Math.min(P[a + 2], P[b + 2], P[c + 2]), hiz = Math.max(P[a + 2], P[b + 2], P[c + 2]);
      const nz = (P[b] - P[a]) * (P[c + 1] - P[a + 1]) - (P[b + 1] - P[a + 1]) * (P[c] - P[a]);
      if (nz > 0) for (let i = Math.max(0, Math.floor((lox - X0) / C)); i <= Math.min(NX - 1, Math.floor((hix - X0) / C)); i++) for (let j = Math.max(0, Math.floor((loy - Y0) / C)); j <= Math.min(NY - 1, Math.floor((hiy - Y0) / C)); j++) this.g2.cells[j * NX + i].push(t);
      for (let i = Math.floor((lox - X0) / D); i <= Math.floor((hix - X0) / D); i++) for (let j = Math.floor((loy - Y0) / D); j <= Math.floor((hiy - Y0) / D); j++) for (let k = Math.floor((loz - Z0) / D); k <= Math.floor((hiz - Z0) / D); k++) { const key = (i * 64 + j) * 64 + k; let l = this.g3.cells.get(key); if (!l) this.g3.cells.set(key, (l = [])); l.push(t); }
    }
  }
  // the skin seen from the front at (x, y): { t, b: [b0, b1, b2], z } or null
  front(x, y) {
    const { C, X0, Y0, NX, NY, cells } = this.g2, i = Math.floor((x - X0) / C), j = Math.floor((y - Y0) / C); if (i < 0 || j < 0 || i >= NX || j >= NY) return null;
    const P = this.P, tri = this.tri; let best = null;
    for (const t of cells[j * NX + i]) {
      const a = tri[t * 3] * 3, b = tri[t * 3 + 1] * 3, c = tri[t * 3 + 2] * 3, d = (P[b] - P[a]) * (P[c + 1] - P[a + 1]) - (P[b + 1] - P[a + 1]) * (P[c] - P[a]);
      const b1 = ((x - P[a]) * (P[c + 1] - P[a + 1]) - (y - P[a + 1]) * (P[c] - P[a])) / d, b2 = ((P[b] - P[a]) * (y - P[a + 1]) - (P[b + 1] - P[a + 1]) * (x - P[a])) / d, b0 = 1 - b1 - b2;
      if (b0 < -1e-4 || b1 < -1e-4 || b2 < -1e-4) continue;
      const z = b0 * P[a + 2] + b1 * P[b + 2] + b2 * P[c + 2]; if (!best || z > best.z) best = { t, b: [b0, b1, b2], z };
    }
    return best;
  }
  // the point of the skin nearest to (x, y, z): { t, b, d2 }
  near(x, y, z) {
    const { D, X0, Y0, Z0, cells } = this.g3, ci = Math.floor((x - X0) / D), cj = Math.floor((y - Y0) / D), ck = Math.floor((z - Z0) / D); let best = null;
    for (let r = 0; r < 12; r++) {
      if (best && best.d2 < ((r - 1) * D) ** 2) break;
      for (let i = ci - r; i <= ci + r; i++) for (let j = cj - r; j <= cj + r; j++) for (let k = ck - r; k <= ck + r; k++) {
        if (Math.max(Math.abs(i - ci), Math.abs(j - cj), Math.abs(k - ck)) !== r) continue;
        const l = cells.get((i * 64 + j) * 64 + k); if (!l) continue;
        for (const t of l) { const q = this._closest(t, x, y, z); if (!best || q.d2 < best.d2) best = q; }
      }
    }
    return best;
  }
  _closest(t, x, y, z) { // closest point of triangle t, as weights of its corners
    const P = this.P, tri = this.tri, A = tri[t * 3] * 3, B = tri[t * 3 + 1] * 3, Cc = tri[t * 3 + 2] * 3;
    const abx = P[B] - P[A], aby = P[B + 1] - P[A + 1], abz = P[B + 2] - P[A + 2], acx = P[Cc] - P[A], acy = P[Cc + 1] - P[A + 1], acz = P[Cc + 2] - P[A + 2];
    const apx = x - P[A], apy = y - P[A + 1], apz = z - P[A + 2];
    const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz; let b;
    const bpx = x - P[B], bpy = y - P[B + 1], bpz = z - P[B + 2], d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = x - P[Cc], cpy = y - P[Cc + 1], cpz = z - P[Cc + 2], d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d1 <= 0 && d2 <= 0) b = [1, 0, 0];
    else if (d3 >= 0 && d4 <= d3) b = [0, 1, 0];
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); b = [1 - v, v, 0]; }
    else if (d6 >= 0 && d5 <= d6) b = [0, 0, 1];
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); b = [1 - w, 0, w]; }
    else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / (d4 - d3 + (d5 - d6)); b = [0, 1 - w, w]; }
    else { const dn = 1 / (va + vb + vc); b = [1 - vb * dn - vc * dn, vb * dn, vc * dn]; }
    const qx = b[0] * P[A] + b[1] * P[B] + b[2] * P[Cc] - x, qy = b[0] * P[A + 1] + b[1] * P[B + 1] + b[2] * P[Cc + 1] - y, qz = b[0] * P[A + 2] + b[1] * P[B + 2] + b[2] * P[Cc + 2] - z;
    return { t, b, d2: qx * qx + qy * qy + qz * qz };
  }
}

// A small mesh whose vertices are bound to the skin: place on a triangle and distance from it (along the normal).
class Bound {
  constructor(head, hits, heights, index, material) {
    const m = hits.length;
    this.head = head; this.tv = new Int32Array(m * 3); this.bw = new Float32Array(m * 3); this.h = new Float32Array(m);
    this.bind(hits, heights);
    this.pos = new Float32Array(m * 3); this.nrm = new Float32Array(m * 3); this.index = index;
    const g = (this.geometry = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('normal', new THREE.BufferAttribute(this.nrm, 3).setUsage(THREE.DynamicDrawUsage));
    g.setIndex(new THREE.BufferAttribute(index, 1));
    this.mesh = new THREE.Mesh(g, material); this.mesh.frustumCulled = false; this.mesh.castShadow = true; this.mesh.receiveShadow = true;
  }
  // (again) the places on the skin: a thing that slides over the skin is bound anew whenever it has moved
  bind(hits, heights) {
    const head = this.head, tri = head.base.tri, src = head.src, P = head.rest, sc = head.grow;
    for (let i = 0; i < hits.length; i++) {
      const { t, b } = hits[i], a0 = tri[t * 3], a1 = tri[t * 3 + 1], a2 = tri[t * 3 + 2];
      this.tv[i * 3] = a0; this.tv[i * 3 + 1] = a1; this.tv[i * 3 + 2] = a2; this.bw[i * 3] = b[0]; this.bw[i * 3 + 1] = b[1]; this.bw[i * 3 + 2] = b[2];
      this.h[i] = heights[i] * (b[0] * sc[a0] + b[1] * sc[a1] + b[2] * sc[a2]); // the distance grows with the skin
    }
  }
  update() {
    const { tv, bw, h, pos, nrm, index } = this, C = this.head.cur, N = this.head.normal, m = h.length;
    for (let i = 0; i < m; i++) {
      const a = tv[i * 3] * 3, b = tv[i * 3 + 1] * 3, c = tv[i * 3 + 2] * 3, wa = bw[i * 3], wb = bw[i * 3 + 1], wc = bw[i * 3 + 2];
      let nx = wa * N[a] + wb * N[b] + wc * N[c], ny = wa * N[a + 1] + wb * N[b + 1] + wc * N[c + 1], nz = wa * N[a + 2] + wb * N[b + 2] + wc * N[c + 2]; const l = h[i] / (Math.hypot(nx, ny, nz) || 1);
      pos[i * 3] = wa * C[a] + wb * C[b] + wc * C[c] + nx * l; pos[i * 3 + 1] = wa * C[a + 1] + wb * C[b + 1] + wc * C[c + 1] + ny * l; pos[i * 3 + 2] = wa * C[a + 2] + wb * C[b + 2] + wc * C[c + 2] + nz * l;
    }
    nrm.fill(0);
    for (let t = 0; t < index.length; t += 3) {
      const a = index[t] * 3, b = index[t + 1] * 3, c = index[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      nrm[a] += nx; nrm[a + 1] += ny; nrm[a + 2] += nz; nrm[b] += nx; nrm[b + 1] += ny; nrm[b + 2] += nz; nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
    }
    for (let i = 0; i < nrm.length; i += 3) { const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1; nrm[i] /= l; nrm[i + 1] /= l; nrm[i + 2] /= l; }
    this.geometry.attributes.position.needsUpdate = true; this.geometry.attributes.normal.needsUpdate = true;
  }
}

// A thin rounded line that lies on the skin and follows it (the lash line of a lid): a chain of points bound to the
// skin, each with a radius.
class Tube {
  constructor(head, hits, radii, material, sides = 6) {
    const m = hits.length, tri = head.base.tri; this.head = head; this.m = m; this.sides = sides; this.r = Float32Array.from(radii);
    this.tv = new Int32Array(m * 3); this.bw = new Float32Array(m * 3); this.c = new Float32Array(m * 3); this.cn = new Float32Array(m * 3);
    for (let i = 0; i < m; i++) { const { t, b } = hits[i]; this.tv[i * 3] = tri[t * 3]; this.tv[i * 3 + 1] = tri[t * 3 + 1]; this.tv[i * 3 + 2] = tri[t * 3 + 2]; this.bw.set(b, i * 3); }
    this.pos = new Float32Array(m * sides * 3); this.nrm = new Float32Array(m * sides * 3); const idx = [];
    for (let i = 0; i < m - 1; i++) for (let k = 0; k < sides; k++) { const a = i * sides + k, b = i * sides + ((k + 1) % sides), c = a + sides, d = b + sides; idx.push(a, b, c, b, d, c); }
    const g = (this.geometry = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('normal', new THREE.BufferAttribute(this.nrm, 3).setUsage(THREE.DynamicDrawUsage));
    g.setIndex(idx); this.mesh = new THREE.Mesh(g, material); this.mesh.frustumCulled = false; this.mesh.castShadow = false; this.mesh.receiveShadow = true;
  }
  update() {
    const { tv, bw, c, cn, r, m, sides, pos, nrm } = this, C = this.head.cur, N = this.head.normal;
    for (let i = 0; i < m; i++) {
      const a = tv[i * 3] * 3, b = tv[i * 3 + 1] * 3, d = tv[i * 3 + 2] * 3, wa = bw[i * 3], wb = bw[i * 3 + 1], wc = bw[i * 3 + 2];
      let nx = wa * N[a] + wb * N[b] + wc * N[d], ny = wa * N[a + 1] + wb * N[b + 1] + wc * N[d + 1], nz = wa * N[a + 2] + wb * N[b + 2] + wc * N[d + 2]; const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
      const h = r[i] * 0.35; cn[i * 3] = nx; cn[i * 3 + 1] = ny; cn[i * 3 + 2] = nz;
      c[i * 3] = wa * C[a] + wb * C[b] + wc * C[d] + nx * h; c[i * 3 + 1] = wa * C[a + 1] + wb * C[b + 1] + wc * C[d + 1] + ny * h; c[i * 3 + 2] = wa * C[a + 2] + wb * C[b + 2] + wc * C[d + 2] + nz * h;
    }
    for (let i = 0; i < m; i++) {
      const p = Math.max(0, i - 1) * 3, q = Math.min(m - 1, i + 1) * 3, o = i * 3;
      let tx = c[q] - c[p], ty = c[q + 1] - c[p + 1], tz = c[q + 2] - c[p + 2]; const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
      let nx = cn[o], ny = cn[o + 1], nz = cn[o + 2]; const dn = nx * tx + ny * ty + nz * tz; nx -= dn * tx; ny -= dn * ty; nz -= dn * tz; const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
      const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
      for (let k = 0; k < sides; k++) { const a = (k / sides) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a), j = (i * sides + k) * 3, dx = ca * nx + sa * bx, dy = ca * ny + sa * by, dz = ca * nz + sa * bz;
        nrm[j] = dx; nrm[j + 1] = dy; nrm[j + 2] = dz; pos[j] = c[o] + dx * r[i]; pos[j + 1] = c[o + 1] + dy * r[i]; pos[j + 2] = c[o + 2] + dz * r[i]; }
    }
    this.geometry.attributes.position.needsUpdate = true; this.geometry.attributes.normal.needsUpdate = true;
  }
}

const REGION_COLOR = { mouth: '#5a1f24', socketL: '#6b3a36', socketR: '#6b3a36', gums: '#d9717c', teeth: '#f4f1ea', lashL: '#1d1512', lashR: '#1d1512' };

export class Head {
  // design: { shape: {...DEFAULT_SHAPE}, pose: { shape name: weight of the rest face }, skin, lip, lid, lipRings, lipAmount, iris, irisSize, pupilSize, hide: [regions] }
  constructor(design = {}, base = BASE) {
    if (!base) throw new Error('loadHeadBase() first');
    this.base = base; this.design = design;
    const sh = (this.shape = { ...DEFAULT_SHAPE, ...(design.shape || {}) }), n = base.n;
    this._weld();
    // the published head with its fine relief ironed out (folds beside the nose, bags under the eyes; less on the ears), the nose more
    const src = (this.src = Float32Array.from(base.pos));
    if (sh.smooth > 0) this._iron(src, sh.smooth, (x, y, z) => 1 - 0.6 * smooth(6.6, 7.6, Math.abs(x)) * (1 - smooth(4, 5, z)));
    if (sh.neckSmooth > 0) this._iron(src, sh.neckSmooth, (x, y, z) => smooth(1.5, -1.5, headPlane(y, z)));
    if (sh.nostril > 0) this._iron(src, sh.nostril, (x, y, z) => 1 - smooth(0.8, 1.1, Math.hypot(Math.abs(x) - 0.8, y - 0.15, z - 10.85)), true);
    if (sh.noseSmooth > 0) this._iron(src, sh.noseSmooth, (x, y, z) => 1 - smooth(2.2, 3.6, Math.hypot(x, y + 0.2, z - 11.3)));
    if (sh.noseForm) this._iron(src, sh.noseFilm ?? 150, noseZone, true); // with a nose of the character's own: the published one is drawn flat like a film, so that its skin spreads evenly over the new one
    const W = (this.warp = makeWarp(base, sh)), o = [0, 0, 0], q = [0, 0, 0];
    const P = (this.rest = new Float32Array(n * 3));
    for (let i = 0; i < n; i++) { W(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], o); P[i * 3] = o[0]; P[i * 3 + 1] = o[1]; P[i * 3 + 2] = o[2]; }
    // An expression shape moves a vertex of the published head. Around the eyes the character's vertex goes exactly where
    // that moved point goes, so the lids still meet on a larger eye. Elsewhere the movement keeps its direction and is
    // only made larger or smaller with the skin there (the same in every direction), so a smile keeps its curve on a
    // face that is wider and shorter than the published one, and no folds appear where the proportions change.
    const iso = new Float32Array(n), exact = new Float32Array(n), E = base.meta.eyes, e1 = [0, 0, 0], e2 = [0, 0, 0], e3 = [0, 0, 0], h = 0.05;
    for (let i = 0; i < n; i++) {
      const x = src[i * 3], y = src[i * 3 + 1], z = src[i * 3 + 2]; W(x, y, z, o); W(x + h, y, z, e1); W(x, y + h, z, e2); W(x, y, z + h, e3);
      const a = (e1[0] - o[0]) / h, b = (e2[0] - o[0]) / h, c = (e3[0] - o[0]) / h, d = (e1[1] - o[1]) / h, e = (e2[1] - o[1]) / h, f = (e3[1] - o[1]) / h, g = (e1[2] - o[2]) / h, k = (e2[2] - o[2]) / h, l = (e3[2] - o[2]) / h;
      iso[i] = Math.cbrt(Math.abs(a * (e * l - f * k) - b * (d * l - f * g) + c * (d * k - e * g)));
      const ce = x > 0 ? E.eyeL : E.eyeR, ex = x - ce.centre[0], ey = y - ce.centre[1], ez = z - ce.centre[2] - ce.radius;
      exact[i] = 1 - smooth(1.9, 3.4, Math.hypot(ex, ey, ez * 0.55));
    }
    const P0 = sh.relax > 0 ? Float32Array.from(P) : P; // where the change of proportions alone puts each vertex
    // Where the change of proportions has pressed the skin together, the gentle slopes of the published face have become
    // steep and read as creases (under a much larger eye, beside the nose). Ironing takes them out: more where the skin
    // was pressed more.
    if (sh.reliefSmooth > 0 && W.ball) this._calmRelief(P, W.ball.R1, sh.reliefSmooth);
    if (sh.relax > 0) this._iron(P, sh.relax, (x, y, z, i) => 0.3 + 0.7 * smooth(0.95, 0.62, iso[i]));
    this.cur = Float32Array.from(P);
    const { grow, turn } = this._follow(src, P); this.grow = grow; // how much the skin has grown at each vertex
    this.shapes = base.shapes.map((s) => {
      const d = new Float32Array(s.d.length);
      for (let j = 0; j < s.idx.length; j++) {
        const v = s.idx[j], i = v * 3, k = j * 3, we = exact[v], wi = (1 - we) * grow[v];
        let ax = s.d[k], ay = s.d[k + 1], az = s.d[k + 2]; const rx = turn[i], ry = turn[i + 1], rz = turn[i + 2], th = Math.hypot(rx, ry, rz);
        if (th > 1e-4) { // turned about the axis (rx, ry, rz) by its length
          const kx = rx / th, ky = ry / th, kz = rz / th, c = Math.cos(th), sn = Math.sin(th), dot = (kx * ax + ky * ay + kz * az) * (1 - c);
          const bx = ax * c + (ky * az - kz * ay) * sn + kx * dot, by = ay * c + (kz * ax - kx * az) * sn + ky * dot, bz = az * c + (kx * ay - ky * ax) * sn + kz * dot; ax = bx; ay = by; az = bz;
        }
        d[k] = wi * ax; d[k + 1] = wi * ay; d[k + 2] = wi * az;
        if (we > 0) { W(src[i] + s.d[k], src[i + 1] + s.d[k + 1], src[i + 2] + s.d[k + 2], q); d[k] += we * (q[0] - P0[i]); d[k + 1] += we * (q[1] - P0[i + 1]); d[k + 2] += we * (q[2] - P0[i + 2]); }
      }
      return { name: s.name, idx: s.idx, d };
    });
    if (sh.noseForm) this._noseFollows();
    if (sh.calm > 0) this._calm(sh.calm);
    this.w = new Float32Array(this.shapes.length); this._lastW = new Float32Array(this.shapes.length).fill(NaN);
    if (design.pose) { this.restPose = new Float32Array(this.shapes.length); for (const k in design.pose) { const i = base.index[k]; if (i !== undefined) this.restPose[i] = design.pose[k]; } this.w.set(this.restPose); }
    const g = (this.geometry = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.cur, 3).setUsage(THREE.DynamicDrawUsage));
    this.normal = new Float32Array(n * 3); g.setAttribute('normal', new THREE.BufferAttribute(this.normal, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(this._colors(), 3)); g.setAttribute('ao', new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    // Drawn teeth are one smooth band above and one below (see _teeth): the published teeth are not shown, and of the gums
    // and tongue only the tongue.
    const bands = design.teeth !== 'real', R = base.meta.regions, G = base.meta.groups, hide = design.hide || (bands ? ['lashL', 'lashR', 'teeth'] : ['lashL', 'lashR']); this.materials = this._materials(R);
    { const T = base.tri, Q = base.pos, keep = [], cut = design.cut ? (v) => design.cut(Q[v * 3], Q[v * 3 + 1], Q[v * 3 + 2]) : null; // cut(x, y, z) on the published head: skin the character does not show (the bust under the clothes)
      const tongue = (v) => Q[v * 3 + 1] < -2.95 && (Q[v * 3] / 2.05) ** 2 + ((Q[v * 3 + 2] - 3.8) / 5.75) ** 2 < 1;
      R.forEach((r, i) => { const at = keep.length; if (!hide.includes(r)) for (let t = G[i][0]; t < G[i][0] + G[i][1]; t++) { const a = T[t * 3], b = T[t * 3 + 1], c = T[t * 3 + 2];
        if (i <= 1 ? !(cut && cut(a) && cut(b) && cut(c)) : r !== 'gums' || !bands || (tongue(a) && tongue(b) && tongue(c))) keep.push(a, b, c); } g.addGroup(at, keep.length - at, i); });
      g.setIndex(new THREE.BufferAttribute(Uint32Array.from(keep), 1)); }
    this.mesh = new THREE.Mesh(g, this.materials); this.mesh.frustumCulled = false; this.mesh.castShadow = true; this.mesh.receiveShadow = true;
    this.root = new THREE.Group(); this.root.add(this.mesh);
    this.eyes = [0, 1].map((e) => this._eye(e)); this.eyes.forEach((e) => this.root.add(e.mesh));
    if (design.glint !== false) for (const e of [0, 1]) { this.root.add(this._glint(e, 13, 9, 5.5)); this.root.add(this._glint(e, -12, -11, 2.4)); }
    this.bound = []; this.sculpts = [];
    if (design.lash !== false) for (const e of [0, 1]) { const l = this._lash(e, design.lash || {}); this.bound.push(l); this.root.add(l.mesh); }
    if (bands) this._teeth(design.teethShape || {});
    this.brows = [];
    if (design.brow !== false) for (const side of [1, -1]) { const b = this._brow(side, design.brow || {}); this.bound.push(b); this.brows.push(b); this.root.add(b.mesh); }
    this.look(0, 0); this.update(true);
  }
  // vertices of different parts that sit at the same place (lip line, lid rim) share their normal, so no seam shows
  _weld() {
    const { n } = this.base, P = this.base.pos, key = new Map(), first = new Int32Array(n);
    for (let i = 0; i < n; i++) { const k = `${Math.round(P[i * 3] * 2000)},${Math.round(P[i * 3 + 1] * 2000)},${Math.round(P[i * 3 + 2] * 2000)}`; const f = key.get(k); if (f === undefined) { key.set(k, i); first[i] = i; } else first[i] = f; }
    this.first = first; this.dup = []; for (let i = 0; i < n; i++) if (first[i] !== i) this.dup.push(i);
  }
  // Ironing the movement of the expression shapes over the skin: what each vertex moves is averaged with what its
  // neighbours move. The large movement stays; the sharp folds a real face makes (beside the nose in a smile) go.
  // The rims of the lids and the lip line keep their own movement, so eyes and mouth still close.
  _calm(passes) {
    const n = this.base.n, first = this.first; this._iron(new Float32Array(0), 0, () => 1); const nb = this._nb, keep = this._keep;
    const D = new Float32Array(n * 3), Q = new Float32Array(n * 3), on = new Uint8Array(n);
    for (const s of this.shapes) {
      D.fill(0); on.fill(0); const { idx, d } = s;
      for (let j = 0; j < idx.length; j++) { const i = idx[j] * 3; D[i] = d[j * 3]; D[i + 1] = d[j * 3 + 1]; D[i + 2] = d[j * 3 + 2]; on[first[idx[j]]] = 1; }
      const list = []; for (let j = 0; j < idx.length; j++) { const v = idx[j]; if (first[v] === v && keep[v] > 0) list.push(v); }
      for (let p = 0; p < passes; p++) {
        for (const v of list) { const a = nb[v]; let x = 0, y = 0, z = 0; for (let k = 0; k < a.length; k++) { const u = a[k] * 3; x += D[u]; y += D[u + 1]; z += D[u + 2]; }
          const g = (0.6 * keep[v]) / a.length, o = v * 3; Q[o] = D[o] + g * (x - a.length * D[o]); Q[o + 1] = D[o + 1] + g * (y - a.length * D[o + 1]); Q[o + 2] = D[o + 2] + g * (z - a.length * D[o + 2]); }
        for (const v of list) { const o = v * 3; D[o] = Q[o]; D[o + 1] = Q[o + 1]; D[o + 2] = Q[o + 2]; }
      }
      for (let j = 0; j < idx.length; j++) { const f = first[idx[j]] * 3; d[j * 3] = D[f]; d[j * 3 + 1] = D[f + 1]; d[j * 3 + 2] = D[f + 2]; }
    }
  }
  // With a nose of the character's own, the skin of the published nose lies on it like a film, and what the expression
  // shapes move there (wings, nostrils) no longer fits together. So the nose goes along with the skin around it: inside
  // the nose every shape moves the way its rim does (the plain trend of the movement across the rim takes the place of
  // the movement itself).
  _noseFollows() {
    const { n, pos, region } = this.base, zone = [], rim = [], slot = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) { if (region[i] > 1) continue; const w = noseZone(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]); if (w > 0.02) { zone.push(i, w); if (w < 0.6) rim.push(i); } }
    // the least-squares plane a + b x + c y through values given at the rim: the matrix is the same for every shape
    let s1 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (const i of rim) { const x = pos[i * 3], y = pos[i * 3 + 1] - 1; s1++; sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y; }
    const det = s1 * (sxx * syy - sxy * sxy) - sx * (sx * syy - sxy * sy) + sy * (sx * sxy - sxx * sy);
    const inv = [sxx * syy - sxy * sxy, -(sx * syy - sxy * sy), sx * sxy - sxx * sy, s1 * syy - sy * sy, -(s1 * sxy - sx * sy), s1 * sxx - sx * sx].map((v) => v / det); // symmetric: [00, 01, 02, 11, 12, 22]
    for (const s of this.shapes) {
      const { idx, d } = s; for (let j = 0; j < idx.length; j++) slot[idx[j]] = j;
      const r = new Float64Array(9); let any = false; // right-hand sides for the three directions
      for (const i of rim) { const j = slot[i]; if (j < 0) continue; any = true; const x = pos[i * 3], y = pos[i * 3 + 1] - 1; for (let k = 0; k < 3; k++) { const v = d[j * 3 + k]; r[k * 3] += v; r[k * 3 + 1] += v * x; r[k * 3 + 2] += v * y; } }
      if (!any) { let moves = false; for (let q = 0; q < zone.length; q += 2) if (slot[zone[q]] >= 0) { moves = true; break; } if (!moves) { for (let j = 0; j < idx.length; j++) slot[idx[j]] = -1; continue; } }
      const co = new Float64Array(9); for (let k = 0; k < 3; k++) { const a = r[k * 3], b = r[k * 3 + 1], c = r[k * 3 + 2]; co[k * 3] = inv[0] * a + inv[1] * b + inv[2] * c; co[k * 3 + 1] = inv[1] * a + inv[3] * b + inv[4] * c; co[k * 3 + 2] = inv[2] * a + inv[4] * b + inv[5] * c; }
      const moreI = [], moreD = [];
      for (let q = 0; q < zone.length; q += 2) {
        const i = zone[q], w = zone[q + 1], j = slot[i], x = pos[i * 3], y = pos[i * 3 + 1] - 1, f = [0, 0, 0];
        for (let k = 0; k < 3; k++) { const fit = co[k * 3] + co[k * 3 + 1] * x + co[k * 3 + 2] * y, d0 = j < 0 ? 0 : d[j * 3 + k]; f[k] = d0 + w * (fit - d0); }
        if (j >= 0) { d[j * 3] = f[0]; d[j * 3 + 1] = f[1]; d[j * 3 + 2] = f[2]; } else if (Math.abs(f[0]) + Math.abs(f[1]) + Math.abs(f[2]) > 2e-4) { moreI.push(i); moreD.push(f[0], f[1], f[2]); }
      }
      for (let j = 0; j < idx.length; j++) slot[idx[j]] = -1;
      if (moreI.length) { const I = new Uint32Array(idx.length + moreI.length), D = new Float32Array(d.length + moreD.length); I.set(idx); I.set(moreI, idx.length); D.set(d); D.set(moreD, d.length); s.idx = I; s.d = D; }
    }
  }
  // How the surface has grown and turned at each vertex between the published head (A) and the character (B). What an
  // expression shape moves at a vertex is made larger or smaller by the first and turned by the second, so the movement
  // stays on the surface of the new face: the corner of a smiling mouth runs up the round cheek instead of digging into it.
  _follow(A, B) {
    const { n, tri, region, rings } = this.base, R = this.base.meta.regions, first = this.first; this._iron(new Float32Array(0), 0, () => 1); const nb = this._nb;
    const a0 = new Float32Array(n), a1 = new Float32Array(n), n0 = new Float32Array(n * 3), n1 = new Float32Array(n * 3);
    const sum = (P, ar, nr) => { for (let t = 0; t < tri.length; t += 3) { const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3, ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2], nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, l = Math.hypot(nx, ny, nz);
      for (let k = 0; k < 3; k++) { const f = first[tri[t + k]]; ar[f] += l; nr[f * 3] += nx; nr[f * 3 + 1] += ny; nr[f * 3 + 2] += nz; } } };
    sum(A, a0, n0); sum(B, a1, n1);
    const grow = new Float32Array(n).fill(1), turn = new Float32Array(n * 3), list = [];
    for (let i = 0; i < n; i++) {
      if (first[i] !== i || !(a0[i] > 0)) continue; list.push(i); grow[i] = Math.sqrt(a1[i] / a0[i]);
      const o = i * 3, l0 = Math.hypot(n0[o], n0[o + 1], n0[o + 2]) || 1, l1 = Math.hypot(n1[o], n1[o + 1], n1[o + 2]) || 1, px = n0[o] / l0, py = n0[o + 1] / l0, pz = n0[o + 2] / l0, qx = n1[o] / l1, qy = n1[o + 1] / l1, qz = n1[o + 2] / l1;
      const cx = py * qz - pz * qy, cy = pz * qx - px * qz, cz = px * qy - py * qx, sn = Math.hypot(cx, cy, cz), cs = px * qx + py * qy + pz * qz;
      if (sn > 1e-6) { const k = Math.min(0.9, Math.atan2(sn, cs)) / sn; turn[o] = cx * k; turn[o + 1] = cy * k; turn[o + 2] = cz * k; }
    }
    // a calm field: each value averaged with those around it, a few times
    const q = new Float32Array(n * 4);
    for (let p = 0; p < 10; p++) {
      for (const i of list) { const a = nb[i]; let g = 0, x = 0, y = 0, z = 0; for (let j = 0; j < a.length; j++) { const v = a[j]; g += grow[v]; x += turn[v * 3]; y += turn[v * 3 + 1]; z += turn[v * 3 + 2]; } const m = a.length || 1, o = i * 4;
        q[o] = 0.5 * (grow[i] + g / m); q[o + 1] = 0.5 * (turn[i * 3] + x / m); q[o + 2] = 0.5 * (turn[i * 3 + 1] + y / m); q[o + 3] = 0.5 * (turn[i * 3 + 2] + z / m); }
      for (const i of list) { const o = i * 4; grow[i] = q[o]; turn[i * 3] = q[o + 1]; turn[i * 3 + 1] = q[o + 2]; turn[i * 3 + 2] = q[o + 3]; }
    }
    // teeth, gums and tongue move as one piece, the way the lips do; the lining of the mouth goes over from the lips to that
    let gm = 0, tx = 0, m = 0;
    for (const i of list) { const r = R[region[i]]; if ((r === 'face' || r === 'head') && rings[i * 3] <= 9) { gm += grow[i]; tx += turn[i * 3]; m++; } }
    gm /= m || 1; tx /= m || 1; this.mouthGrow = gm;
    for (const i of list) { const r = R[region[i]]; if (r !== 'teeth' && r !== 'gums' && r !== 'mouth') continue; const t = r === 'mouth' ? smooth(1, 8, rings[i * 3]) : 1, o = i * 3;
      grow[i] += t * (gm - grow[i]); turn[o] += t * (tx - turn[o]); turn[o + 1] *= 1 - t; turn[o + 2] *= 1 - t; }
    for (let i = 0; i < n; i++) if (first[i] !== i) { const f = first[i]; grow[i] = grow[f]; turn[i * 3] = turn[f * 3]; turn[i * 3 + 1] = turn[f * 3 + 1]; turn[i * 3 + 2] = turn[f * 3 + 2]; }
    return { grow, turn };
  }
  // The relief of the face between its features is evened out: how far the skin stands off the smooth form of the head
  // is averaged over the skin, again and again, while the lids, the nose, the lips and the ears keep theirs. What is left
  // between them is one calm surface: no bags under the eyes, no shelf where the cheek begins, no fold beside the nose.
  _calmRelief(P, R1, passes) {
    const { n, pos, region, rings } = this.base, R = this.base.meta.regions, first = this.first, C = BALL_C; this._iron(new Float32Array(0), 0, () => 1); const nb = this._nb;
    const e = new Float32Array(n), dir = new Float32Array(n * 3), b1 = new Float32Array(n), list = [], wt = [];
    for (let i = 0; i < n; i++) {
      if (first[i] !== i) continue; const r0 = R[region[i]]; if (r0 !== 'face' && r0 !== 'head') continue;
      const qx = P[i * 3] - C[0], qy = P[i * 3 + 1] - C[1], qz = P[i * 3 + 2] - C[2], r = Math.hypot(qx, qy, qz); dir[i * 3] = qx / r; dir[i * 3 + 1] = qy / r; dir[i * 3 + 2] = qz / r;
      b1[i] = R1.at(qx / r, qy / r, qz / r); e[i] = r - b1[i];
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      let w = this._keep[i] > 0 ? 1 : 0;
      if (this.shape.noseForm) { // lids: what grows with the eye as a whole; lips: the red of the lips
        const c = this.base.meta.eyes.eyeL.centre, B = this.shape.eyeBox, ex = Math.abs(x) - c[0], ey = y - c[1];
        w *= smooth(0.85, 1.3, Math.hypot(ex / (ex < 0 ? B[0] : B[1]), ey / (ey < 0 ? B[2] : B[3]))) * smooth(10, 16, rings[i * 3]);
      } else w *= smooth(5, 12, Math.min(rings[i * 3 + 1], rings[i * 3 + 2])) * smooth(13, 20, rings[i * 3]); // lids, lips
      if (!this.shape.noseForm) w *= smooth(1.9, 3.2, Math.hypot(x, (y + 0.3) * 0.8, (z - 11.2) * 0.7)); // nose
      w *= 1 - smooth(6.3, 7.2, Math.abs(x)) * (1 - smooth(4.2, 5.4, z)); // ears
      w *= smooth(0.5, 3, headPlane(y, z)) * smooth(-2, 3.5, z); // not the neck, not the back of the head
      if (w > 0.01) { list.push(i); wt.push(w); }
    }
    // The relief that a film stretched between the features would have (every vertex at the mean of its neighbours,
    // the features held), and then a soft change-over from the features' own relief to that: no edge shows where the
    // one ends and the other begins.
    const q = new Float32Array(list.length), e0 = Float32Array.from(e);
    for (let p = 0; p < passes; p++) {
      for (let k = 0; k < list.length; k++) { const i = list[k], a = nb[i]; let t = 0; for (let j = 0; j < a.length; j++) t += e[a[j]]; q[k] = e[i] + 0.6 * (t / a.length - e[i]); }
      for (let k = 0; k < list.length; k++) e[list[k]] = q[k];
    }
    for (let k = 0; k < list.length; k++) { const i = list[k]; e[i] = e0[i] + wt[k] * (e[i] - e0[i]); }
    for (const i of list) { const r = b1[i] + e[i]; P[i * 3] = C[0] + dir[i * 3] * r; P[i * 3 + 1] = C[1] + dir[i * 3 + 1] * r; P[i * 3 + 2] = C[2] + dir[i * 3 + 2] * r; }
    for (let i = 0; i < n; i++) if (first[i] !== i) { const f = first[i] * 3, o = i * 3; P[o] = P[f]; P[o + 1] = P[f + 1]; P[o + 2] = P[f + 2]; }
  }
  // Ironing the skin: each pass moves a vertex towards the middle of its neighbours and back out again, so the large
  // forms keep their size while folds and ripples go. `amount(i)` says how much each vertex takes part (0 to 1). The rims
  // of the lids and the line where the lips meet never move; they have to fit the eyes and the teeth.
  _iron(P, passes, amount, plain = false) { // plain: towards the middle only, which draws the skin flat like a film
    const { n, pos, tri, region, rings } = this.base, R = this.base.meta.regions, first = this.first;
    if (!this._nb) {
      const nb = Array.from({ length: n }, () => new Set());
      for (let t = 0; t < tri.length; t += 3) { const a = first[tri[t]], b = first[tri[t + 1]], c = first[tri[t + 2]]; nb[a].add(b).add(c); nb[b].add(a).add(c); nb[c].add(a).add(b); }
      const keep = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        if (first[i] !== i) continue; const r = R[region[i]];
        if (r !== 'face' && r !== 'head') continue;
        keep[i] = smooth(2, 7, Math.min(rings[i * 3 + 1], rings[i * 3 + 2])) * smooth(0, 4, rings[i * 3]);
      }
      // a skin vertex that shares its place with a vertex of another part (mouth, socket) stays
      for (let i = 0; i < n; i++) if (first[i] !== i) { const r = R[region[i]]; if (r !== 'face' && r !== 'head') keep[first[i]] = 0; }
      this._nb = nb.map((s) => Int32Array.from(s)); this._keep = keep;
    }
    const nb = this._nb, list = [], wt = [];
    for (let i = 0; i < n; i++) { const k = this._keep[i] * (this._keep[i] > 0 ? amount(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], i) : 0); if (k > 0.01) { list.push(i); wt.push(k); } }
    const Q = new Float32Array(list.length * 3);
    const pass = (f) => {
      for (let k = 0; k < list.length; k++) { const i = list[k], a = nb[i]; let x = 0, y = 0, z = 0; for (let j = 0; j < a.length; j++) { const v = a[j] * 3; x += P[v]; y += P[v + 1]; z += P[v + 2]; }
        const g = (f * wt[k]) / a.length, o = i * 3, q = k * 3; Q[q] = P[o] + g * (x - a.length * P[o]); Q[q + 1] = P[o + 1] + g * (y - a.length * P[o + 1]); Q[q + 2] = P[o + 2] + g * (z - a.length * P[o + 2]); }
      for (let k = 0; k < list.length; k++) { const o = list[k] * 3, q = k * 3; P[o] = Q[q]; P[o + 1] = Q[q + 1]; P[o + 2] = Q[q + 2]; }
    };
    for (let k = 0; k < passes; k++) { pass(0.5); if (!plain) pass(-0.53); }
    for (let i = 0; i < n; i++) if (first[i] !== i) { const f = first[i] * 3, o = i * 3; P[o] = P[f]; P[o + 1] = P[f + 1]; P[o + 2] = P[f + 2]; }
    return P;
  }
  _colors() {
    const { n, region, rings, pos } = this.base, R = this.base.meta.regions, d = this.design, c = new Float32Array(n * 3);
    const skin = new THREE.Color(d.skin || '#f1c7a5'), lip = new THREE.Color(d.lip || '#c9736b'), lid = new THREE.Color(d.lid || d.skin || '#f1c7a5').lerp(new THREE.Color(d.blush || '#e8806f'), 0.25), blush = new THREE.Color(d.blush || '#e8806f'), tmp = new THREE.Color();
    const lipRings = d.lipRings ?? 13, bl = d.blushAmount ?? 0.3, lipIn = lip.clone().multiplyScalar(0.8);
    for (let i = 0; i < n; i++) {
      const r = R[region[i]], x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      if (r === 'face' || r === 'head') {
        const m = rings[i * 3], er = Math.min(rings[i * 3 + 1], rings[i * 3 + 2]), l = r === 'face' ? 1 - smooth(lipRings - 2.5, lipRings + 0.5, m) : 0, e = 1 - smooth(2, 14, er);
        // warmer where the blood shows: cheeks, tip of the nose, ears, lids
        const ax = Math.abs(x), cheek = Math.exp(-((ax - 4.5) ** 2 + (y + 0.4) ** 2 * 1.3 + (z - 9.2) ** 2 * 0.6) / 6.5), nose = 0.55 * Math.exp(-(x * x + (y - 0.0) ** 2 + (z - 12.6) ** 2) / 1.6), ear = 0.5 * smooth(7.5, 8.6, ax) * (1 - smooth(3, 5, z));
        tmp.copy(skin).lerp(blush, Math.min(1, bl * (cheek + nose + ear))).lerp(lid, e * 0.5).lerp(lip, l * (d.lipAmount ?? 0.85)).lerp(lipIn, l * (1 - smooth(4, 9, m)) * 0.6);
      } else if (r === 'mouth') { const m = rings[i * 3]; tmp.set(REGION_COLOR.mouth).lerp(lipIn, 1 - smooth(1, 6, m)); }
      else if (r === 'gums') tmp.set(REGION_COLOR.gums).lerp(new THREE.Color('#e88a94'), smooth(-4.4, -3.4, y) * smooth(7, 9, z) * 0); // gums and tongue
      else tmp.set(REGION_COLOR[r]);
      c[i * 3] = tmp.r; c[i * 3 + 1] = tmp.g; c[i * 3 + 2] = tmp.b;
    }
    return c;
  }
  _materials(R) {
    const skin = occluded(new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.5, metalness: 0, sheen: 0.4, sheenRoughness: 0.55, sheenColor: new THREE.Color('#ff9a7a'), clearcoat: 0.06, clearcoatRoughness: 0.5 }), this.design.shade || '#8a3f33');
    const wet = occluded(new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.3, clearcoat: 0.5, clearcoatRoughness: 0.2 }), '#30121a', 0.8);
    const teeth = occluded(new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.22, clearcoat: 0.6, clearcoatRoughness: 0.15 }), '#3a2a35', 0.8);
    const lash = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide });
    const inner = occluded(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 }), '#1a090c', 0.9);
    return R.map((r) => (r === 'face' || r === 'head' ? skin : r === 'gums' ? wet : r === 'teeth' ? teeth : r.startsWith('lash') ? lash : inner));
  }
  // Soft shadows that do not move, measured once (bake.js): in the rest face for everything; with closed eyes too for the
  // skin, so that the part of a lid that is folded away while the eye is open is not dark when it closes. The inside of
  // the mouth is not measured (it is closed in the rest face): it simply gets darker towards the throat.
  bake(renderer, top = this.root, others = []) {
    const save = Float32Array.from(this.w); this.set({}); this.update(true);
    const meshes = [this.mesh, ...this.eyes.map((e) => e.mesh), ...this.sculpts, ...others];
    bakeOcclusion(renderer, top, meshes);
    const bi = ['eyeBlink_L', 'eyeBlink_R'].map((k) => this.base.index[k]); for (const i of bi) this.w[i] = 1; this.update(true);
    bakeOcclusion(renderer, top, [this.mesh], { combine: 'max', dirs: 40 });
    const { n, region, rings, pos } = this.base, R = this.base.meta.regions, A = this.geometry.attributes.ao.array;
    for (let i = 0; i < n; i++) {
      const r = R[region[i]];
      if (r === 'mouth' || r === 'gums' || r === 'teeth') A[i] = 0.22 + 0.75 * smooth(5.5, 10.3, pos[i * 3 + 2]);
      else if (r === 'face' && rings[i * 3] < 10) A[i] = Math.max(A[i], 0.5 + 0.05 * rings[i * 3]);
    }
    smoothOcclusion(this.geometry, 2);
    for (const m of meshes) if (m !== this.mesh) smoothOcclusion(m.geometry, 1);
    this.w.set(save); this.update(true);
  }
  // Drawn teeth: one smooth white band behind the upper lip and one behind the lower lip, each following the line of the
  // lips from corner to corner and on into the cheeks. The upper one sits still in the head; the lower one goes with the
  // jaw (it takes its place from three of the published lower teeth, which are moved but not shown).
  // d: { show (cm of the upper band below the line of the lips), back (cm behind the front of the lips), tall, color }
  _teeth(d) {
    const W = this.warp, o = [0, 0, 0], N = 9, X = [], Y = [], Z = [];
    for (let k = 0; k < N; k++) { const x = (2.3 * k) / (N - 1); W(x, -3.45, 11.42 - 0.23 * x * x, o); X.push(o[0]); Y.push(o[1]); Z.push(o[2]); } // the line where the lips meet, from the middle to a corner
    // the height of that line as a + b x^2 (least squares)
    const fit = (V) => { let s0 = 0, s2 = 0, s4 = 0, v0 = 0, v2 = 0; for (let k = 0; k < N; k++) { const q = X[k] * X[k]; s0++; s2 += q; s4 += q * q; v0 += V[k]; v2 += V[k] * q; } const det = s0 * s4 - s2 * s2; return [(v0 * s4 - v2 * s2) / det, (s0 * v2 - s2 * v0) / det]; };
    const fy = fit(Y), xc = X[N - 1], gm = this.mouthGrow || 1, back = (d.back ?? 0.8) * gm, show = (d.show ?? 0.45) * gm, tall = (d.tall ?? 1.3) * gm, xe = xc + 0.9 * gm, r = 0.16 * gm;
    // The bands lie at one depth under the form of the head (the smooth form, which the cheeks and the chin follow), so
    // they stay inside the head wherever they reach: front(x, y) is the form there, off how far under it the teeth lie.
    const C = BALL_C, R1 = W.ball.R1, front = (x, y) => { let z = Z[0]; for (let k = 0; k < 6; k++) { const qx = x - C[0], qy = y - C[1], qz = z - C[2], l = Math.hypot(qx, qy, qz), R = R1.at(qx / l, qy / l, qz / l); z = C[2] + Math.sqrt(Math.max(0.01, R * R - qx * qx - qy * qy)); } return z; };
    const off = front(0, Y[0]) - Z[0] + back;
    const mat = occluded(new THREE.MeshPhysicalMaterial({ color: d.color || '#f7f4ee', roughness: 0.25, clearcoat: 0.5, clearcoatRoughness: 0.2 }), '#4a3038', 0.85);
    const band = (y0, y1, zOff, edgeDown) => { // from y0 (the edge that shows, rounded) to y1 (hidden behind the lip), both from the line of the lips
      const M = 44, pos = [], ao = [], idx = [], sg = edgeDown ? 1 : -1, P = 7;
      for (let i = 0; i <= M; i++) {
        const x = -xe + (2 * xe * i) / M, ax = Math.abs(x), q = Math.min(x * x, xc * xc) + 2 * xc * Math.max(0, ax - xc), ys = fy[0] + fy[1] * q, thin = 1 - 0.85 * smooth(xc, xe, ax); // (beyond the corners the line of the lips is carried on straight, and the band runs out)
        const e = ys + y0 * thin, t = ys + y1 * thin, z = (y) => front(x, y) - off + zOff;
        const prof = [[t, -0.12 * gm], [t - sg * 0.1 * gm, 0], [(e + t) / 2, 0.03 * gm], [e + sg * r, 0], [e + sg * 0.29 * r, -0.29 * r], [e, -r], [e + sg * 0.25 * gm, -r - 0.3 * gm]];
        for (const [py, dz] of prof) { pos.push(x, py, z(py) + dz); ao.push(0.55 + 0.45 * (1 - smooth(0.5 * xc, 1.15 * xc, ax))); }
        if (i < M) for (let k = 0; k < P - 1; k++) { const a = i * P + k, b = a + 1, c = a + P, e2 = c + 1; if (edgeDown) idx.push(a, b, c, b, e2, c); else idx.push(a, c, b, b, c, e2); }
      }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('ao', new THREE.Float32BufferAttribute(ao, 1)); g.setIndex(idx); g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, mat); mesh.frustumCulled = false; mesh.receiveShadow = true; return mesh;
    };
    this.teethUpper = band(-show, tall, 0, true); this.root.add(this.teethUpper);
    this.teethLower = band(-(d.lowerShow ?? 0.15) * gm, -(d.lowerTall ?? 1.5) * gm, -0.14 * gm, false); this.teethLower.matrixAutoUpdate = false; this.root.add(this.teethLower);
    // three of the published lower teeth: the middle one in front and the hindmost on each side
    const { n, pos, region } = this.base, ti = this.base.meta.regions.indexOf('teeth'); let a = -1, b = -1, c = -1;
    for (let i = 0; i < n; i++) { if (region[i] !== ti || pos[i * 3 + 1] > -3.6) continue; if (Math.abs(pos[i * 3]) < 0.5 && (a < 0 || pos[i * 3 + 2] > pos[a * 3 + 2])) a = i; if (b < 0 || pos[i * 3] > pos[b * 3]) b = i; if (c < 0 || pos[i * 3] < pos[c * 3]) c = i; }
    this.jaw = { v: [a, b, c], rest: new THREE.Matrix4(), now: new THREE.Matrix4() }; this._frame(this.rest, this.jaw.rest); this.jaw.rest.invert();
  }
  // the place and bearing of the lower jaw, from its three teeth
  _frame(P, m) {
    const [a, b, c] = this.jaw.v, A = a * 3, B = b * 3, Cc = c * 3;
    let ex = P[B] - P[Cc], ey = P[B + 1] - P[Cc + 1], ez = P[B + 2] - P[Cc + 2]; const l = Math.hypot(ex, ey, ez); ex /= l; ey /= l; ez /= l;
    let tx = P[A] - (P[B] + P[Cc]) / 2, ty = P[A + 1] - (P[B + 1] + P[Cc + 1]) / 2, tz = P[A + 2] - (P[B + 2] + P[Cc + 2]) / 2; const dt = tx * ex + ty * ey + tz * ez; tx -= dt * ex; ty -= dt * ey; tz -= dt * ez; const l2 = Math.hypot(tx, ty, tz); tx /= l2; ty /= l2; tz /= l2;
    const ux = ty * ez - tz * ey, uy = tz * ex - tx * ez, uz = tx * ey - ty * ex; // up = forward x side
    m.set(ex, ux, tx, P[A], ey, uy, ty, P[A + 1], ez, uz, tz, P[A + 2], 0, 0, 0, 1); return m;
  }
  _jawMoves() { const j = this.jaw; this._frame(this.cur, j.now); this.teethLower.matrix.multiplyMatrices(j.now, j.rest); this.teethLower.matrixWorldNeedsUpdate = true; }
  // A sculpted shape (hair, a hat) described on the published head: it is made into a mesh there and taken through the
  // character's proportions like everything else, so it fits whatever head the character has.
  addSculpt(field, material, cell = 0.4) {
    const t0 = performance.now(), m = polygonise(field, cell), o = [0, 0, 0], P = m.pos;
    for (let i = 0; i < P.length; i += 3) { this.warp(P[i], P[i + 1], P[i + 2], o); P[i] = o[0]; P[i + 1] = o[1]; P[i + 2] = o[2]; }
    if (field.after) field.after(P); // changes made on the character itself, after the proportions
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(P, 3)); g.setIndex(new THREE.BufferAttribute(m.idx, 1)); g.computeVertexNormals(); g.setAttribute('ao', new THREE.BufferAttribute(new Float32Array(P.length / 3).fill(1), 1));
    const mesh = new THREE.Mesh(g, material); mesh.castShadow = true; mesh.receiveShadow = true; this.root.add(mesh); this.sculpts.push(mesh);
    mesh.userData.buildMs = performance.now() - t0; return mesh;
  }
  get skin() { if (!this._skin) { const G = this.base.meta.groups; this._skin = new Skin(this.src, this.base.tri, G[0][0], G[1][0] + G[1][1]); } return this._skin; }
  // The lash line of the upper lid: a dark rounded line along the rim, thicker towards the outer corner, with a short wing.
  // d: { thick (cm), wing (cm, 0 for none), color }
  _lash(e, d) {
    const sg = e ? -1 : 1, skin = this.skin, hits = [], xs = [], N = 26;
    // the rim of the upper lid as seen from the front: going up from the opening of the eye, the first skin that is met
    for (let i = 0; i < N; i++) {
      const x = 2.1 + (4.42 - 2.1) * (i / (N - 1)), y0 = 3.36 + (0.09 * (x - 2.05)) / 2.4; let lo = y0, hi = y0 + 1.3;
      if (skin.front(x * sg, lo) || !skin.front(x * sg, hi)) continue;
      for (let k = 0; k < 14; k++) { const m = (lo + hi) / 2; if (skin.front(x * sg, m)) hi = m; else lo = m; }
      const hit = skin.front(x * sg, hi + 0.03); if (hit) { hits.push(hit); xs.push([x, hi + 0.03, hit.z]); }
    }
    const m = hits.length, thick = d.thick ?? 0.07, wing = d.wing ?? 0.3, radii = xs.map((p, i) => thick * (0.35 + 0.65 * smooth(0, 0.6, i / (m - 1))));
    if (wing > 0) { const a = xs[m - 3], b = xs[m - 1]; let dx = b[0] - a[0], dy = b[1] - a[1] + 0.9 * Math.hypot(b[0] - a[0], b[1] - a[1]), dz = b[2] - a[2]; const l = Math.hypot(dx, dy, dz); dx /= l; dy /= l; dz /= l;
      for (const [f, rr] of [[0.35, 0.95], [0.7, 0.7], [1, 0.3]]) { hits.push(skin.near((b[0] + dx * wing * f) * sg, b[1] + dy * wing * f, b[2] + dz * wing * f)); radii.push(thick * rr); } }
    const mat = this._lashMat || (this._lashMat = new THREE.MeshStandardMaterial({ color: d.color || '#1f1512', roughness: 0.6 }));
    // radii grow with the skin
    const o = [0, 0, 0], q = [0, 0, 0], c = xs[m >> 1]; this.warp(c[0] * sg, c[1], c[2], o); this.warp(c[0] * sg + 0.1, c[1], c[2], q); const sc = Math.hypot(q[0] - o[0], q[1] - o[1], q[2] - o[2]) / 0.1;
    return new Tube(this, hits, radii.map((r) => r * sc), mat, 8);
  }
  // A brow: a strip of hair standing a little off the skin, drawn on the published head as seen from the front.
  // d: { in: [x, y], top: [x, y], out: [x, y] (inner end, highest point, outer end; cm, left side), thick: [inner, middle, outer] (cm), lift (cm), color }
  _brow(side, d) {
    const nT = 22, nS = 6, idx = [];
    for (let i = 0; i < nT; i++) for (let j = 0; j < nS; j++) { const a = i * (nS + 1) + j, b = a + 1, c = a + nS + 1, e = c + 1; if (side > 0) idx.push(a, c, b, b, c, e); else idx.push(a, b, c, b, e, c); }
    const mat = this._browMat || (this._browMat = new THREE.MeshStandardMaterial({ color: d.color || this.design.hair || '#3a2a20', roughness: 0.75 }));
    const b = new Bound(this, this._browHits(side, d, null), this._browHs, Uint16Array.from(idx), mat);
    b.side = side; b.d = d; b.key = ''; return b;
  }
  // where the points of a brow lie on the published head. move: [inner dx, dy, top dx, dy, outer dx, dy] (cm) or null
  _browHits(side, d, move) {
    const m = move || [0, 0, 0, 0, 0, 0], a0 = d.in || [1.25, 5.0], b0 = d.top || [4.0, 5.8], c0 = d.out || [5.9, 5.05], th = d.thick || [0.55, 0.5, 0.12], lift = d.lift ?? 0.13;
    const A = [a0[0] + m[0], a0[1] + m[1]], B = [b0[0] + m[2], b0[1] + m[3]], C = [c0[0] + m[4], c0[1] + m[5]];
    const nT = 22, nS = 6, hits = [], hs = [], skin = this.skin;
    // parabola through the three points, y as a function of x
    const yOf = (x) => (A[1] * (x - B[0]) * (x - C[0])) / ((A[0] - B[0]) * (A[0] - C[0])) + (B[1] * (x - A[0]) * (x - C[0])) / ((B[0] - A[0]) * (B[0] - C[0])) + (C[1] * (x - A[0]) * (x - B[0])) / ((C[0] - A[0]) * (C[0] - B[0]));
    const tb = (C[0] - B[0] > 0 ? (B[0] - A[0]) / (C[0] - A[0]) : 0.6), round = d.round ?? 0.12; // round: how much of the length the rounded inner end takes
    for (let i = 0; i <= nT; i++) {
      const t = i / nT, x = A[0] + (C[0] - A[0]) * t, y = yOf(x), dydx = (yOf(x + 0.01) - yOf(x - 0.01)) / 0.02, l = Math.hypot(1, dydx), nx = -dydx / l, ny = 1 / l;
      const cap = t < round ? Math.sqrt(Math.max(0.02, 1 - (1 - t / round) ** 2)) : 1; // the inner end is round, the outer end a point
      const half = 0.5 * (t < tb ? th[0] + (th[1] - th[0]) * smooth(0, tb, t) : th[1] + (th[2] - th[1]) * smooth(tb, 1, t)) * cap * (1 - 0.6 * smooth(0.9, 1, t));
      for (let j = 0; j <= nS; j++) {
        const u = (j / nS) * 2 - 1, px = (x + nx * half * u) * side, py = y + ny * half * u;
        hits.push(skin.front(px, py) || skin.near(px, py, 9));
        if (!move) hs.push(lift * Math.sqrt(Math.max(0, 1 - u * u)) * Math.min(1, cap + 0.15) * (1 - 0.7 * smooth(0.85, 1, t)) - 0.02);
      }
    }
    if (!move) this._browHs = hs;
    return hits;
  }
  // A brow slides over the skin (the skin under it moves too, with the expression shapes): cartoon brows travel much
  // further than real ones. move: [inner dx, dy, top dx, dy, outer dx, dy] in cm on the published head, x outwards.
  browMove(which, move) {
    const b = this.brows[which]; if (!b) return;
    const key = move.map((v) => Math.round(v * 40)).join(','); if (key === b.key) return; b.key = key;
    b.bind(this._browHits(b.side, b.d, move), this._browHs); this.dirty = true;
  }
  // An eyeball: the sphere of the published head, taken through the same change of shape as the lids around it, so it
  // always fits them. It does not turn: the iris slides over it (see look), which is what lets an eye be larger and
  // flatter than a ball.
  _eye(e) {
    const E = this.base.meta.eyes[e ? 'eyeR' : 'eyeL'], c = E.centre, R = E.radius * 0.995, d = this.design;
    const geo = new THREE.SphereGeometry(1, 48, 36); geo.rotateX(Math.PI / 2); // poles to the front and the back
    const p = geo.attributes.position, dir = Float32Array.from(p.array), o = [0, 0, 0];
    for (let i = 0; i < p.count; i++) { this.warp(c[0] + dir[i * 3] * R, c[1] + dir[i * 3 + 1] * R, c[2] + dir[i * 3 + 2] * R, o); p.setXYZ(i, o[0], o[1], o[2]); }
    geo.computeVertexNormals();
    // seam of the sphere: the same place twice, give both the same normal
    { const nrm = geo.attributes.normal, key = new Map(); for (let i = 0; i < p.count; i++) { const k = `${Math.round(dir[i * 3] * 1e4)},${Math.round(dir[i * 3 + 1] * 1e4)},${Math.round(dir[i * 3 + 2] * 1e4)}`; const a = key.get(k); if (!a) key.set(k, [i]); else a.push(i); }
      for (const a of key.values()) if (a.length > 1) { let x = 0, y = 0, z = 0; for (const i of a) { x += nrm.getX(i); y += nrm.getY(i); z += nrm.getZ(i); } const l = Math.hypot(x, y, z) || 1; for (const i of a) nrm.setXYZ(i, x / l, y / l, z / l); } }
    geo.setAttribute('ao', new THREE.BufferAttribute(new Float32Array(p.count).fill(1), 1));
    const mesh = new THREE.Mesh(geo, this._eyeMat || (this._eyeMat = occluded(new THREE.MeshPhysicalMaterial({ map: this._eyeTexture(), roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.03 }), '#4a4458', 0.85)));
    mesh.frustumCulled = false; mesh.receiveShadow = true;
    // how much taller than wide the front of the eye has become: the iris is drawn that much flatter, so it stays round
    const q = [0, 0, 0], off = 0.3 * R, f = Math.sqrt(R * R - off * off), far = (ax, ay) => { this.warp(c[0] + ax, c[1] + ay, c[2] + f, o); this.warp(c[0] - ax, c[1] - ay, c[2] + f, q); return Math.hypot(o[0] - q[0], o[1] - q[1], o[2] - q[2]); };
    return { mesh, dir, uv: geo.attributes.uv, tall: far(0, off) / far(off, 0) };
  }
  // the eye seen along its line of sight: white, iris, pupil. Painted once per head.
  _eyeTexture() {
    if (this._eyeTex) return this._eyeTex;
    const d = this.design, S = 1024, c = document.createElement('canvas'); c.width = c.height = S; const g = c.getContext('2d'), m = S / 2;
    const hex = (col) => '#' + col.getHexString(), rgba = (col, a) => { const h = col.getHex(THREE.SRGBColorSpace); return `rgba(${h >> 16},${(h >> 8) & 255},${h & 255},${a})`; };
    // the white: a little warmer and darker away from the middle
    g.fillStyle = d.white || '#f6f3ee'; g.fillRect(0, 0, S, S);
    let grd = g.createRadialGradient(m, m, S * 0.28, m, m, S * 0.5); grd.addColorStop(0, 'rgba(190,150,150,0)'); grd.addColorStop(1, 'rgba(190,150,150,0.35)'); g.fillStyle = grd; g.fillRect(0, 0, S, S);
    const ir = m * (d.irisSize ?? 0.56), iris = new THREE.Color(d.iris || '#6a4a2f'), dark = iris.clone().multiplyScalar(0.42), light = iris.clone().lerp(new THREE.Color('#fff2d8'), 0.38);
    g.save(); g.beginPath(); g.arc(m, m, ir, 0, 7); g.clip();
    grd = g.createRadialGradient(m, m, ir * 0.2, m, m, ir); grd.addColorStop(0, hex(light)); grd.addColorStop(0.6, hex(iris)); grd.addColorStop(0.88, hex(dark)); grd.addColorStop(1, hex(dark.clone().multiplyScalar(0.55)));
    g.fillStyle = grd; g.fillRect(0, 0, S, S);
    // fibres of the iris
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let k = 0; k < 220; k++) { const a = rnd() * Math.PI * 2, r0 = ir * (0.34 + 0.12 * rnd()), r1 = ir * (0.7 + 0.25 * rnd()); g.strokeStyle = rnd() < 0.6 ? rgba(light, 0.10 + 0.16 * rnd()) : rgba(dark, 0.12 + 0.16 * rnd()); g.lineWidth = 1 + 3 * rnd(); g.beginPath(); g.moveTo(m + Math.cos(a) * r0, m + Math.sin(a) * r0); g.lineTo(m + Math.cos(a + 0.03 * (rnd() - 0.5)) * r1, m + Math.sin(a) * r1); g.stroke(); }
    // light gathered in the lower half, shade of the lid in the upper
    grd = g.createRadialGradient(m, m + ir * 0.45, 0, m, m + ir * 0.45, ir * 0.75); grd.addColorStop(0, rgba(light, 0.55)); grd.addColorStop(1, rgba(light, 0)); g.globalCompositeOperation = 'screen'; g.fillStyle = grd; g.fillRect(0, 0, S, S);
    g.globalCompositeOperation = 'multiply'; grd = g.createLinearGradient(0, m - ir, 0, m + ir * 0.3); grd.addColorStop(0, 'rgba(40,30,40,1)'); grd.addColorStop(1, 'rgba(255,255,255,1)'); g.fillStyle = grd; g.fillRect(0, 0, S, S);
    g.globalCompositeOperation = 'source-over';
    // pupil with a soft edge
    const pr = ir * (d.pupilSize ?? 0.4); grd = g.createRadialGradient(m, m, pr * 0.82, m, m, pr * 1.06); grd.addColorStop(0, 'rgba(8,6,6,1)'); grd.addColorStop(1, 'rgba(8,6,6,0)'); g.fillStyle = grd; g.beginPath(); g.arc(m, m, pr * 1.1, 0, 7); g.fill();
    g.restore();
    // soft dark rim where iris and white meet
    g.strokeStyle = rgba(dark.clone().multiplyScalar(0.5), 0.55); g.lineWidth = ir * 0.07; g.filter = `blur(${Math.round(ir * 0.03)}px)`; g.beginPath(); g.arc(m, m, ir * 0.985, 0, 7); g.stroke(); g.filter = 'none';
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8; tex.flipY = false;
    return (this._eyeTex = tex);
  }
  // The glint in an eye: a small bright patch on the eyeball that stays where it is while the iris moves under it, like
  // the mirror image of a lamp. dir: where on the eye it sits, as yaw and pitch in degrees.
  _glint(e, yaw, pitch, size) {
    const E = this.base.meta.eyes[e ? 'eyeR' : 'eyeL'], c = E.centre, R = E.radius * 1.006, a = (yaw * Math.PI) / 180, b = (pitch * Math.PI) / 180;
    const f = [Math.sin(a) * Math.cos(b), Math.sin(b), Math.cos(a) * Math.cos(b)], rt = [Math.cos(a), 0, -Math.sin(a)], up = [f[1] * rt[2] - f[2] * rt[1], f[2] * rt[0] - f[0] * rt[2], f[0] * rt[1] - f[1] * rt[0]];
    const n = 16, pos = [], col = [], idx = [], o = [0, 0, 0], put = (u, v, al) => { const x = f[0] + rt[0] * u + up[0] * v, y = f[1] + rt[1] * u + up[1] * v, z = f[2] + rt[2] * u + up[2] * v, l = Math.hypot(x, y, z); this.warp(c[0] + (x / l) * R, c[1] + (y / l) * R, c[2] + (z / l) * R, o); pos.push(o[0], o[1], o[2]); col.push(1, 1, 1, al); };
    const s = Math.tan((size * Math.PI) / 180); put(0, 0, 1);
    for (let k = 0; k < n; k++) { const t = (k / n) * Math.PI * 2; put(Math.cos(t) * s * 0.55, Math.sin(t) * s * 0.55, 0.95); }
    for (let k = 0; k < n; k++) { const t = (k / n) * Math.PI * 2; put(Math.cos(t) * s, Math.sin(t) * s, 0); }
    for (let k = 0; k < n; k++) { const k1 = (k + 1) % n; idx.push(0, 1 + k, 1 + k1, 1 + k, 1 + n + k, 1 + n + k1, 1 + k, 1 + n + k1, 1 + k1); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 4)); g.setIndex(idx);
    const mat = this._glintMat || (this._glintMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
    const mesh = new THREE.Mesh(g, mat); mesh.frustumCulled = false; mesh.renderOrder = 2; return mesh;
  }
  // weights by shape name, e.g. { jawOpen: 0.4, mouthSmile_L: 1 }; shapes not named go to 0
  set(weights) { const w = this.w, rp = this.restPose; if (rp) w.set(rp); else w.fill(0); for (const k in weights) { const i = this.base.index[k]; if (i !== undefined) w[i] += weights[k]; } }
  // line of sight in degrees: positive yaw = towards the character's left, positive pitch = up. Both eyes, or one (0 left, 1 right).
  // size: the iris with its pupil larger (> 1) or smaller.
  look(yaw, pitch, which = -1, size = 1) {
    const key = `${Math.round(yaw * 20)},${Math.round(pitch * 20)},${which},${Math.round(size * 200)}`; if (key === this._lookKey) return; this._lookKey = key;
    const a = (yaw * Math.PI) / 180, b = (pitch * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), k = 0.5 / size;
    // right and up of the eye when it looks along (sa cb, sb, ca cb)
    const rx = ca, rz = -sa, ux = -sa * sb, uy = cb, uz = -ca * sb;
    for (let e = 0; e < 2; e++) { if (which >= 0 && which !== e) continue; const { dir, uv } = this.eyes[e], t = uv.array;
      const kv = k * (this.design.irisRound === false ? 1 : this.eyes[e].tall);
      for (let i = 0, j = 0; i < dir.length; i += 3, j += 2) { t[j] = 0.5 + k * (rx * dir[i] + rz * dir[i + 2]); t[j + 1] = 0.5 - kv * (ux * dir[i] + uy * dir[i + 1] + uz * dir[i + 2]); }
      uv.needsUpdate = true; }
  }
  update(force = false) {
    const w = this.w, lw = this._lastW; let same = !force;
    if (same) for (let i = 0; i < w.length; i++) if (w[i] !== lw[i]) { same = false; break; }
    if (same && !this.dirty) return;
    lw.set(w); this.dirty = false;
    const cur = this.cur; cur.set(this.rest);
    for (let s = 0; s < this.shapes.length; s++) {
      const a = w[s]; if (!a) continue;
      const { idx, d } = this.shapes[s];
      for (let j = 0; j < idx.length; j++) { const i = idx[j] * 3, k = j * 3; cur[i] += a * d[k]; cur[i + 1] += a * d[k + 1]; cur[i + 2] += a * d[k + 2]; }
    }
    if (this.deform) this.deform(cur); // the body's own doing: the neck that stays on the shoulders (figure.js)
    if (this.jaw) this._jawMoves();
    // normals: area-weighted sum of the triangle normals around each vertex
    const N = this.normal, tri = this.base.tri; N.fill(0);
    for (let t = 0; t < tri.length; t += 3) {
      const a = tri[t] * 3, b = tri[t + 1] * 3, c = tri[t + 2] * 3;
      const ux = cur[b] - cur[a], uy = cur[b + 1] - cur[a + 1], uz = cur[b + 2] - cur[a + 2], vx = cur[c] - cur[a], vy = cur[c + 1] - cur[a + 1], vz = cur[c + 2] - cur[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      N[a] += nx; N[a + 1] += ny; N[a + 2] += nz; N[b] += nx; N[b + 1] += ny; N[b + 2] += nz; N[c] += nx; N[c + 1] += ny; N[c + 2] += nz;
    }
    const first = this.first;
    for (const i of this.dup) { const f = first[i] * 3, o = i * 3; N[f] += N[o]; N[f + 1] += N[o + 1]; N[f + 2] += N[o + 2]; }
    for (const i of this.dup) { const f = first[i] * 3, o = i * 3; N[o] = N[f]; N[o + 1] = N[f + 1]; N[o + 2] = N[f + 2]; }
    for (let i = 0; i < N.length; i += 3) { const l = Math.hypot(N[i], N[i + 1], N[i + 2]) || 1; N[i] /= l; N[i + 1] /= l; N[i + 2] /= l; }
    this.geometry.attributes.position.needsUpdate = true; this.geometry.attributes.normal.needsUpdate = true;
    for (const b of this.bound) b.update();
  }
}
