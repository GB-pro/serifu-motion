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
const ramp = (u, a = 1) => 0.5 * (u + Math.sqrt(u * u + a * a)); // u for u >> 0, 0 for u << 0, smooth in between
// Weight of a local scaling by s about a centre, as a function of the distance d from it: 1 up to r0, 0 from r1 on, and
// in between such that the distance after the scaling still grows with d (the skin is compressed evenly, never folded).
function radial(d, s, r0, r1) {
  if (d <= r0) return 1; if (d >= r1) return 0;
  if (Math.abs(s - 1) < 0.02) return 1 - smooth(r0, r1, d);
  r1 = Math.max(r1, s * r0 * 1.25);
  const h = r1 - r0, t = (d - r0) / h, t2 = t * t, t3 = t2 * t; // cubic through (r0, s r0) with slope s and (r1, r1) with slope 1
  const f = (2 * t3 - 3 * t2 + 1) * s * r0 + (t3 - 2 * t2 + t) * h * s + (-2 * t3 + 3 * t2) * r1 + (t3 - t2) * h;
  return Math.min(1, Math.max(0, (f / d - 1) / (s - 1)));
}

// Proportions of a character: a smooth change of place for every point of the published head (centimetres; x to the
// character's left, y up, z forward), the sum of a few local changes. The neutral head, every expression shape and the
// eyeballs go through the same change, so every expression keeps working: a larger eye still closes fully.
export const DEFAULT_SHAPE = {
  eye: 1, eyeTall: 1, eyeDepth: 1, eyeApart: 0, eyeDown: 0, eyeIn: 0, eyeZone: 3.2, // eyes with their lids: size, height, depth, place (cm), how far the skin around makes room
  skull: 1, skullWide: 1, skullTall: 1, skullDeep: 1, // head above the cheeks
  feature: 1, featureUp: 0, featureEdge: 2.0, // the face below the eyes (nose, mouth, chin, jaw) as one piece: size (it moves towards the eyes as it shrinks), shift (cm)
  bridge: 0, // the bridge of the nose between the eyes set back (cm)
  wide: 1, jaw: 1, chin: 1, cheek: 0, // width of the outline of the head, of the jaw; chin; fullness of the cheeks (cm)
  nose: 1, noseWide: 1, mouth: 1, ear: 1, neck: 1,
  calm: 0, // ironing passes on the movement of every expression shape: no folds beside the nose when the character smiles
  smooth: 0, noseSmooth: 0, relax: 0, // ironing passes: fine relief of the published head, the nose, ripples after the change of proportions
};
export function makeWarp(base, sh) {
  const eyes = ['eyeL', 'eyeR'].map((k) => { const e = base.meta.eyes[k]; return { x: e.centre[0], y: e.centre[1], z: e.centre[2] + e.radius, sg: Math.sign(e.centre[0]) }; });
  const sxE = sh.eye - 1, syE = sh.eye * sh.eyeTall - 1, szE = sh.eyeDepth - 1;
  const skx = sh.skull * sh.skullWide - 1, sky = sh.skull * sh.skullTall - 1, skz = sh.skull * sh.skullDeep - 1;
  return (x, y, z, o) => {
    let dx = 0, dy = 0, dz = 0, w;
    for (const e of eyes) { // the eye with its lids grows about the front of the eyeball; the skin around makes room over a wide area
      const ex = x - e.x, ey = y - e.y, ez = z - e.z, u2 = (ex * ex + ey * ey + 0.3 * ez * ez) / (sh.eyeZone * sh.eyeZone);
      w = 1 / ((1 + u2) * (1 + u2));
      dx += w * (sxE * ex + e.sg * sh.eyeApart); dy += w * (syE * ey - sh.eyeDown); dz += w * (szE * ez - sh.eyeIn);
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
    { const cx = x, cy = y + 7.6, cz = z - 8.5; w = radial(Math.hypot(cx, cy, cz), sh.chin, 1.8, 4.2); const k = w * (sh.chin - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
    { const cx = x, cy = y + 0.2, cz = z - 11; w = radial(Math.hypot(cx, cy, cz), sh.nose, 2.3, 4.4); const k = w * (sh.nose - 1); dx += k * cx + w * (sh.noseWide - 1) * sh.nose * cx; dy += k * cy; dz += k * cz; }
    { const cx = x, cy = y + 3.4, cz = z - 9.8; w = radial(Math.hypot(cx / 1.3, cy, cz / 1.2), sh.mouth, 2.2, 4); const k = w * (sh.mouth - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
    if (sh.cheek) { const cy = y + 1.2, cz = z - 8.8, r = cy * cy + cz * cz, gl = Math.exp(-((x - 4.3) * (x - 4.3) + r) / 11.5) * sh.cheek, gr = Math.exp(-((x + 4.3) * (x + 4.3) + r) / 11.5) * sh.cheek; dx += (gl - gr) * 0.8; dz += (gl + gr) * 0.6; }
    w = smooth(-7, -11, y) * (1 - smooth(4.5, 8.5, z)); dx += w * (sh.neck - 1) * x; dz += w * (sh.neck - 1) * (z + 1); // neck
    for (const sg of [-1, 1]) { const cx = x - sg * 8.1, cy = y - 1.3, cz = z - 0.8; w = radial(Math.hypot(cx, cy, cz), sh.ear, 3, 4.6); const k = w * (sh.ear - 1); dx += k * cx; dy += k * cy; dz += k * cz; }
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
    const m = hits.length, tri = head.base.tri, src = head.src, P = head.rest;
    this.head = head; this.tv = new Int32Array(m * 3); this.bw = new Float32Array(m * 3); this.h = new Float32Array(m);
    const area = (Q, a, b, c) => { const ux = Q[b] - Q[a], uy = Q[b + 1] - Q[a + 1], uz = Q[b + 2] - Q[a + 2], vx = Q[c] - Q[a], vy = Q[c + 1] - Q[a + 1], vz = Q[c + 2] - Q[a + 2]; return Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx); };
    for (let i = 0; i < m; i++) {
      const { t, b } = hits[i], a0 = tri[t * 3], a1 = tri[t * 3 + 1], a2 = tri[t * 3 + 2];
      this.tv[i * 3] = a0; this.tv[i * 3 + 1] = a1; this.tv[i * 3 + 2] = a2; this.bw.set(b, i * 3);
      this.h[i] = heights[i] * Math.sqrt(area(P, a0 * 3, a1 * 3, a2 * 3) / (area(src, a0 * 3, a1 * 3, a2 * 3) || 1)); // the distance grows with the skin
    }
    this.pos = new Float32Array(m * 3); this.nrm = new Float32Array(m * 3); this.index = index;
    const g = (this.geometry = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('normal', new THREE.BufferAttribute(this.nrm, 3).setUsage(THREE.DynamicDrawUsage));
    g.setIndex(new THREE.BufferAttribute(index, 1));
    this.mesh = new THREE.Mesh(g, material); this.mesh.frustumCulled = false; this.mesh.castShadow = true; this.mesh.receiveShadow = true;
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
    if (sh.noseSmooth > 0) this._iron(src, sh.noseSmooth, (x, y, z) => 1 - smooth(2.2, 3.6, Math.hypot(x, y + 0.2, z - 11.3)));
    const W = (this.warp = makeWarp(base, sh)), o = [0, 0, 0], q = [0, 0, 0];
    const P = (this.rest = new Float32Array(n * 3));
    for (let i = 0; i < n; i++) { W(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], o); P[i * 3] = o[0]; P[i * 3 + 1] = o[1]; P[i * 3 + 2] = o[2]; }
    const P0 = sh.relax > 0 ? Float32Array.from(P) : P; // where the change of proportions alone puts each vertex
    if (sh.relax > 0) this._iron(P, sh.relax, () => 1); // ripples the change of proportions left in the skin
    this.cur = Float32Array.from(P);
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
    this.shapes = base.shapes.map((s) => {
      const d = new Float32Array(s.d.length);
      for (let j = 0; j < s.idx.length; j++) {
        const v = s.idx[j], i = v * 3, k = j * 3, we = exact[v], wi = (1 - we) * iso[v];
        d[k] = wi * s.d[k]; d[k + 1] = wi * s.d[k + 1]; d[k + 2] = wi * s.d[k + 2];
        if (we > 0) { W(src[i] + s.d[k], src[i + 1] + s.d[k + 1], src[i + 2] + s.d[k + 2], q); d[k] += we * (q[0] - P0[i]); d[k + 1] += we * (q[1] - P0[i + 1]); d[k + 2] += we * (q[2] - P0[i + 2]); }
      }
      return { name: s.name, idx: s.idx, d };
    });
    if (sh.calm > 0) this._calm(sh.calm);
    this.w = new Float32Array(this.shapes.length); this._lastW = new Float32Array(this.shapes.length).fill(NaN);
    if (design.pose) { this.restPose = new Float32Array(this.shapes.length); for (const k in design.pose) { const i = base.index[k]; if (i !== undefined) this.restPose[i] = design.pose[k]; } this.w.set(this.restPose); }
    const g = (this.geometry = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.cur, 3).setUsage(THREE.DynamicDrawUsage));
    this.normal = new Float32Array(n * 3); g.setAttribute('normal', new THREE.BufferAttribute(this.normal, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(this._colors(), 3)); g.setAttribute('ao', new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
    g.setIndex(new THREE.BufferAttribute(base.tri, 1));
    const R = base.meta.regions, G = base.meta.groups; this.materials = this._materials(R);
    R.forEach((r, i) => { if (!(design.hide || []).includes(r)) g.addGroup(G[i][0] * 3, G[i][1] * 3, i); });
    this.mesh = new THREE.Mesh(g, this.materials); this.mesh.frustumCulled = false; this.mesh.castShadow = true; this.mesh.receiveShadow = true;
    this.root = new THREE.Group(); this.root.add(this.mesh);
    this.eyes = [0, 1].map((e) => this._eye(e)); this.eyes.forEach((e) => this.root.add(e.mesh));
    if (design.glint !== false) for (const e of [0, 1]) { this.root.add(this._glint(e, 13, 9, 5.5)); this.root.add(this._glint(e, -12, -11, 2.4)); }
    this.bound = []; this.sculpts = [];
    if (design.lash !== false) for (const e of [0, 1]) { const l = this._lash(e, design.lash || {}); this.bound.push(l); this.root.add(l.mesh); }
    if (design.brow !== false) for (const side of [1, -1]) { const b = this._brow(side, design.brow || {}); this.bound.push(b); this.root.add(b.mesh); }
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
  // Ironing the skin: each pass moves a vertex towards the middle of its neighbours and back out again, so the large
  // forms keep their size while folds and ripples go. `amount(i)` says how much each vertex takes part (0 to 1). The rims
  // of the lids and the line where the lips meet never move; they have to fit the eyes and the teeth.
  _iron(P, passes, amount) {
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
    for (let i = 0; i < n; i++) { const k = this._keep[i] * (this._keep[i] > 0 ? amount(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]) : 0); if (k > 0.01) { list.push(i); wt.push(k); } }
    const Q = new Float32Array(list.length * 3);
    const pass = (f) => {
      for (let k = 0; k < list.length; k++) { const i = list[k], a = nb[i]; let x = 0, y = 0, z = 0; for (let j = 0; j < a.length; j++) { const v = a[j] * 3; x += P[v]; y += P[v + 1]; z += P[v + 2]; }
        const g = (f * wt[k]) / a.length, o = i * 3, q = k * 3; Q[q] = P[o] + g * (x - a.length * P[o]); Q[q + 1] = P[o + 1] + g * (y - a.length * P[o + 1]); Q[q + 2] = P[o + 2] + g * (z - a.length * P[o + 2]); }
      for (let k = 0; k < list.length; k++) { const o = list[k] * 3, q = k * 3; P[o] = Q[q]; P[o + 1] = Q[q + 1]; P[o + 2] = Q[q + 2]; }
    };
    for (let k = 0; k < passes; k++) { pass(0.5); pass(-0.53); }
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
  // A sculpted shape (hair, a hat) described on the published head: it is made into a mesh there and taken through the
  // character's proportions like everything else, so it fits whatever head the character has.
  addSculpt(field, material, cell = 0.4) {
    const t0 = performance.now(), m = polygonise(field, cell), o = [0, 0, 0], P = m.pos;
    for (let i = 0; i < P.length; i += 3) { this.warp(P[i], P[i + 1], P[i + 2], o); P[i] = o[0]; P[i + 1] = o[1]; P[i + 2] = o[2]; }
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
    const A = d.in || [1.25, 5.0], B = d.top || [4.0, 5.8], C = d.out || [5.9, 5.05], th = d.thick || [0.55, 0.5, 0.12], lift = d.lift ?? 0.13;
    const nT = 22, nS = 6, hits = [], hs = [], idx = [];
    // parabola through the three points, y as a function of x
    const yOf = (x) => (A[1] * (x - B[0]) * (x - C[0])) / ((A[0] - B[0]) * (A[0] - C[0])) + (B[1] * (x - A[0]) * (x - C[0])) / ((B[0] - A[0]) * (B[0] - C[0])) + (C[1] * (x - A[0]) * (x - B[0])) / ((C[0] - A[0]) * (C[0] - B[0]));
    for (let i = 0; i <= nT; i++) {
      const t = i / nT, x = A[0] + (C[0] - A[0]) * t, y = yOf(x), dydx = (yOf(x + 0.01) - yOf(x - 0.01)) / 0.02, l = Math.hypot(1, dydx), nx = -dydx / l, ny = 1 / l;
      const tb = (C[0] - B[0] > 0 ? (B[0] - A[0]) / (C[0] - A[0]) : 0.6), half = 0.5 * (t < tb ? th[0] + (th[1] - th[0]) * smooth(0, tb, t) : th[1] + (th[2] - th[1]) * smooth(tb, 1, t)) * Math.sqrt(Math.min(1, t / 0.06 + 0.25)) * (1 - 0.6 * smooth(0.9, 1, t));
      for (let j = 0; j <= nS; j++) {
        const u = (j / nS) * 2 - 1, px = (x + nx * half * u) * side, py = y + ny * half * u;
        const hit = this.skin.front(px, py) || this.skin.near(px, py, 9);
        hits.push(hit); hs.push(lift * Math.sqrt(Math.max(0, 1 - u * u)) * Math.min(1, t / 0.05 + 0.3) * (1 - 0.7 * smooth(0.85, 1, t)) - 0.02);
      }
    }
    for (let i = 0; i < nT; i++) for (let j = 0; j < nS; j++) { const a = i * (nS + 1) + j, b = a + 1, c = a + nS + 1, e = c + 1; if (side > 0) idx.push(a, c, b, b, c, e); else idx.push(a, b, c, b, e, c); }
    const mat = this._browMat || (this._browMat = new THREE.MeshStandardMaterial({ color: d.color || this.design.hair || '#3a2a20', roughness: 0.75 }));
    return new Bound(this, hits, hs, Uint16Array.from(idx), mat);
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
    return { mesh, dir, uv: geo.attributes.uv };
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
  look(yaw, pitch, which = -1) {
    const a = (yaw * Math.PI) / 180, b = (pitch * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
    // right and up of the eye when it looks along (sa cb, sb, ca cb)
    const rx = ca, rz = -sa, ux = -sa * sb, uy = cb, uz = -ca * sb;
    for (let e = 0; e < 2; e++) { if (which >= 0 && which !== e) continue; const { dir, uv } = this.eyes[e], t = uv.array;
      for (let i = 0, j = 0; i < dir.length; i += 3, j += 2) { t[j] = 0.5 + 0.5 * (rx * dir[i] + rz * dir[i + 2]); t[j + 1] = 0.5 - 0.5 * (ux * dir[i] + uy * dir[i + 1] + uz * dir[i + 2]); }
      uv.needsUpdate = true; }
  }
  update(force = false) {
    const w = this.w, lw = this._lastW; let same = !force;
    if (same) for (let i = 0; i < w.length; i++) if (w[i] !== lw[i]) { same = false; break; }
    if (same) return;
    lw.set(w);
    const cur = this.cur; cur.set(this.rest);
    for (let s = 0; s < this.shapes.length; s++) {
      const a = w[s]; if (!a) continue;
      const { idx, d } = this.shapes[s];
      for (let j = 0; j < idx.length; j++) { const i = idx[j] * 3, k = j * 3; cur[i] += a * d[k]; cur[i + 1] += a * d[k + 1]; cur[i + 2] += a * d[k + 2]; }
    }
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
