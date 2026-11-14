// Serifu Motion — original stylized characters (not based on any existing franchise character).
import { Character } from './rig.js';

function hairMat(ch, THREE, color, rough = 0.5) {
  return new THREE.MeshPhysicalMaterial({ color, roughness: rough, sheen: 1, sheenColor: new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.5), sheenRoughness: 0.35, clearcoat: 0.3, clearcoatRoughness: 0.5 });
}
// ellipsoid helper on the face group
function cap(ch, THREE, mat, { s = [1.07, 1.1, 1.06], line = 0.85, cover = 2.2, y = 0.05, z = -0.04 } = {}) {
  const tilt = -(cover - line);
  // hair cap: a partial sphere tilted back so the hairline sits high on the forehead
  const g = new THREE.SphereGeometry(1, 72, 48, 0, Math.PI * 2, 0, cover);
  const m = new THREE.Mesh(g, mat); m.rotation.x = tilt; m.scale.set(...s); m.position.set(0, y, z);
  mat.side = THREE.DoubleSide; ch.face.add(m); return m;
}
// dyn: {pivot:[x,y,z], stiff, damp, gain, grav, max} makes the piece lag and settle after head moves (follow-through)
function blob(ch, THREE, mat, pos, scale, rot = [0, 0, 0], seg = 48, dyn = null) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, seg, Math.round(seg * 0.75)), mat);
  m.position.set(...pos); m.scale.set(...scale); m.rotation.set(...rot); ch.face.add(m);
  if (dyn) ch.addDyn(m, dyn.pivot, dyn);
  return m;
}
const LOCK = { stiff: 170, damp: 7, gain: 1.0, grav: 0.65, max: 0.45 };   // hanging hair
const TUFT = { stiff: 900, damp: 13, gain: 0.9, max: 0.3 };                // short springy hair
const SOFT = { stiff: 420, damp: 11, gain: 0.45, max: 0.2 };               // heavy hair mass


// ---- block-head characters: hair pieces follow the superellipsoid head (Character.shell) instead of a sphere ----
// a hair cap over a block head: the part of the head above a tilted hairline, pushed out by off
function blockCap(ch, THREE, mat, { line = 0.85, cover = 2.2, off = 0.05, seg = 96 } = {}) {
  const g = new THREE.SphereGeometry(1, seg, Math.round(seg * 0.6), 0, Math.PI * 2, 0, cover);
  g.rotateX(-(cover - line)); ch.shell(g, off); mat.side = THREE.DoubleSide;
  const m = new THREE.Mesh(g, mat); ch.face.add(m); return m;
}
// a point on the head surface in direction (yaw, pitch) (radians; yaw 0 = straight ahead), pushed out by off
function onHead(ch, yaw, pitch, off = 0) {
  const u = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)], r = ch.surfR(...u) + off;
  return [u[0] * r, u[1] * r, u[2] * r];
}
// a spike of hair rooted on the head at (yaw, pitch), leaning along the surface: tilt forward/back (tx) and sideways (tz)
function spike(ch, THREE, mat, yaw, pitch, { len = 0.45, r = 0.16, tx = 0, tz = 0, sink = 0.1, dyn = TUFT, flat = 0.75, k = 1 } = {}) {
  len *= k; r *= k;
  const geo = new THREE.ConeGeometry(r, len, 20); geo.translate(0, len / 2, 0);
  const m = new THREE.Mesh(geo, mat); const base = onHead(ch, yaw, pitch, -sink);
  const n = new THREE.Vector3(...base).normalize();
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);
  m.rotateX(tx); m.rotateZ(tz); m.scale.set(1, 1, flat); m.position.set(...base); ch.face.add(m);
  if (dyn) ch.addDyn(m, base, dyn);
  return m;
}

export const DESIGNS = {
  hana: {
    name: 'Hana', label: { ja: 'ハナ', en: 'Hana' },
    skin: '#f6c9a8',
    head: { sx: 1.0, sy: 1.08, sz: 0.97, jaw: 0.34, cheek: 0.06, chin: 0.03, scale: 1.0 },
    eyes: { r: 0.215, sep: 0.37, y: -0.02, depth: 0.42, iris: 0.62, color: '#3d8f5a', pupil: 0.4, lash: '#2a1712', lashW: 0.085, lashFlick: true, splay: 0.14 },
    brows: { color: "#5b3420", inner: 0.12, outer: 0.58, y: 0.28, arch: 0.05, tilt: 0.03, thick: 0.03 },
    nose: { y: -0.24, r: 0.06, sx: 1.25, sy: 0.85, sz: 0.9 },
    mouth: { y: -0.47, w: 0.44, lip: '#e07f7c', lipT: 0.032, bow: 0.008, open: 1.0 },
    cheeks: { blush: '#ff7f86', blushAlpha: 0.22 },
    body: { height: 1.25, shoulder: 0.56, waist: 0.42, color: '#f2c14e', collar: '#ffffff', sleeve: '#f2c14e', front: [{ type: 'bow', color: '#e2557a' }, { type: 'buttons', color: '#fff6dc', n: 3, y: -0.42, dy: 0.21 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#6b3a1f');
      cap(ch, THREE, m, { s: [1.08, 1.12, 1.07], line: 0.85 });
      blob(ch, THREE, m, [0, -0.2, -0.35], [1.12, 0.8, 0.85], [0, 0, 0], 48, { pivot: [0, 0.45, -0.3], ...SOFT }); // bob back
      for (const s of [-1, 1]) blob(ch, THREE, m, [s * 0.88, -0.32, 0.05], [0.28, 0.6, 0.45], [0, 0, s * 0.15], 48, { pivot: [s * 0.86, 0.22, 0.02], ...LOCK }); // side locks
      // bangs: swoosh across forehead
      blob(ch, THREE, m, [-0.28, 0.8, 0.45], [0.62, 0.2, 0.4], [0.6, 0.2, 0.3], 48, { pivot: [-0.2, 0.98, 0.2], ...TUFT, gain: 0.5 });
      blob(ch, THREE, m, [0.36, 0.82, 0.42], [0.5, 0.18, 0.4], [0.6, -0.2, -0.4], 48, { pivot: [0.3, 0.98, 0.2], ...TUFT, gain: 0.5 });
      // hair clip
      const clip = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.025, 10, 20), new THREE.MeshStandardMaterial({ color: '#ff5d7a', roughness: 0.3 }));
      clip.position.set(0.62, 0.72, 0.52); clip.rotation.set(0.3, -0.5, 0.6); ch.face.add(clip);
    },
  },
  kenta: {
    name: 'Kenta', label: { ja: 'ケンタ', en: 'Kenta' },
    skin: '#e8b48f',
    head: { sx: 1.02, sy: 1.04, sz: 0.98, jaw: 0.26, cheek: 0.04, chin: 0.04, scale: 1.0 },
    eyes: { r: 0.2, sep: 0.38, y: -0.02, depth: 0.45, iris: 0.6, color: '#6b4426', pupil: 0.42, lash: '#1a1210', lashW: 0.06, splay: 0.12 },
    brows: { color: '#1d1714', inner: 0.11, outer: 0.6, y: 0.31, arch: 0.03, tilt: -0.01, thick: 0.042 },
    nose: { y: -0.23, r: 0.075, sx: 1.2, sy: 0.9, sz: 1.0 },
    mouth: { y: -0.48, w: 0.46, lip: '#c98472', lipT: 0.022, bow: 0.004, open: 1.0 },
    cheeks: { blush: '#ff9f86', blushAlpha: 0.1 },
    body: { height: 1.3, shoulder: 0.64, waist: 0.5, color: '#3d6fd8', collar: '#2c55ad', sleeve: '#3d6fd8', front: [{ type: 'pocket', color: '#3160c4', y: -0.86, w: 0.56, h: 0.26 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#2a2220', 0.75);
      cap(ch, THREE, m, { s: [1.07, 1.08, 1.07], line: 0.8, y: 0.07 });
      // big swept fringe + crown tufts made of soft ellipsoids
      blob(ch, THREE, m, [-0.3, 0.78, 0.5], [0.55, 0.22, 0.36], [0.7, 0.3, 0.5], 48, { pivot: [-0.25, 0.98, 0.25], ...TUFT, gain: 0.55 });
      blob(ch, THREE, m, [0.15, 0.84, 0.5], [0.5, 0.2, 0.34], [0.7, -0.1, 0.25], 48, { pivot: [0.1, 1.0, 0.25], ...TUFT, gain: 0.55 });
      blob(ch, THREE, m, [0.5, 0.72, 0.45], [0.36, 0.2, 0.3], [0.6, -0.4, -0.3], 48, { pivot: [0.45, 0.9, 0.25], ...TUFT, gain: 0.55 });
      for (const [x, y, z, rz] of [[-0.35, 1.02, 0.15, 0.5], [0.05, 1.1, 0.1, 0.1], [0.4, 1.0, 0.1, -0.4], [-0.1, 1.08, -0.3, 0.2]]) {
        const c = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.42, 28), m);
        c.scale.set(1, 1, 0.7); c.position.set(x, y, z); c.rotation.set(-0.3, 0, rz); ch.face.add(c);
        ch.addDyn(c, [x * 0.9, y - 0.2, z], TUFT);
      }
    },
    extras: (ch, THREE) => { // hoodie strings
      const mat = new THREE.MeshStandardMaterial({ color: '#f4f4f4', roughness: 0.6 });
      for (const s of [-1, 1]) { const c = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.35, 8), mat); c.position.set(s * 0.1, -0.22, 0.34); ch.torso.add(c); ch.addDyn(c, [s * 0.1, -0.05, 0.34], { stiff: 110, damp: 4, gain: 1.4, grav: 1, max: 0.6 }); }
      const hood = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.11, 14, 32), new THREE.MeshPhysicalMaterial({ color: '#2c55ad', roughness: 0.85, sheen: 1 }));
      hood.rotation.x = Math.PI / 2 - 0.25; hood.position.set(0, 0.02, -0.08); hood.scale.set(1.15, 1, 1); ch.torso.add(hood);
    },
  },
  moku: {
    name: 'Moku', label: { ja: 'モクじい', en: 'Grandpa Moku' },
    skin: '#e2a88a',
    head: { sx: 1.05, sy: 1.0, sz: 0.98, jaw: 0.18, cheek: 0.1, chin: 0.02, scale: 1.0 },
    eyes: { r: 0.17, sep: 0.36, y: 0.0, depth: 0.5, iris: 0.6, color: '#4b6fa8', pupil: 0.44, lash: '#6b5a52', lashW: 0.04, splay: 0.1 },
    brows: { color: '#cfcac2', inner: 0.12, outer: 0.66, y: 0.3, arch: 0.06, tilt: 0.06, thick: 0.065 },
    nose: { y: -0.2, r: 0.12, sx: 1.05, sy: 0.9, sz: 1.0 },
    mouth: { y: -0.5, w: 0.44, lip: '#c07a6c', lipT: 0.02, bow: 0.002, open: 0.9 },
    cheeks: { blush: '#ff8a7a', blushAlpha: 0.2 },
    body: { height: 1.25, shoulder: 0.66, waist: 0.58, color: '#7a5a8c', collar: '#e9e2d0', sleeve: '#7a5a8c', front: [{ type: 'tie', color: '#d9a441', len: 0.5 }, { type: 'buttons', color: '#e9e2d0', n: 2, y: -0.8, dy: 0.2, r: 0.03 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#efefea', 0.7);
      for (const s of [-1, 1]) {
        blob(ch, THREE, m, [s * 0.86, 0.05, -0.2], [0.3, 0.42, 0.55], [0, 0, s * 0.2], 48, { pivot: [s * 0.8, 0.4, -0.2], ...SOFT, gain: 0.7 });
        blob(ch, THREE, m, [s * 0.7, -0.15, -0.5], [0.35, 0.38, 0.45], [0, 0, 0], 48, { pivot: [s * 0.65, 0.2, -0.5], ...SOFT, gain: 0.7 });
      }
      blob(ch, THREE, m, [0, -0.05, -0.75], [0.75, 0.45, 0.35]);
      // mustache (moves with upper lip)
      const mus = new THREE.Group(); ch.face.add(mus); ch.mustache = mus;
      for (const s of [-1, 1]) {
        const b = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), m);
        b.scale.set(0.2, 0.075, 0.1); b.rotation.z = -s * 0.25; b.position.set(s * 0.15, -0.37, ch.surfZ(s * 0.15, -0.37) + 0.02); mus.add(b);
        ch.addDyn(b, [s * 0.02, -0.3, ch.surfZ(0, -0.33)], { stiff: 520, damp: 9, gain: 0.8, max: 0.3 });
      }
      // round glasses
      const gm = new THREE.MeshStandardMaterial({ color: '#3b2d2a', roughness: 0.3, metalness: 0.5 });
      for (const s of [-1, 1]) { const r = new THREE.Mesh(new THREE.TorusGeometry(0.21, 0.014, 8, 40), gm); r.position.set(s * 0.36, 0.0, ch.surfZ(s * 0.36, 0) + 0.11); ch.face.add(r); }
      const br = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.16, 6), gm); br.rotation.z = Math.PI / 2; br.position.set(0, 0.03, ch.surfZ(0, 0.03) + 0.06); ch.face.add(br);
    },
  },
  mimi: {
    name: 'Mimi', label: { ja: 'ミミ', en: 'Mimi' },
    skin: '#ffe3cf',
    head: { sx: 1.05, sy: 1.0, sz: 0.95, jaw: 0.3, cheek: 0.1, chin: 0.0, scale: 1.0 },
    ears: false,
    eyes: { r: 0.24, sep: 0.39, y: -0.06, depth: 0.42, iris: 0.66, color: '#8a4fd0', pupil: 0.38, lash: '#2a1730', lashW: 0.08, lashFlick: true, splay: 0.15 },
    brows: { color: '#3a2a55', inner: 0.14, outer: 0.55, y: 0.27, arch: 0.05, tilt: 0.02, thick: 0.026 },
    nose: { y: -0.27, r: 0.045, sx: 1.4, sy: 0.8, sz: 0.8 },
    mouth: { y: -0.48, w: 0.38, lip: '#f08a98', lipT: 0.026, bow: 0.006, open: 1.1 },
    cheeks: { blush: '#ff6f9a', blushAlpha: 0.3 },
    body: { height: 1.1, shoulder: 0.52, waist: 0.4, color: '#46c2b4', collar: '#ffffff', sleeve: '#46c2b4', front: [{ type: 'emblem', color: '#ffffff', inner: '#ffd24a', y: -0.4, r: 0.13 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#4a3570');
      cap(ch, THREE, m, { s: [1.1, 1.07, 1.05], line: 1.0 });
      blob(ch, THREE, m, [0, -0.1, -0.4], [1.08, 0.85, 0.75]);
      // twin buns
      for (const s of [-1, 1]) { blob(ch, THREE, m, [s * 0.8, 0.75, -0.2], [0.32, 0.32, 0.32], [0, 0, 0], 48, { pivot: [s * 0.6, 0.52, -0.2], stiff: 420, damp: 7, gain: 1.1, max: 0.35 }); }
      // fringe
      for (let i = -2; i <= 2; i++) blob(ch, THREE, m, [i * 0.21, 0.7 - Math.abs(i) * 0.05, 0.6 - Math.abs(i) * 0.1], [0.19, 0.2, 0.14], [0.5, 0, i * 0.15], 48, { pivot: [i * 0.2, 0.92, 0.4 - Math.abs(i) * 0.08], ...TUFT, gain: 0.45 });
      // star clip
      const star = new THREE.Mesh(new THREE.OctahedronGeometry(0.07), new THREE.MeshStandardMaterial({ color: '#ffd23f', roughness: 0.3, metalness: 0.2 }));
      star.position.set(-0.55, 0.62, 0.62); ch.face.add(star);
    },
  },

  // ---- block-head cast: toy-like heads (rounded blocks), simple drawn-on faces, box bodies ----
  riku: {
    name: 'Riku', label: { ja: 'リク', en: 'Riku' },
    skin: '#f3c79f', ears: false,
    head: { sx: 0.98, sy: 0.9, sz: 0.88, jaw: 0, cheek: 0, chin: 0, scale: 1.0, block: { p: 3.2, q: 5 } },
    eyes: { r: 0.17, sep: 0.31, y: 0.0, depth: 0.5, iris: 0.66, color: '#1b120e', pupil: 0.6, lash: '#1b120e', lashW: 0.06, splay: 0.05, oval: 1.35, set: 0.6, gaze: 1.35 }, // white around a big dark pupil, so where he looks reads
    brows: { color: '#4a2412', inner: 0.12, outer: 0.44, y: 0.3, arch: 0.04, tilt: 0.0, thick: 0.05 },
    nose: { y: -0.2, r: 0.04, hide: true },
    mouth: { y: -0.38, w: 0.5, lip: '#a85a50', lipT: 0.012, bow: 0.0, open: 1.05 },
    cheeks: { blush: '#ff9a86', blushAlpha: 0.0 },
    body: { block: true, height: 1.25, shoulder: 0.62, depth: 0.56, armW: 0.42, color: '#232326', sleeve: '#232326', front: [{ type: 'pocket', color: '#2c2c30', y: -0.92, w: 0.66, h: 0.26 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#7b3b17', 0.55);
      blockCap(ch, THREE, m, { line: 0.92, cover: 1.6, off: 0.05 });
      // a messy crown of spikes, the front ones swept up and to the side
      for (const [yw, pt, len, r, tx, tz] of [
        [-0.55, 0.62, 0.42, 0.17, 0.5, 0.5], [-0.15, 0.7, 0.48, 0.19, 0.6, 0.25], [0.25, 0.7, 0.5, 0.19, 0.55, -0.2], [0.6, 0.6, 0.42, 0.17, 0.5, -0.55],
        [-0.9, 0.95, 0.42, 0.18, 0.1, 0.4], [-0.3, 1.05, 0.46, 0.2, 0.0, 0.1], [0.35, 1.05, 0.46, 0.2, -0.1, -0.15], [0.95, 0.9, 0.42, 0.18, 0.1, -0.45],
        [-1.6, 0.8, 0.36, 0.17, -0.2, 0.5], [1.6, 0.8, 0.36, 0.17, -0.2, -0.5], [2.6, 0.9, 0.4, 0.18, -0.4, 0.2], [3.4, 0.9, 0.4, 0.18, -0.4, -0.2],
        [-1.25, 0.35, 0.3, 0.14, 0.2, 0.9], [1.25, 0.35, 0.3, 0.14, 0.2, -0.9]]) spike(ch, THREE, m, yw, pt, { len, r, tx, tz, k: 1.45 });
      // fringe locks over the forehead
      for (const [yw, tz] of [[-0.32, 0.9], [0.05, 0.6], [0.4, 0.35]]) spike(ch, THREE, m, yw, 0.55, { len: 0.46, r: 0.19, tx: 1.45, tz, sink: 0.02, flat: 0.5 });
    },
    extras: (ch, THREE) => { // hoodie strings
      const mat = new THREE.MeshStandardMaterial({ color: '#f1f1f1', roughness: 0.6 });
      for (const s of [-1, 1]) { const c = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.38, 8), mat); c.position.set(s * 0.12, -0.24, 0.33); ch.torso.add(c); ch.addDyn(c, [s * 0.12, -0.05, 0.33], { stiff: 110, damp: 4, gain: 1.4, grav: 1, max: 0.6 }); }
      const hood = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.1, 12, 32), new THREE.MeshPhysicalMaterial({ color: '#1b1b1e', roughness: 0.9, sheen: 1 }));
      hood.rotation.x = Math.PI / 2; hood.position.set(0, 0.02, -0.04); hood.scale.set(1.2, 0.85, 1); ch.torso.add(hood);
    },
  },
  gen: {
    name: 'Gen', label: { ja: 'ゲン', en: 'Gen' },
    skin: '#e9b98c', ears: false,
    head: { sx: 0.98, sy: 0.92, sz: 0.88, jaw: 0, cheek: 0, chin: 0, scale: 1.0, block: { p: 3.2, q: 5 } },
    eyes: { r: 0.12, sep: 0.3, y: 0.0, depth: 0.5, iris: 0.92, color: '#2a1f18', pupil: 0.55, lash: '#2a1f18', lashW: 0.09, splay: 0.05, oval: 0.85, set: 0.6 },
    brows: { color: '#8f8a86', inner: 0.1, outer: 0.46, y: 0.2, arch: 0.02, tilt: 0.04, thick: 0.03 },
    nose: { y: -0.16, r: 0.05, sx: 1.0, sy: 1.2, sz: 0.6 },
    mouth: { y: -0.36, w: 0.4, lip: '#9d6a58', lipT: 0.01, bow: 0.0, open: 0.9 },
    cheeks: { blush: '#ff8a7a', blushAlpha: 0.0 },
    body: { block: true, height: 1.3, shoulder: 0.64, depth: 0.58, armW: 0.44, color: '#2a2a2c', sleeve: '#ecebe6', hand: 0.18,
      front: [{ type: 'tie', color: '#9a9a9c', len: 0.5 }, { type: 'buttons', color: '#c9a35a', n: 4, y: -0.42, dy: 0.17, r: 0.025 }] },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#d9d9dc', 0.5);
      blockCap(ch, THREE, m, { line: 0.62, cover: 1.45, off: 0.05 });
      // swept back and up: the spikes lean toward the back of the head
      for (const [yw, pt, len, r, tx, tz] of [
        [-0.45, 0.62, 0.45, 0.18, -0.9, 0.35], [0.0, 0.66, 0.5, 0.2, -1.0, 0.0], [0.45, 0.62, 0.45, 0.18, -0.9, -0.35],
        [-0.8, 0.9, 0.42, 0.17, -0.8, 0.5], [-0.25, 0.98, 0.45, 0.19, -0.9, 0.15], [0.25, 0.98, 0.45, 0.19, -0.9, -0.15], [0.8, 0.9, 0.42, 0.17, -0.8, -0.5],
        [-1.5, 0.75, 0.38, 0.17, -0.7, 0.5], [1.5, 0.75, 0.38, 0.17, -0.7, -0.5], [2.6, 0.75, 0.36, 0.17, -0.5, 0.3], [3.6, 0.75, 0.36, 0.17, -0.5, -0.3],
        [-1.4, 0.3, 0.28, 0.13, -0.6, 0.8], [1.4, 0.3, 0.28, 0.13, -0.6, -0.8]]) spike(ch, THREE, m, yw, pt, { len, r, tx, tz, k: 1.5, dyn: { ...TUFT, gain: 0.5 } });
    },
    extras: (ch, THREE) => { // white long coat over the shoulders with a dark fur collar
      const coat = new THREE.MeshPhysicalMaterial({ color: '#ecebe6', roughness: 0.85, sheen: 0.6 });
      for (const s of [-1, 1]) { const f = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.32, 0.04), coat); f.position.set(s * 0.45, -0.66, 0.3); ch.torso.add(f); }
      const fur = hairMat(ch, THREE, '#1f3a26', 0.9);
      let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      for (const s of [-1, 1]) {
        const b = new THREE.Mesh(new THREE.SphereGeometry(0.3, 18, 12), fur); b.scale.set(1.25, 0.55, 1.2); b.position.set(s * 0.5, 0.0, 0.05); ch.torso.add(b);
        for (let i = 0; i < 9; i++) { const x = s * (0.26 + 0.055 * i), tuft = new THREE.Mesh(new THREE.ConeGeometry(0.1, 0.4, 8), fur);
          tuft.position.set(x, -0.02 + rnd() * 0.06, 0.18 + rnd() * 0.18); tuft.rotation.set(-0.6 + rnd() * 0.8, 0, -s * (0.6 + rnd() * 0.8)); ch.torso.add(tuft); }
      }
    },
  },
  yuki: {
    name: 'Yuki', label: { ja: 'ユキ', en: 'Yuki' },
    skin: '#ffe0cf', ears: false,
    head: { sx: 1.0, sy: 0.92, sz: 0.88, jaw: 0, cheek: 0, chin: 0, scale: 1.0, block: { p: 2.6, q: 3.6 } },
    eyes: { r: 0.22, sep: 0.34, y: -0.06, depth: 0.42, iris: 0.8, color: '#f05a74', pupil: 0.4, lash: '#2a1a1e', lashW: 0.1, lashFlick: true, splay: 0.06, oval: 1.25 },
    brows: { color: '#d9d4d4', inner: 0.14, outer: 0.48, y: 0.25, arch: 0.04, tilt: 0.02, thick: 0.022 },
    nose: { y: -0.24, r: 0.03, sx: 1.2, sy: 0.8, sz: 0.6 },
    mouth: { y: -0.42, w: 0.3, lip: '#ef8f98', lipT: 0.02, bow: 0.004, open: 1.15 },
    cheeks: { blush: '#ff7f9a', blushAlpha: 0.32 },
    body: { block: true, height: 1.15, shoulder: 0.58, depth: 0.54, armW: 0.38, color: '#efe8de', sleeve: '#efe8de', hand: 0.16 },
    hair: (ch, THREE) => {
      const m = hairMat(ch, THREE, '#e8e7ef', 0.5);
      blockCap(ch, THREE, m, { line: 0.86, off: 0.05 });
      // long fluffy hair: a mass behind the head and locks down the sides
      blob(ch, THREE, m, [0, -0.45, -0.45], [1.05, 1.0, 0.6], [0, 0, 0], 48, { pivot: [0, 0.4, -0.4], ...SOFT });
      for (const s of [-1, 1]) {
        blob(ch, THREE, m, [s * 0.92, -0.45, 0.12], [0.22, 0.75, 0.3], [0, 0, s * 0.08], 48, { pivot: [s * 0.9, 0.3, 0.1], ...LOCK });
        blob(ch, THREE, m, [s * 0.98, -0.7, -0.25], [0.24, 0.8, 0.3], [0, 0, s * 0.12], 48, { pivot: [s * 0.95, 0.2, -0.2], ...LOCK });
      }
      // wispy fringe
      for (let i = -3; i <= 3; i++) spike(ch, THREE, m, i * 0.2, 0.5 + Math.abs(i) * 0.02, { len: 0.62 - Math.abs(i) * 0.04, r: 0.19, tx: Math.PI - 0.22, tz: -i * 0.1, sink: -0.03, flat: 0.4, dyn: { ...TUFT, gain: 0.45 } });
      spike(ch, THREE, m, 0.1, 1.3, { len: 0.4, r: 0.05, tx: 0.6, tz: -0.4, dyn: { stiff: 300, damp: 5, gain: 1.2, max: 0.5 } }); // ahoge
      // cat ears with pink inside
      const pink = new THREE.MeshStandardMaterial({ color: '#f6a9b8', roughness: 0.6 });
      for (const s of [-1, 1]) {
        const e = spike(ch, THREE, m, s * 0.62, 1.0, { len: 0.56, r: 0.27, tx: -0.15, tz: -s * 0.2, sink: 0.08, flat: 0.45, dyn: { stiff: 600, damp: 9, gain: 0.6, max: 0.25 } });
        const inner = new THREE.Mesh(new THREE.ConeGeometry(0.17, 0.38, 16), pink); inner.geometry.translate(0, 0.19, 0); inner.position.set(0, 0.04, 0.07); inner.scale.set(1, 1, 0.4); e.add(inner);
      }
    },
  },
};

export function createCharacter(key) { return new Character(DESIGNS[key]); }
