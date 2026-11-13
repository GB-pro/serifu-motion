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
};

export function createCharacter(key) { return new Character(DESIGNS[key]); }
