// Serifu Motion — a character: the sculpted head (head.js) on a sculpted body, moved by the emo channels.
//
// Lengths inside are centimetres in the space of the head at rest (x to the character's left, y up, z forward); the
// whole figure is brought to stage units at the top (UNIT). apply(frame, dt) takes the channels of core/au.js with the
// same meaning and the same sizes as the first figures did (rig.js), so recorded emo plays on both.
import * as THREE from 'three';
import { Head, Field } from './head.js';
import { polygonise } from './sculpt.js';
import { occluded } from './bake.js';

export const UNIT = 0.092; // stage units per centimetre
const D2R = Math.PI / 180;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const AUS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU20', 'AU22', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43', 'AU45'];

export const cloth = (color, o = {}) => occluded(new THREE.MeshPhysicalMaterial({ color, roughness: o.roughness ?? 0.82, sheen: o.sheen ?? 0.7, sheenRoughness: 0.55, sheenColor: new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.45), ...(o.more || {}) }), o.shade || '#3b3a4a', 0.5);
export const hairMaterial = (color, o = {}) => occluded(new THREE.MeshPhysicalMaterial({ color, roughness: o.roughness ?? 0.5, sheen: 1, sheenColor: new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.5), sheenRoughness: 0.35, clearcoat: 0.3, clearcoatRoughness: 0.5 }), o.shade || '#2a2030', 0.55);

// a ring around a centre (collar, hood): negative inside. r: [rx, rz] of the centre line, t: thickness, tilt: how much higher the back is (per cm of z)
export const ring = (c, r, t, tilt = 0, squash = 1) => (x, y, z) => { const dx = x - c[0], dz = z - c[2], dy = (y - c[1] + tilt * dz) / squash, a = Math.hypot(dx / r[0], dz / r[1]); return Math.hypot((a - 1) * Math.min(r[0], r[1]), dy) - t; };

export class Figure {
  // design: { name, head: {design of Head}, hair(head, figure), body: { neckY, shoulder, depth, color, ... }, wear(figure, F) }
  constructor(design) {
    const d = (this.d = design), body = (this.body = { neckY: -10.4, shoulder: 8.3, depth: 4.7, chest: 1, color: '#6f8fd8', ...(d.body || {}) }), ny = body.neckY;
    this.root = new THREE.Group(); this.root.name = d.name || 'figure';
    const rig = (this.rig = new THREE.Group()); rig.scale.setScalar(UNIT); rig.position.y = -ny * UNIT; this.root.add(rig);
    const group = () => { const g = new THREE.Group(); g.matrixAutoUpdate = false; return g; };
    this.gWaist = group(); this.gNeck = group(); this.gSkull = group(); rig.add(this.gWaist); this.gWaist.add(this.gNeck); this.gNeck.add(this.gSkull);
    const head = (this.head = new Head({ ...d.head, cut: (x, y) => y < ny - 2.2 })); this.gSkull.add(head.root);
    if (d.hair) d.hair(head, this);
    // joints
    let chin = 0; { const P = head.rest, S = head.src; for (let i = 0; i < head.base.n; i++) if (Math.abs(S[i * 3]) < 1.5 && S[i * 3 + 2] > 6 && S[i * 3 + 1] < -7) chin = Math.min(chin, P[i * 3 + 1]); }
    this.pivot = { waist: new THREE.Vector3(0, ny - 9.8, -1.2), neck: new THREE.Vector3(0, ny, -1.2), skull: new THREE.Vector3(0, body.skullY ?? -2.8, -0.8), chin: new THREE.Vector3(0, chin, 2) };
    this._neck(ny);
    // body
    this.bodyMeshes = [];
    const F = this.bodyField(); this.field = F;
    this.addBody(F, cloth(body.color), body.cell ?? 0.45);
    if (d.wear) d.wear(this, F);
    this._m = { n: new THREE.Matrix4(), s: new THREE.Matrix4(), h: new THREE.Matrix4(), a: new THREE.Matrix4(), b: new THREE.Matrix4(), t: new THREE.Matrix4(), q: new THREE.Quaternion(), e: new THREE.Euler(), v: new THREE.Vector3(), one: new THREE.Vector3(1, 1, 1) };
    this._w = {}; this._shrug = 0; this._shrugDone = null; this._brow = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
    this.apply({});
  }

  // The neck: skin between the head and the shoulders. Each of its vertices follows the shoulders, the neck joint and
  // the head by shares that change smoothly along the neck, so the neck bends instead of breaking.
  _neck(ny) {
    const head = this.head, P = head.rest, n = head.base.n, list = [], wt = [];
    const top = (z) => -4.4 - 0.29 * (z + 5.3), bot = (z) => ny - 0.18 * (z + 1.2);
    for (let i = 0; i < n; i++) {
      const y = P[i * 3 + 1], z = P[i * 3 + 2], a = y - bot(z), b = top(z) - y; if (b <= 0) continue;
      let u = a <= 0 ? 0 : a / (a + b); u = u * u * (3 - 2 * u);
      list.push(i); wt.push(u * u, 2 * u * (1 - u), (1 - u) * (1 - u)); // head, neck joint, shoulders
    }
    this.neckList = Int32Array.from(list); this.neckW = Float32Array.from(wt);
    head.deform = (cur) => {
      const A = this._m.a.elements, B = this._m.b.elements, L = this.neckList, W = this.neckW, sh = this._shrug;
      for (let k = 0; k < L.length; k++) {
        const o = L[k] * 3, x = cur[o], y = cur[o + 1], z = cur[o + 2], wh = W[k * 3], wn = W[k * 3 + 1], ws = W[k * 3 + 2];
        const ys = y + sh * this.shrugAt(x, y); // on the shoulders
        cur[o] = wh * x + wn * (B[0] * x + B[4] * y + B[8] * z + B[12]) + ws * (A[0] * x + A[4] * ys + A[8] * z + A[12]);
        cur[o + 1] = wh * y + wn * (B[1] * x + B[5] * y + B[9] * z + B[13]) + ws * (A[1] * x + A[5] * ys + A[9] * z + A[13]);
        cur[o + 2] = wh * z + wn * (B[2] * x + B[6] * y + B[10] * z + B[14]) + ws * (A[2] * x + A[6] * ys + A[10] * z + A[14]);
      }
    };
  }
  // how much of a shrug reaches a point of the body: all of it at the shoulders, nothing at the breastbone and the waist
  shrugAt(x, y) { const ny = this.body.neckY; return smooth(1.2, 6.5, Math.abs(x)) * smooth(ny - 10, ny - 3.5, y); }

  // the torso as soft primitives: chest, shoulders, upper arms, cut flat below what the stage shows
  bodyField() {
    const b = this.body, ny = b.neckY, sw = b.shoulder, dp = b.depth, cz = -1.2, sy = ny - 2.5, f = new Field(), ch = b.chest;
    f.ell([0, ny - 8.2, cz + 0.3], [sw * 0.84 * ch, 8.4, dp * ch]);
    for (const s of [-1, 1]) {
      f.cone([s * 1.6, ny - 1.5, cz - 0.4], [s * sw, sy, cz], 2.2, b.arm ?? 2.6, { k: 1.6 });
      f.cone([s * sw, sy, cz], [s * (sw + 1.3), ny - 17, cz + 0.4], b.arm ?? 2.6, (b.arm ?? 2.6) * 0.86, { k: 1.4 });
    }
    f.ell([0, ny - 15.5, cz], [sw * 0.72 * (b.waist ?? 1), 7, dp * 0.92], { k: 2.2 });
    f.half([0, ny - 17.5, 0], [0, -1, 0], { mode: 'keep', k: 0.5 });
    f.ell([0, ny + 0.7, cz + 0.5], [b.open ?? 3.7, 1.9, (b.open ?? 3.7) + 0.2], { mode: 'cut', k: 0.6 }); // the opening for the neck
    return f;
  }
  // a part of the body or of the clothes: a field in the figure's centimetres, made into a mesh that rides on the torso
  addBody(field, material, cell = 0.4, box) {
    const m = polygonise(field, cell, box), g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(m.pos, 3).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('normal', new THREE.BufferAttribute(m.nrm, 3)); g.setIndex(new THREE.BufferAttribute(m.idx, 1));
    g.setAttribute('ao', new THREE.BufferAttribute(new Float32Array(m.pos.length / 3).fill(1), 1));
    const mesh = new THREE.Mesh(g, material); mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false; this.gWaist.add(mesh);
    const ws = new Float32Array(m.pos.length / 3); for (let i = 0; i < ws.length; i++) ws[i] = this.shrugAt(m.pos[i * 3], m.pos[i * 3 + 1]);
    this.bodyMeshes.push({ mesh, base: Float32Array.from(m.pos), ws }); return mesh;
  }
  // a ready-made mesh (a button) that rides on the torso; at: [x, y, z] in centimetres
  addProp(mesh, at) { mesh.position.set(at[0], at[1], at[2]); mesh.castShadow = true; mesh.receiveShadow = true; this.gWaist.add(mesh); (this.props || (this.props = [])).push({ mesh, y: at[1], ws: this.shrugAt(at[0], at[1]) }); return mesh; }
  // the front of the body at (x, y): z of its surface
  front(x, y, field = this.field) { let lo = -1, hi = 14; if (field.value(x, y, lo) > 0) return lo; for (let k = 0; k < 22; k++) { const m = (lo + hi) / 2; if (field.value(x, y, m) < 0) lo = m; else hi = m; } return hi; }

  // soft shadows, measured once the figure is complete
  bake(renderer) { this.apply({}); this.head.bake(renderer, this.rig, this.bodyMeshes.map((b) => b.mesh)); this.apply({}); }

  _setShrug(s) {
    if (this._shrugDone !== null && Math.abs(s - this._shrugDone) < 0.012) return; this._shrugDone = s;
    for (const b of this.bodyMeshes) { const p = b.mesh.geometry.attributes.position, a = p.array; for (let i = 0; i < b.ws.length; i++) a[i * 3 + 1] = b.base[i * 3 + 1] + s * b.ws[i]; p.needsUpdate = true; b.mesh.geometry.computeVertexNormals(); }
    for (const p of this.props || []) p.mesh.position.y = p.y + s * p.ws;
  }

  // ---------- one emo frame ----------
  // a: { AU1..AU45, gazeX, gazeY, headPitch, headYaw, headRoll, headZ, torsoRoll, torsoShift, torsoShrug, torsoLean,
  //      AU43, browAsym, mouthAsym, eyeScale, pupil, squash, headX, headY, torsoYaw, torsoY }
  // dt: seconds since the call before (for parts that swing after a movement); 0 for a still pose
  apply(a, dt = 0) {
    const A = (k) => a[k] || 0, au = {}, w = this._w, head = this.head, act = this.d.acting || {};
    for (const k of AUS) au[k] = clamp(A(k), 0, 1.3);
    for (const k in w) w[k] = 0;
    const bA = clamp(A('browAsym'), -1, 1), mA = clamp(A('mouthAsym'), -1, 1), eyeS = clamp(A('eyeScale'), -1, 1), gx = clamp(A('gazeX'), -1, 1), gy = clamp(A('gazeY'), -1, 1);
    const blink = clamp(au.AU45), droop = clamp(au.AU43), wide = au.AU5, squint = au.AU7, cheek = au.AU6, smile = au.AU12;
    const jaw = clamp(au.AU26 + au.AU16 * 0.2 + au.AU25 * 0.15, 0, 0.95) * (act.open ?? 1);
    const round = clamp(au.AU18 + au.AU22), shut = clamp(au.AU24 + au.AU28);
    [['_L', 1], ['_R', -1]].forEach(([sf, s], side) => {
      // brows: one side can go up while the other presses down
      const u = clamp(bA * s, -1, 1), uu = Math.max(0, u), ud = Math.max(0, -u);
      const a1 = au.AU1 * (1 - 0.25 * uu), a2 = clamp(au.AU2 + uu * 0.55, 0, 1.3), a4 = clamp(au.AU4 + ud * 0.45, 0, 1.3);
      w['browInnerUp' + sf] = a1; w['browOuterUp' + sf] = a2; w['browDown' + sf] = a4;
      const bm = this._brow[side], bk = act.brow ?? 1;
      bm[0] = bk * (-0.3 * a4 + 0.08 * a1); bm[1] = bk * (0.75 * a1 - 0.5 * a4 + 0.25 * a2 + 0.12 * wide - 0.15 * au.AU9);
      bm[2] = 0; bm[3] = bk * (0.32 * a1 + 0.6 * a2 - 0.3 * a4 + 0.12 * wide);
      bm[4] = 0; bm[5] = bk * (0.5 * a2 - 0.18 * a1 * (1 - a2) - 0.04 * a4 + 0.05 * wide);
      head.browMove(side, bm);
      // lids
      const open = (1 - blink) * (1 - 0.7 * droop);
      w['eyeWide' + sf] = clamp((wide + 0.3 * uu + 0.45 * Math.max(0, eyeS)) * open, 0, 1.3);
      w['eyeSquint' + sf] = clamp(0.9 * squint + 0.3 * cheek + 0.22 * a4 * (1 - wide) + 0.2 * ud + 0.35 * Math.max(0, -eyeS), 0, 1.2) * (1 - 0.6 * blink);
      w['eyeBlink' + sf] = clamp(blink + (1 - blink) * (0.55 * droop + 0.1 * a4 * (1 - wide)));
      w['eyeLookUp' + sf] = Math.max(0, gy) * 0.8 * (1 - blink); w['eyeLookDown' + sf] = Math.max(0, -gy) * 0.9 * (1 - blink);
      w['eyeLookOut' + sf] = Math.max(0, gx * s) * 0.7; w['eyeLookIn' + sf] = Math.max(0, -gx * s) * 0.7;
      // cheeks and nose
      w['cheekSquint' + sf] = clamp(0.8 * cheek + 0.25 * smile); w['cheekRaiser' + sf] = clamp(0.6 * cheek + 0.3 * smile);
      w['noseSneer' + sf] = clamp(au.AU9 + 0.25 * au.AU10);
      // mouth: a smirk pulls one corner up and out
      const m = mA * s, mu = Math.max(0, m);
      w['mouthSmile' + sf] = clamp(smile * (1 + 0.3 * m) + 0.35 * mu * (1 - 0.5 * round), 0, 1.3);
      w['mouthFrown' + sf] = clamp(au.AU15 * (1 - 0.25 * m), 0, 1.2);
      w['mouthStretch' + sf] = clamp(au.AU20 * 0.8 * (1 + 0.2 * m));
      w['mouthDimple' + sf] = clamp(au.AU14 + 0.15 * mu);
      w['mouthUpperUp' + sf] = clamp(au.AU10 * (1 + 0.5 * m) + 0.2 * au.AU9 + 0.12 * smile * jaw);
      w['mouthLowerDown' + sf] = clamp(au.AU16 * 0.9 + 0.1 * jaw * (1 - round));
      w['mouthPress' + sf] = clamp(au.AU24);
    });
    w.jawOpen = jaw; w.mouthFunnel = clamp(au.AU22); w.mouthPucker = clamp(au.AU18);
    w.mouthRollLower = clamp(au.AU28 * 0.9); w.mouthRollUpper = clamp(au.AU28 * 0.7);
    w.mouthShrugLower = clamp(au.AU17 * 0.9); w.mouthShrugUpper = clamp(au.AU17 * 0.35);
    w.mouthClose = clamp(jaw * shut); w.mouthLeft = Math.max(0, mA) * 0.35 * (1 - 0.5 * round); w.mouthRight = Math.max(0, -mA) * 0.35 * (1 - 0.5 * round);
    head.set(w);
    head.look(gx * 20.6, gy * 17.2, -1, 1 + 0.2 * clamp(A('pupil'), -1, 1));

    // ---- torso: lean, twist and tilt about the waist ----
    const M = this._m, P = this.pivot, per = 1 / UNIT; // channel sizes were stage units: per centimetre here
    const lean = A('torsoLean') * 1.5, by = A('torsoY') * 0.0125 * per;
    M.e.set(lean * D2R, (A('torsoYaw') + A('headYaw') * 0.12) * D2R, -A('torsoRoll') * D2R, 'YXZ'); M.q.setFromEuler(M.e);
    M.v.set(A('torsoShift') * 0.0125 * per, by, 0);
    const k = by * UNIT; this._pivoted(this.gWaist.matrix, P.waist, M.q, M.v, M.one.set(1 - k * 0.25, 1 + k * 0.5, 1 - k * 0.25)); M.one.set(1, 1, 1);
    // shoulders (negative torsoShrug = raised, as the tracker measures it)
    const shrug = clamp(-A('torsoShrug'), -5, 14) * 0.0125 * per * (act.shrug ?? 1.3);
    this._shrug = shrug; this._setShrug(shrug);
    // ---- head: the neck joint swings the head in an arc, the skull adds its own tilt. A leaning torso carries the
    // head with it: the head counters most of that so the face stays towards the viewer ----
    const hp = (A('headPitch') - 0.6 * lean) * D2R, hy = A('headYaw') * D2R, hr = A('headRoll') * D2R;
    M.e.set(hp * 0.55, hy * 0.5, hr * 0.55, 'YXZ'); M.q.setFromEuler(M.e);
    M.v.set(A('headX') * 0.02 * per, A('headY') * 0.02 * per + shrug * 0.25, A('headZ') * 0.024 * per);
    this._pivoted(M.n, P.neck, M.q, M.v, M.one); this.gNeck.matrix.copy(M.n);
    // squash and stretch keep the volume and hold the chin where it is
    const q = clamp(A('squash'), -1, 1) * 0.13, sy = 1 + q, sxz = 1 / Math.sqrt(sy);
    M.e.set(hp * 0.45, hy * 0.5, hr * 0.45, 'YXZ'); M.q.setFromEuler(M.e);
    this._pivoted(M.s, P.skull, M.q, M.v.set(0, 0, 0), M.one);
    if (q) { M.t.makeTranslation(P.chin.x, P.chin.y, P.chin.z); M.s.multiply(M.t); M.t.makeScale(sxz, sy, sxz); M.s.multiply(M.t); M.t.makeTranslation(-P.chin.x, -P.chin.y, -P.chin.z); M.s.multiply(M.t); }
    this.gSkull.matrix.copy(M.s);
    // for the neck: from the shoulders' and from the neck joint's space into the head's
    M.h.multiplyMatrices(M.n, M.s); M.a.copy(M.h).invert(); M.b.copy(M.s).invert();
    this.gWaist.matrixWorldNeedsUpdate = true;
    head.dirty = true; head.update();
  }
  // matrix of a joint: turn by q (and scale by s) about the point c, then move by t
  _pivoted(out, c, q, t, s) { const M = this._m; out.compose(M.v2 || (M.v2 = new THREE.Vector3()).set(0, 0, 0), q, s); const e = out.elements, x = c.x, y = c.y, z = c.z;
    e[12] = c.x + t.x - (e[0] * x + e[4] * y + e[8] * z); e[13] = c.y + t.y - (e[1] * x + e[5] * y + e[9] * z); e[14] = c.z + t.z - (e[2] * x + e[6] * y + e[10] * z); return out; }
}
