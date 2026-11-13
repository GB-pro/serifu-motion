// Serifu Motion — procedural stylized 3D character rig driven by FACS Action Units.
// Every facial feature is parametric, so AU values map directly onto geometry (no blendshape names).
import * as THREE from 'three';

const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
let STENCIL_REF = 1;

function canvasTex(w, h, draw) {
  const c = typeof document !== 'undefined' ? document.createElement('canvas') : null;
  c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

function irisTexture(color, pupil = 0.42) {
  return canvasTex(256, 256, (g, w) => {
    const c = new THREE.Color(color);
    const dark = c.clone().multiplyScalar(0.35), light = c.clone().lerp(new THREE.Color('#ffffff'), 0.35);
    const r = w / 2;
    g.fillStyle = '#f7f4ef'; g.fillRect(0, 0, w, w);
    let grd = g.createRadialGradient(r, r, r * 0.1, r, r, r * 0.98);
    grd.addColorStop(0, '#' + light.getHexString()); grd.addColorStop(0.55, '#' + c.getHexString()); grd.addColorStop(0.92, '#' + dark.getHexString()); grd.addColorStop(1, '#101010');
    g.fillStyle = grd; g.beginPath(); g.arc(r, r, r * 0.98, 0, Math.PI * 2); g.fill();
    // fibres
    g.globalAlpha = 0.18; g.strokeStyle = '#' + light.getHexString(); g.lineWidth = 2;
    for (let i = 0; i < 70; i++) { const a = i / 70 * Math.PI * 2; g.beginPath(); g.moveTo(r + Math.cos(a) * r * pupil, r + Math.sin(a) * r * pupil); g.lineTo(r + Math.cos(a) * r * 0.9, r + Math.sin(a) * r * 0.9); g.stroke(); }
    g.globalAlpha = 1;
    grd = g.createRadialGradient(r, r, 0, r, r, r * pupil);
    grd.addColorStop(0, '#000'); grd.addColorStop(0.85, '#050505'); grd.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grd; g.beginPath(); g.arc(r, r, r * pupil, 0, Math.PI * 2); g.fill();
  });
}

// planar UVs for a cap that faces +z
function planarUV(geo, rad) {
  const p = geo.attributes.position, uv = geo.attributes.uv;
  for (let i = 0; i < p.count; i++) uv.setXY(i, 0.5 + p.getX(i) / (2 * rad), 0.5 + p.getY(i) / (2 * rad));
  uv.needsUpdate = true;
}

export class Character {
  constructor(design) {
    this.d = design;
    this.root = new THREE.Group();
    this.root.name = design.name;
    this.stencil = STENCIL_REF++;
    this.state = {};
    this.dyn = [];            // secondary-motion parts (hair, accessories): spring-driven follow-through
    this._build();
  }

  // Register a part that should lag behind and settle after the body moves (follow-through / overlap).
  // obj must already be attached; pivot is the attach point in the parent's space.
  // opts: stiff (spring), damp, gain (reaction to acceleration), grav (0..1 how much it hangs with gravity), max (rad)
  addDyn(obj, pivot, opts = {}) {
    const parent = obj.parent, g = new THREE.Group();
    g.position.set(pivot[0], pivot[1], pivot[2]); parent.add(g);
    obj.position.sub(g.position); g.add(obj);
    const dir = obj.position.y >= 0 ? 1 : -1; // part sits above (tuft, bun) or below (lock, string) its attach point
    this.dyn.push({ g, dir, ax: 0, az: 0, vx: 0, vz: 0, prev: null, vel: new THREE.Vector3(), stiff: opts.stiff ?? 90, damp: opts.damp ?? 7, gain: opts.gain ?? 1, grav: opts.grav ?? 0, max: opts.max ?? 0.5 });
    return g;
  }

  // ---------- head surface (analytic ellipsoid with jaw taper) ----------
  taper(y) { const d = this.d.head; const t = y < 0 ? Math.pow(clamp(-y / d.sy, 0, 1), 1.6) : 0; return 1 - d.jaw * t; }
  surfZ(x, y) {
    const d = this.d.head; const t = this.taper(y);
    const x0 = x / (d.sx * t), y0 = y / d.sy;
    return d.sz * Math.sqrt(Math.max(0.0001, 1 - x0 * x0 - y0 * y0));
  }
  surfNormal(x, y) {
    const d = this.d.head, z = this.surfZ(x, y), t = this.taper(y);
    return new THREE.Vector3(x / (d.sx * d.sx * t * t), y / (d.sy * d.sy), z / (d.sz * d.sz)).normalize();
  }
  onFace(x, y, off = 0.004) { const n = this.surfNormal(x, y); return new THREE.Vector3(x, y, this.surfZ(x, y)).addScaledVector(n, off); }
  // The lower face of the head mesh slides down and back with the jaw, and the cheeks push forward in a smile
  // (_updateJaw). The mouth is drawn on that deformed surface: on the analytic one a wide-open mouth sinks into the chin.
  _jawW(y, z) { const my = this.d.mouth.y; return y < my + 0.02 && z > -0.2 ? clamp((my + 0.02 - y) / 0.25) * clamp((z + 0.2) / 0.6) : 0; }
  onMouth(x, y, off) {
    const jd = this._jawAmt || 0, my = this.d.mouth.y;
    let y0 = y; // the point of the undeformed surface that the jaw carries to height y
    if (jd > 0) for (let i = 0; i < 8; i++) y0 = y + this._jawW(y0, this.surfZ(x, y0)) * jd * 0.13;
    const p = this.onFace(x, y0, off), w = jd > 0 ? this._jawW(y0, this.surfZ(x, y0)) : 0;
    p.y -= w * jd * 0.13; p.z -= w * jd * 0.03;
    const push = (this._smileAmt || 0) * 0.02 + (this._cheekAmt || 0) * 0.015;
    if (push > 0) for (const s of [-1, 1]) { const dx = x - s * 0.38, dy = y0 - (my + 0.12); p.z += Math.exp(-(dx * dx + dy * dy) / 0.02) * push; }
    return p;
  }

  _mat(color, opts = {}) {
    return new THREE.MeshPhysicalMaterial({ color, roughness: opts.roughness ?? 0.55, metalness: 0, sheen: opts.sheen ?? 0.3, sheenColor: new THREE.Color(opts.sheenColor ?? '#ffd9c7'), clearcoat: opts.clearcoat ?? 0, clearcoatRoughness: 0.4, ...opts.extra });
  }

  _build() {
    const d = this.d;
    const skin = this._mat(d.skin, { roughness: 0.62, sheen: 0.5 });
    this.skinMat = skin;
    // ---- body ---- (waist is the pivot for lean / twist / roll; `torso` keeps its origin at the neck base)
    const bodyH = d.body.height, sw = d.body.shoulder;
    this.waistY = -bodyH * 0.72;
    this.waist = new THREE.Group(); this.waist.position.y = this.waistY; this.root.add(this.waist);
    this.torso = new THREE.Group(); this.torso.position.y = -this.waistY; this.waist.add(this.torso);
    const prof = [];
    const N = 24;
    for (let i = 0; i <= N; i++) {
      const t = i / N; // 0 bottom -> 1 top
      let r = lerp(d.body.waist, sw, Math.pow(t, 0.6));
      if (t > 0.86) r = sw * Math.cos((t - 0.86) / 0.14 * Math.PI / 2 * 0.92) + 0.001;
      prof.push(new THREE.Vector2(Math.max(0.001, r), t * bodyH));
    }
    const shirtGeo = new THREE.LatheGeometry(prof, 48);
    shirtGeo.scale(1, 1, 0.62);
    const shirt = new THREE.Mesh(shirtGeo, this._mat(d.body.color, { roughness: 0.8, sheen: 0.8, sheenColor: '#ffffff' }));
    shirt.position.y = -bodyH;
    this.torso.add(shirt);
    this.shirt = shirt; this.shirtBase = Float32Array.from(shirtGeo.attributes.position.array); this._shrugKey = null;
    this.collar = null;
    if (d.body.collar) {
      const col = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.06, 12, 32), this._mat(d.body.collar, { roughness: 0.7 }));
      col.rotation.x = Math.PI / 2; col.scale.set(1.2, 1, 1); col.position.y = -0.02; this.torso.add(col); this.collar = col;
    }
    // arms hang from the shoulders (they rise with a shrug and stay vertical when the torso leans)
    this.arms = [];
    for (const s of [-1, 1]) {
      const arm = new THREE.Group();
      arm.position.set(s * sw * 0.92, -0.12, 0);
      const up = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, bodyH * 0.55, 6, 16), this._mat(d.body.sleeve || d.body.color, { roughness: 0.8, sheen: 0.8 }));
      up.position.y = -bodyH * 0.35; arm.add(up);
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 14), skin); hand.position.y = -bodyH * 0.7; hand.scale.set(1, 1.2, 0.9); arm.add(hand);
      arm.rotation.z = s * 0.12;
      this.torso.add(arm); this.arms.push({ g: arm, s, sw: 0, sv: 0, fw: 0, fv: 0 });
    }
    // front details: they make leans, twists and tilts of the torso readable (a plain lathe body hides them)
    this.shirtR = (y) => { // radius of the shirt at torso-local height y (0 = neck base, negative = down)
      const t = clamp((y + bodyH) / bodyH), i = Math.min(N - 1, Math.floor(t * N)), f = t * N - i;
      return lerp(prof[i].x, prof[i + 1].x, f);
    };
    this.onShirt = (x, y, off = 0.006) => { const r = this.shirtR(y); return new THREE.Vector3(x, y, 0.62 * Math.sqrt(Math.max(0, r * r - x * x)) + off); };
    for (const f of d.body.front || []) {
      const m = this._mat(f.color, { roughness: f.rough ?? 0.6, sheen: 0.4 });
      if (f.type === 'buttons') {
        for (let i = 0; i < (f.n || 3); i++) { const b = new THREE.Mesh(new THREE.SphereGeometry(f.r || 0.034, 14, 10), m); b.position.copy(this.onShirt(0, (f.y ?? -0.34) - i * (f.dy ?? 0.2), 0.004)); b.scale.z = 0.45; this.torso.add(b); }
      } else if (f.type === 'bow') {
        const y = f.y ?? -0.13, g = new THREE.Group(); g.position.copy(this.onShirt(0, y, 0.03));
        for (const sd of [-1, 1]) { const w = new THREE.Mesh(new THREE.SphereGeometry(0.1, 18, 12), m); w.scale.set(1.25, 0.8, 0.4); w.position.set(sd * 0.115, 0, 0); w.rotation.z = sd * 0.25; g.add(w);
          const tail = new THREE.Mesh(new THREE.SphereGeometry(0.06, 14, 10), m); tail.scale.set(0.8, 1.7, 0.35); tail.position.set(sd * 0.06, -0.12, -0.01); tail.rotation.z = sd * 0.3; g.add(tail); }
        const k = new THREE.Mesh(new THREE.SphereGeometry(0.05, 14, 10), m); k.scale.set(1, 1, 0.7); g.add(k);
        this.torso.add(g);
      } else if (f.type === 'tie') {
        const g = new THREE.Group(), y = f.y ?? -0.1; g.position.copy(this.onShirt(0, y, 0.012));
        const knot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 14, 10), m); knot.scale.set(1.1, 0.85, 0.5); g.add(knot);
        const L = f.len ?? 0.55, blade = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.095, L, 4, 1), m); blade.rotation.y = Math.PI / 4; blade.scale.z = 0.3;
        const pB = this.onShirt(0, y - L / 2 - 0.04, 0.014); blade.position.set(0, -L / 2 - 0.04, pB.z - g.position.z); blade.rotation.x = Math.atan2(this.onShirt(0, y - L, 0).z - this.onShirt(0, y, 0).z, L) * -1; g.add(blade);
        this.torso.add(g);
      } else if (f.type === 'pocket') {
        const y = f.y ?? -0.82, w = f.w ?? 0.5, h = f.h ?? 0.24, n = 14, pos = [], idx = [];
        for (let i = 0; i <= n; i++) { const u = i / n * 2 - 1, x = u * w / 2, top = h / 2 * (1 - 0.35 * u * u), a1 = this.onShirt(x, y + top, 0.008), a2 = this.onShirt(x, y - h / 2, 0.008); pos.push(a1.x, a1.y, a1.z, a2.x, a2.y, a2.z); }
        for (let i = 0; i < n; i++) { const o = i * 2; idx.push(o, o + 1, o + 2, o + 2, o + 1, o + 3); }
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
        const pk = new THREE.Mesh(g, m); m.side = THREE.DoubleSide; this.torso.add(pk);
      } else if (f.type === 'emblem') {
        const y = f.y ?? -0.42, r = f.r ?? 0.13, c = this.onShirt(f.x || 0, y, 0.008);
        const e1 = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 14), m); e1.position.copy(c); e1.scale.z = 0.12; this.torso.add(e1);
        if (f.inner) { const e2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.55, 20, 12), this._mat(f.inner, { roughness: 0.5 })); e2.position.copy(c); e2.position.z += 0.012; e2.scale.z = 0.12; this.torso.add(e2); }
      }
    }
    // neck
    this.neck = new THREE.Group(); this.neck.position.y = 0.0; this.torso.add(this.neck);
    const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.23, 0.4, 24), skin);
    neckMesh.position.y = 0.1; this.neck.add(neckMesh);
    // ---- head ---- (headPivot = base of the skull: nods swing the whole head in an arc; `head` adds the skull's own tilt)
    this.headPivot = new THREE.Group(); this.headPivot.position.y = 0.06; this.neck.add(this.headPivot);
    this.head = new THREE.Group();
    this.headUp = 0.12 + d.head.sy * d.head.scale * 0.9 - 0.06;
    this.head.position.y = this.headUp;
    this.headPivot.add(this.head);
    this.face = new THREE.Group(); this.face.scale.setScalar(d.head.scale); this.head.add(this.face);
    const hg = new THREE.SphereGeometry(1, 128, 96);
    const p = hg.attributes.position;
    this.headBase = new Float32Array(p.array.length);
    for (let i = 0; i < p.count; i++) {
      let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      x *= d.head.sx; y *= d.head.sy; z *= d.head.sz;
      x *= this.taper(y);
      // cheek fullness
      for (const s of [-1, 1]) {
        const dx = x - s * 0.5, dy = y + 0.28, dz = z - 0.6;
        const w = Math.exp(-(dx * dx + dy * dy + dz * dz) / 0.06);
        x += s * w * d.head.cheek * 0.5; z += w * d.head.cheek * 0.4;
      }
      // chin
      { const dy = y + d.head.sy * 0.86, dz = z - 0.45; const w = Math.exp(-(x * x / 0.04 + dy * dy / 0.03 + dz * dz / 0.08)); z += w * d.head.chin; y -= w * d.head.chin * 0.5; }
      p.setXYZ(i, x, y, z);
    }
    this.headBase.set(p.array);
    hg.computeVertexNormals();
    this.headMesh = new THREE.Mesh(hg, skin);
    this.face.add(this.headMesh);
    // ears
    for (const s of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 16), skin);
      ear.scale.set(0.45, 1, 0.75); ear.position.set(s * d.head.sx * 0.97, -0.06, -0.05); ear.rotation.y = s * 0.3;
      if (d.ears !== false) this.face.add(ear);
    }
    this._buildEyes();
    this._buildBrows();
    this._buildNose();
    this._buildMouth();
    this._buildCheeks();
    if (d.hair) d.hair(this, THREE);
    if (d.extras) d.extras(this, THREE);
    this.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  }

  _buildEyes() {
    const d = this.d, e = d.eyes;
    this.eyes = [];
    this._m3 = new THREE.Matrix4(); this._m3b = new THREE.Matrix4();
    const white = new THREE.MeshPhysicalMaterial({ color: '#fbfaf7', roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.05 });
    const tex = irisTexture(e.color, e.pupil ?? 0.42);
    const irisMat = new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.03 });
    const lidMat = this._mat(new THREE.Color(d.skin).multiplyScalar(0.975).getStyle(), { roughness: 0.62, sheen: 0.5, extra: { side: THREE.DoubleSide } });
    const lashMat = new THREE.MeshStandardMaterial({ color: e.lash || '#1b1210', roughness: 0.6 });
    // The design gives the visible eye size (r, depth); the eyeball itself is a larger, deeper-set sphere so the lids
    // read as part of the face instead of a protruding ball.
    const depth = e.set ?? 0.74;
    const apR = e.r * Math.sqrt(1 - e.depth * e.depth) * (e.grow ?? 1.08); // radius of the visible opening
    const r = apR / Math.sqrt(1 - depth * depth);
    const irisR = e.r * Math.sin(e.iris * 1.18) * (e.grow ?? 1.08) * 0.98;
    const capA = Math.asin(Math.min(0.95, irisR / r));
    this.eyeAp = Math.acos(depth); // half-angle of the visible opening as seen from the eye centre
    this.eyeGeom = { cx: e.sep, cy: e.y, cz: this.surfZ(e.sep, e.y) - r * depth, R: r * 1.03, oval: e.oval ?? 1.1 };
    this.eyeIrisA = capA;
    for (const s of [-1, 1]) {
      const g = new THREE.Group();
      const cx = s * e.sep, cy = e.y;
      const cz = this.surfZ(cx, cy) - r * depth;
      g.position.set(cx, cy, cz);
      g.rotation.y = s * (e.splay ?? 0.12);
      const sq = new THREE.Group(); sq.scale.set(1, e.oval ?? 1.1, 1); g.add(sq); // taller cartoon eye
      this.face.add(g);
      const ball = new THREE.Group(); sq.add(ball);
      ball.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), white));
      const irisGeo = new THREE.SphereGeometry(r * 1.003, 40, 12, 0, Math.PI * 2, 0, capA);
      irisGeo.rotateX(Math.PI / 2);
      planarUV(irisGeo, r * Math.sin(capA));
      const iris = new THREE.Mesh(irisGeo, irisMat); ball.add(iris);
      // glints (fixed to the head, they do not rotate with the eyeball)
      const glMat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.95 });
      const gl1 = new THREE.Mesh(new THREE.SphereGeometry(irisR * 0.24, 12, 8), glMat); gl1.position.set(-irisR * 0.42, irisR * 0.5, r * 0.985); gl1.scale.z = 0.25; sq.add(gl1);
      const gl2 = new THREE.Mesh(new THREE.SphereGeometry(irisR * 0.11, 10, 8), glMat); gl2.position.set(irisR * 0.4, -irisR * 0.36, r * 0.99); gl2.scale.z = 0.25; sq.add(gl2);
      // lids: hemispherical shells rotating about the eye centre
      const upGeo = new THREE.SphereGeometry(r * 1.02, 48, 20, 0, Math.PI * 2, 0, Math.PI / 2);
      const up = new THREE.Mesh(upGeo, lidMat); const upP = new THREE.Group(); upP.add(up); sq.add(upP);
      const lash = new THREE.Mesh(new THREE.TorusGeometry(r * 1.024, apR * (e.lashW ?? 0.07) * 1.15, 8, 48, Math.PI), lashMat);
      lash.rotation.x = Math.PI / 2; lash.scale.set(1, 1, 0.6); upP.add(lash);
      if (e.lashFlick) { const fl = new THREE.Mesh(new THREE.ConeGeometry(apR * 0.09, apR * 0.42, 8), lashMat); fl.position.set(s * apR * 0.98, apR * 0.02, r * depth + apR * 0.05); fl.rotation.z = -s * 1.15; upP.add(fl); }
      const loGeo = new THREE.SphereGeometry(r * 1.016, 48, 20, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
      const lo = new THREE.Mesh(loGeo, lidMat); const loP = new THREE.Group(); loP.add(lo); sq.add(loP);
      const lids = [[up, upP], [lo, loP]].map(([mesh, pivot]) => ({ mesh, pivot, base: Float32Array.from(mesh.geometry.attributes.normal.array), key: null }));
      this.eyes.push({ g, sq, ball, upP, loP, lids, s, iris, glints: [gl1, gl2], splay: (e.splay ?? 0.12), r, capA, irisKey: 100 });
    }
  }

  // brow position: on the face, but always in front of the lid when a lowered brow overlaps the eye
  onBrow(x, y) {
    const p = this.onFace(x, y, 0.025), g = this.eyeGeom, R = g.R * (this._eyeS || 1);
    for (const s of [-1, 1]) {
      const dx = x - s * g.cx, dy = (y - g.cy) / g.oval, d2 = dx * dx + dy * dy;
      if (d2 < R * R) { const z = g.cz + Math.sqrt(R * R - d2) + 0.022; if (z > p.z) p.z = z; }
    }
    return p;
  }
  // a: per-side values {AU1, AU2, AU4, AU5, AU9, lift}
  _browCurve(s, a) {
    const b = this.d.brows;
    const pts = [];
    const xi = b.inner, xo = b.outer;
    const n = 7;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      let x = lerp(xi, xo, t);
      let y = b.y + Math.sin(t * Math.PI) * b.arch - t * b.tilt;
      // AU1 inner raise (oblique "sad" shape), AU2 outer raise (arched), AU4 lower + knit (steep "angry" V)
      const inner = (1 - t) * (1 - t), outer = t * t, mid = Math.sin(t * Math.PI);
      y += a.AU1 * (0.19 * inner + 0.06 * mid - 0.035 * outer * (1 - a.AU2));
      y += a.AU2 * (0.17 * outer + 0.10 * mid + 0.04 * inner);
      y -= a.AU4 * (0.135 * inner + 0.07 * mid - 0.03 * outer);
      x -= a.AU4 * 0.06 * inner;
      y -= a.AU9 * 0.04 * inner;
      y += a.AU5 * 0.025 + a.lift;
      pts.push(this.onBrow(s * x, y));
    }
    return new THREE.CatmullRomCurve3(pts);
  }
  _buildBrows() {
    const b = this.d.brows;
    this.browMat = new THREE.MeshStandardMaterial({ color: b.color, roughness: 0.8 });
    this.brows = [-1, 1].map(() => { const m = new THREE.Mesh(new THREE.BufferGeometry(), this.browMat); this.face.add(m); return m; });
  }
  _updateBrows(a) {
    const b = this.d.brows;
    this.brows.forEach((m, k) => {
      const s = k ? 1 : -1;
      // unilateral brow action (FACS L/R): one brow up, the other pressed down
      const up = clamp(a.browAsym * s, -1, 1);
      const side = { AU1: a.AU1 * (1 - 0.25 * Math.max(0, up)), AU2: clamp(a.AU2 + Math.max(0, up) * 0.55, 0, 1.3), AU4: clamp(a.AU4 + Math.max(0, -up) * 0.45, 0, 1.3), AU5: a.AU5, AU9: a.AU9, lift: up * 0.045 };
      const curve = this._browCurve(s, side);
      // tapered ribbon-tube
      const segs = 24, radial = 8, pos = [], idx = [];
      for (let i = 0; i <= segs; i++) {
        const t = i / segs, c = curve.getPointAt(t), tan = curve.getTangentAt(t);
        const n = this.surfNormal(c.x, c.y);
        const side = new THREE.Vector3().crossVectors(n, tan).normalize();
        const w = b.thick * (0.65 + 0.6 * Math.sin(Math.min(1, t * 1.25 + 0.15) * Math.PI) * (1 - 0.5 * t));
        for (let j = 0; j < radial; j++) {
          const ang = j / radial * Math.PI * 2;
          const v = c.clone().addScaledVector(side, Math.cos(ang) * w).addScaledVector(n, Math.sin(ang) * w * 0.45);
          pos.push(v.x, v.y, v.z);
        }
      }
      for (let i = 0; i < segs; i++) for (let j = 0; j < radial; j++) {
        const a0 = i * radial + j, a1 = i * radial + (j + 1) % radial, b0 = a0 + radial, b1 = a1 + radial;
        idx.push(a0, b0, a1, a1, b0, b1);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
      m.geometry.dispose(); m.geometry = g;
    });
  }

  _buildNose() {
    const n = this.d.nose;
    this.nose = new THREE.Group();
    const z = this.surfZ(0, n.y);
    this.nose.position.set(0, n.y, z - 0.04);
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(n.r, 24, 18), this.skinMat);
    mesh.scale.set(n.sx ?? 1.2, n.sy ?? 0.9, n.sz ?? 1.0);
    this.nose.add(mesh);
    const wr = new THREE.MeshBasicMaterial({ color: '#7a4a3a', transparent: true, opacity: 0 });
    this.noseWrinkle = wr;
    for (let i = 0; i < 2; i++) {
      const l = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.004, 4, 16, Math.PI * 0.8), wr);
      l.position.set(0, n.r * 0.9 + i * 0.03, n.r * 0.55); l.rotation.z = Math.PI * 0.1; this.nose.add(l);
    }
    this.face.add(this.nose);
  }

  _buildCheeks() {
    const c = this.d.cheeks || {};
    this.cheeks = [];
    const blushMat = new THREE.MeshBasicMaterial({ color: c.blush || '#ff8a8a', transparent: true, opacity: c.blushAlpha ?? 0.18, depthWrite: false });
    for (const s of [-1, 1]) {
      const x = s * 0.48, y = -0.22;
      const m = new THREE.Mesh(new THREE.CircleGeometry(0.12, 24), blushMat);
      const p = this.onFace(x, y, 0.012); m.position.copy(p); m.lookAt(p.clone().add(this.surfNormal(x, y)));
      m.scale.set(1.3, 0.8, 1);
      this.face.add(m);
      this.cheeks.push({ m, s, x, y });
    }
  }

  // The lids are shells around the eyeball, so their own normals shade them like a ball set into the face (a dark
  // "bag" under the eye). Shade them with the normal of the face surface at the same spot instead: the lid then reads
  // as skin that continues from the cheek and the brow bone, and only its edge and the lash line draw the eye.
  _lidNormals(e) {
    const oval = this.d.eyes.oval ?? 1.1, W = 0.86, sc = e.g.scale.x;
    const gx = e.g.position.x, gy = e.g.position.y;
    for (const lid of e.lids) {
      const q = lid.pivot.quaternion, key = Math.round(q.x * 500) + ':' + Math.round(q.z * 500) + ':' + Math.round(sc * 200);
      if (key === lid.key) continue; lid.key = key;
      this._m3.makeRotationFromQuaternion(q); const L = this._m3.elements;       // lid rotation (column-major 4x4)
      this._m3b.makeRotationY(e.g.rotation.y); const G = this._m3b.elements;      // socket splay
      const pos = lid.mesh.geometry.attributes.position, nor = lid.mesh.geometry.attributes.normal, b = lid.base;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        // lid local -> eye (scaled) -> face coordinates
        let ax = L[0] * x + L[4] * y + L[8] * z, ay = (L[1] * x + L[5] * y + L[9] * z) * oval, az = L[2] * x + L[6] * y + L[10] * z;
        const fx = gx + sc * (G[0] * ax + G[8] * az), fy = gy + sc * ay;
        const n = this.surfNormal(fx, fy);
        // face normal -> lid local (inverse of the normal transform)
        const hx = G[0] * n.x + G[2] * n.z, hy = n.y * oval, hz = G[8] * n.x + G[10] * n.z;
        let nx = L[0] * hx + L[1] * hy + L[2] * hz, ny = L[4] * hx + L[5] * hy + L[6] * hz, nz = L[8] * hx + L[9] * hy + L[10] * hz;
        const l = Math.hypot(nx, ny, nz) || 1;
        nx = W * nx / l + (1 - W) * b[i * 3]; ny = W * ny / l + (1 - W) * b[i * 3 + 1]; nz = W * nz / l + (1 - W) * b[i * 3 + 2];
        const l2 = Math.hypot(nx, ny, nz) || 1;
        nor.setXYZ(i, nx / l2, ny / l2, nz / l2);
      }
      nor.needsUpdate = true;
    }
  }

  // ---------- mouth ----------
  _buildMouth() {
    const st = this.stencil;
    this.mouthInnerMat = new THREE.MeshBasicMaterial({ color: '#3a1216', side: THREE.DoubleSide, stencilWrite: true, stencilRef: st, stencilFunc: THREE.AlwaysStencilFunc, stencilZPass: THREE.ReplaceStencilOp, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    const masked = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.35, stencilWrite: true, stencilRef: st, stencilFunc: THREE.EqualStencilFunc, depthTest: false });
    this.lipMat = new THREE.MeshPhysicalMaterial({ color: this.d.mouth.lip, roughness: 0.4, clearcoat: 0.4, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3, polygonOffsetUnits: -3 });
    this.lineMat = new THREE.MeshBasicMaterial({ color: '#3b1a18', side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    this.mouthInner = new THREE.Mesh(new THREE.BufferGeometry(), this.mouthInnerMat); this.mouthInner.renderOrder = 10;
    this.teethTop = new THREE.Mesh(new THREE.BufferGeometry(), masked('#fbfbf6')); this.teethTop.renderOrder = 11;
    this.teethBot = new THREE.Mesh(new THREE.BufferGeometry(), masked('#f2f1ea')); this.teethBot.renderOrder = 11;
    this.tongue = new THREE.Mesh(new THREE.BufferGeometry(), masked('#c9545e')); this.tongue.renderOrder = 11;
    this.upperLip = new THREE.Mesh(new THREE.BufferGeometry(), this.lipMat); this.upperLip.renderOrder = 12;
    this.lowerLip = new THREE.Mesh(new THREE.BufferGeometry(), this.lipMat); this.lowerLip.renderOrder = 12;
    this.lipLine = new THREE.Mesh(new THREE.BufferGeometry(), this.lineMat); this.lipLine.renderOrder = 13;
    for (const m of [this.mouthInner, this.teethTop, this.teethBot, this.tongue, this.upperLip, this.lowerLip, this.lipLine]) this.face.add(m);
  }

  // build a surface strip between two 2D polylines (arrays of [x,y]) projected on the face
  _strip(mesh, A, B, off) {
    const pos = [], idx = [], n = A.length;
    for (let i = 0; i < n; i++) {
      const pa = this.onMouth(A[i][0], A[i][1], off), pb = this.onMouth(B[i][0], B[i][1], off);
      pos.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
    }
    for (let i = 0; i < n - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
    mesh.geometry.dispose(); mesh.geometry = g;
  }

  _updateMouth(a) {
    const m = this.d.mouth;
    const cy = m.y, W = m.w / 2;
    const jaw = clamp(a.AU26 * 1.0 + a.AU16 * 0.2 + (a.AU25 || 0) * 0.15, 0, 0.95);
    // the same amounts _updateJaw applies to the head mesh
    this._jawAmt = clamp(a.AU26 * 1.0 + a.AU16 * 0.2, 0, 0.95) * m.open; this._smileAmt = Math.max(0, a.AU12); this._cheekAmt = Math.max(0, a.AU6 || 0);
    const smile = clamp(a.AU12, 0, 1.3), frown = clamp(a.AU15, 0, 1.2), stretch = clamp(a.AU20, 0, 1);
    const pucker = clamp(a.AU18, 0, 1), funnel = clamp(a.AU22, 0, 1), press = clamp(a.AU24, 0, 1), roll = clamp(a.AU28, 0, 1);
    const upUp = clamp(a.AU10, 0, 1), loDown = clamp(a.AU16, 0, 1), chin = clamp(a.AU17, 0, 1), dimple = clamp(a.AU14, 0, 1);
    const round = clamp(pucker + funnel, 0, 1);
    const asym = clamp(a.mouthAsym || 0, -1, 1);
    // corner positions (per side: a smirk pulls one corner up and out, the whole mouth slides toward it)
    const hw0 = W * (1 + 0.32 * stretch + 0.3 * smile + 0.1 * dimple - 0.42 * round - 0.12 * press - 0.08 * jaw);
    const cY0 = cy + 0.125 * smile - 0.085 * frown + 0.02 * dimple - 0.012 * stretch;
    const cx = asym * 0.035 * (1 - 0.5 * round);
    const hwS = [hw0 * (1 - 0.14 * asym), hw0 * (1 + 0.14 * asym)];
    const cYS = [cY0 - 0.055 * asym, cY0 + 0.055 * asym];
    // vertical opening of the inner contour
    const open = 0.34 * jaw * m.open + 0.035 * funnel + 0.03 * upUp + 0.02 * loDown;
    const upIn = cy + 0.25 * open + 0.03 * upUp - 0.012 * press + 0.012 * smile;
    const loIn = cy - 0.75 * open - 0.035 * loDown * (jaw > 0.05 ? 1 : 0.4) + 0.025 * chin + 0.01 * press - 0.012 * smile * (jaw > 0.1 ? 1 : 0);
    const n = 25;
    const upper = [], lower = [], upOut = [], loOut = [];
    const lipT = m.lipT * (1 - 0.6 * roll) * (1 + 0.5 * round) * (1 - 0.3 * press);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1) * 2 - 1; // -1..1
      const sd = t < 0 ? 0 : 1, u = (t + 1) / 2;
      const hw = hwS[sd], cornerY = lerp(cYS[0], cYS[1], u * u * (3 - 2 * u));
      const x = cx + t * hw;
      const shape = 1 - Math.pow(Math.abs(t), round > 0.3 ? 2.0 : 2.6); // rounder when puckered
      const smileCurve = (smile * 0.06 - frown * 0.05) * (1 - shape); // corners up
      const sneer = upUp * 0.03 * asym * t * shape; // one-sided upper lip raise
      const yU = lerp(cornerY, upIn, shape) + smileCurve * 0.2 + sneer;
      const yL = lerp(cornerY, loIn, shape);
      // closed mouth: curved seam
      const seam = lerp(cYS[0], cYS[1], u) - (smile * 0.125 - frown * 0.085) + (smile * 0.125 - frown * 0.085) * Math.pow(Math.abs(t), 2) + (0.008 * press) + sneer * 0.5;
      const closedness = clamp(1 - (upIn - loIn) / 0.02, 0, 1);
      const uy = lerp(yU, seam + 0.0015, closedness), ly = lerp(yL, seam - 0.0015, closedness);
      upper.push([x, uy]); lower.push([x, ly]);
      const bow = m.bow * Math.exp(-((Math.abs(t) - 0.18) ** 2) / 0.02) * (1 - round * 0.5);
      upOut.push([cx + t * hw * 1.01, uy + lipT * 0.55 * shape + bow * shape + 0.004]);
      loOut.push([cx + t * hw * 1.01, ly - lipT * 0.85 * Math.pow(shape, 0.8) - 0.004]);
    }
    this._strip(this.mouthInner, upper, lower, 0.006);
    this._strip(this.upperLip, upOut, upper, 0.009);
    this._strip(this.lowerLip, lower, loOut, 0.009);
    // lip line (thin dark line along the seam / inner contour)
    const lw = 0.0055 + 0.004 * (1 - clamp(open * 8));
    this._strip(this.lipLine, upper.map(([x, y]) => [x, y + lw * 0.5]), upper.map(([x, y]) => [x, y - lw * 0.5]), 0.011);
    // teeth & tongue (clipped by stencil to the mouth opening): clean bands that follow the lip contours
    const tt = [], tb = [], bt = [], bb = [], to1 = [], to2 = [];
    const toothH = 0.05, lowShow = clamp(0.15 + loDown + 0.7 * stretch + 0.5 * frown + 0.4 * upUp);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1) * 2 - 1, taper = 1 - Math.pow(Math.abs(t), 3);
      const xu = upper[i][0], yu = upper[i][1], yl = lower[i][1];
      tt.push([xu, yu + 0.03]); tb.push([xu, yu - toothH * taper]);
      bt.push([xu, yl + toothH * 0.75 * taper * lowShow]); bb.push([xu, yl - 0.03]);
      to1.push([cx + (xu - cx) * 0.8, loIn + (0.05 + 0.02 * (1 - t * t)) * clamp(open * 5)]); to2.push([cx + (xu - cx) * 0.8, loIn - 0.05]);
    }
    this._strip(this.teethTop, tt, tb, 0.008);
    this._strip(this.teethBot, bt, bb, 0.008);
    this._strip(this.tongue, to1, to2, 0.0075);
    this.teethBot.visible = jaw > 0.12 && lowShow > 0.25;
    if (this.mustache) { this.mustache.position.y = (upIn - cy) * 0.6 + 0.02 * upUp; this.mustache.position.x = cx * 0.6; }
  }

  // jaw/chin deformation of the head mesh (lower face follows AU26)
  _updateJaw(a) {
    const jaw = clamp(a.AU26 * 1.0 + a.AU16 * 0.2, 0, 0.95) * this.d.mouth.open;
    const chin = clamp(a.AU17, 0, 1);
    const key = Math.round(jaw * 200) + ':' + Math.round(chin * 100) + ':' + Math.round(a.AU12 * 50);
    if (key === this._jawKey) return; this._jawKey = key;
    const p = this.headMesh.geometry.attributes.position, b = this.headBase;
    const my = this.d.mouth.y;
    for (let i = 0; i < p.count; i++) {
      const x = b[i * 3], y = b[i * 3 + 1], z = b[i * 3 + 2];
      let ny = y, nz = z, nx = x;
      if (y < my + 0.02 && z > -0.2) {
        const w = clamp((my + 0.02 - y) / 0.25) * clamp((z + 0.2) / 0.6);
        ny -= w * jaw * 0.13; nz -= w * jaw * 0.03;
        const cw = Math.exp(-(x * x / 0.03 + (y - (my - 0.32)) ** 2 / 0.02));
        nz += cw * chin * 0.03; ny += cw * chin * 0.015;
      }
      // smile cheek push
      for (const s of [-1, 1]) {
        const dx = x - s * 0.38, dy = y - (my + 0.12);
        const w = Math.exp(-(dx * dx + dy * dy) / 0.02) * (z > 0 ? 1 : 0);
        ny += w * a.AU12 * 0.025; nz += w * (a.AU12 * 0.02 + (a.AU6 || 0) * 0.015);
      }
      p.setXYZ(i, nx, ny, nz);
    }
    p.needsUpdate = true;
    this.headMesh.geometry.computeVertexNormals();
  }

  // shoulders rise with a shrug: the top of the shirt is pushed up near the shoulder tips
  _updateShrug(up) {
    const key = Math.round(up * 400);
    if (key === this._shrugKey) return; this._shrugKey = key;
    const p = this.shirt.geometry.attributes.position, b = this.shirtBase, H = this.d.body.height, sw = this.d.body.shoulder;
    for (let i = 0; i < p.count; i++) {
      const x = b[i * 3], y = b[i * 3 + 1];
      const t = clamp((y / H - 0.5) / 0.42), w = t * t * (3 - 2 * t) * Math.pow(clamp(Math.abs(x) / (sw * 0.75)), 1.3);
      p.setY(i, y + up * w);
    }
    p.needsUpdate = true; this.shirt.geometry.computeVertexNormals();
  }

  // follow-through: each registered part is a damped spring reacting to the acceleration of its attach point
  _updateDyn(dt) {
    if (!this.dyn.length && !this.arms.length) return;
    const reset = !(dt > 0 && dt < 0.2);
    this.root.updateWorldMatrix(true, true);
    const wp = new THREE.Vector3(), acc = new THREE.Vector3(), q = new THREE.Quaternion(), down = new THREE.Vector3();
    for (const d of this.dyn) {
      const par = d.g.parent;
      wp.setFromMatrixPosition(d.g.matrixWorld);
      par.getWorldQuaternion(q); q.invert();
      if (reset || !d.prev) { d.prev = wp.clone(); d.vel.set(0, 0, 0); d.vx = d.vz = 0; if (reset) { d.ax = d.az = 0; } }
      else {
        const vel = wp.clone().sub(d.prev).divideScalar(dt);
        acc.copy(vel).sub(d.vel).divideScalar(dt).clampLength(0, 60);
        d.vel.copy(vel); d.prev.copy(wp);
        acc.applyQuaternion(q); // into the parent's frame
        down.set(0, -1, 0).applyQuaternion(q);
        // hanging parts keep pointing down when the body tilts (grav), all parts lag behind accelerations
        const eqZ = d.grav * Math.atan2(down.x, -down.y) * (d.dir < 0 ? 1 : 0), eqX = -d.grav * Math.atan2(down.z, -down.y) * (d.dir < 0 ? 1 : 0);
        const steps = 2, h = dt / steps;
        for (let k = 0; k < steps; k++) {
          d.vz += (-d.stiff * (d.az - eqZ) - d.damp * d.vz + d.dir * d.gain * acc.x) * h; d.az += d.vz * h;
          d.vx += (-d.stiff * (d.ax - eqX) - d.damp * d.vx - d.dir * d.gain * acc.z) * h; d.ax += d.vx * h;
        }
        d.az = clamp(d.az, -d.max, d.max); d.ax = clamp(d.ax, -d.max, d.max);
      }
      d.g.rotation.set(d.ax, 0, d.az);
    }
  }

  // ---------- apply one emo frame ----------
  // a: {AU1..AU45, gazeX, gazeY, headPitch, headYaw, headRoll, headZ, torsoRoll, torsoShift, torsoShrug, torsoLean,
  //     + acting channels AU43, browAsym, mouthAsym, eyeScale, pupil, squash, headX, headY, torsoYaw, torsoY}
  // dt: seconds since the previous call (drives hair / accessory follow-through); omit or pass 0 for a still pose
  apply(a, dt = 0) {
    const A = (k) => (a[k] || 0);
    const au = {};
    for (const k of ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU20', 'AU22', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43', 'AU45']) au[k] = clamp(A(k), 0, 1.3);
    au.browAsym = clamp(A('browAsym'), -1, 1); au.mouthAsym = clamp(A('mouthAsym'), -1, 1);
    const eyeS = 1 + 0.17 * clamp(A('eyeScale'), -1, 1), irisS = 1 + 0.2 * clamp(A('pupil'), -1, 1);
    this._eyeS = eyeS;
    // brows
    this._updateBrows(au);
    // eyes / lids
    const blink = clamp(au.AU45), droop = clamp(au.AU43);
    const wide = au.AU5, squint = au.AU7, cheek = au.AU6;
    // lid slant: lowered brows cut the eye with an "angry" line, raised inner brows give the "sad" droop at the outer corner
    const slantBase = 0.4 * au.AU4 * (1 - 0.6 * au.AU1) - 0.3 * au.AU1 * (1 - 0.5 * au.AU2) * (1 - 0.7 * au.AU4);
    const gx = clamp(A('gazeX'), -1, 1), gy = clamp(A('gazeY'), -1, 1);
    const ap = this.eyeAp;
    for (const e of this.eyes) {
      // converge the eyes on the viewer (the sockets are splayed outward with the skull)
      e.ball.rotation.y = gx * 0.36 - e.s * e.splay * 0.92;
      e.ball.rotation.x = -gy * 0.3;
      e.g.scale.setScalar(eyeS);
      // iris / pupil size: rebuild the cap on the eyeball (cheap) so it always lies on the sphere
      const ik = Math.round(irisS * 50);
      if (ik !== e.irisKey) {
        e.irisKey = ik; const ca = Math.min(1.2, e.capA * ik / 50);
        const ig = new THREE.SphereGeometry(e.r * 1.003, 40, 12, 0, Math.PI * 2, 0, ca); ig.rotateX(Math.PI / 2); planarUV(ig, e.r * Math.sin(ca));
        e.iris.geometry.dispose(); e.iris.geometry = ig;
      }
      // highlights stay near the top of the iris: they follow most of the eye's rotation (a highlight left behind in
      // the middle of the iris reads as a white pupil)
      const gry = gx * 0.36 * 0.8, grx = -gy * 0.3 * 0.8;
      for (const gl of e.glints) {
        if (!gl.userData.p0) gl.userData.p0 = gl.position.clone();
        const p0 = gl.userData.p0, px = p0.x * irisS, py = p0.y * irisS, pz = p0.z;
        const y1 = py * Math.cos(grx) - pz * Math.sin(grx), z1 = py * Math.sin(grx) + pz * Math.cos(grx);
        gl.position.set(px * Math.cos(gry) + z1 * Math.sin(gry), y1, -px * Math.sin(gry) + z1 * Math.cos(gry));
        gl.scale.set(irisS, irisS, 0.25);
      }
      const up = clamp(au.browAsym * e.s, -1, 1); // the raised-brow side opens wider, the other side narrows
      // elevation of the lid edges as a share of the visible opening (1 = fully retracted)
      const upEl = ap * (0.74 + 0.42 * wide - 0.4 * squint - 0.2 * au.AU4 - 0.08 * cheek + 0.3 * gy + 0.12 * up);
      const upAng = lerp(lerp(-upEl, -0.12 * ap, droop * 0.8), 0.16 * ap, blink);
      e.upP.rotation.set(upAng, 0, e.s * slantBase * (1 - blink), 'ZYX');
      const loEl = -ap * (0.9 - 0.5 * squint - 0.6 * cheek - 0.12 * au.AU12 * (1 - cheek) + 0.1 * wide - 0.25 * gy);
      e.loP.rotation.set(lerp(-loEl, 0.2 * ap, blink), 0, -e.s * 0.1 * cheek * (1 - blink), 'ZYX');
      // highlights disappear behind the lids
      const lidTop = -upAng, lidBot = -lerp(-loEl, 0.2 * ap, blink);
      for (const gl of e.glints) { const el = Math.asin(clamp(gl.position.y / gl.position.z, -1, 1)); gl.visible = el < lidTop - 0.04 && el > lidBot + 0.02; }
      this._lidNormals(e);
    }
    // nose
    const n = this.d.nose;
    this.nose.position.y = n.y + au.AU9 * 0.035 + au.AU10 * 0.01;
    this.nose.scale.set(1 + au.AU9 * 0.12, 1 - au.AU9 * 0.08, 1);
    this.noseWrinkle.opacity = clamp(au.AU9 * 1.3) * 0.7;
    // cheeks rise
    for (const c of this.cheeks) {
      const y = c.y + cheek * 0.06 + au.AU12 * 0.04;
      const p = this.onFace(c.x, y, 0.012); c.m.position.copy(p);
      c.m.material.opacity = (this.d.cheeks?.blushAlpha ?? 0.18) * (1 + 0.8 * cheek);
    }
    this._updateMouth(au);
    this._updateJaw(au);
    // ---- head: the neck pivot swings the head in an arc, the skull adds its own tilt ----
    const D = Math.PI / 180;
    // a leaning torso carries the head with it: the head counters most of that so the face stays toward the viewer
    const leanDeg = A('torsoLean') * 1.5;
    const hp = (A('headPitch') - 0.6 * leanDeg) * D, hy = A('headYaw') * D, hr = A('headRoll') * D;
    this.headPivot.rotation.set(hp * 0.55, hy * 0.5, hr * 0.55, 'YXZ');
    this.head.rotation.set(hp * 0.45, hy * 0.5, hr * 0.45, 'YXZ');
    // shoulders (negative torsoShrug = raised, as measured by the tracker)
    const shrug = clamp(-A('torsoShrug'), -5, 14) * 0.0125;
    this._updateShrug(shrug);
    this.headPivot.position.set(A('headX') * 0.02, 0.06 + A('headY') * 0.02 - shrug * 0.3, A('headZ') * 0.024);
    if (this.collar) this.collar.position.y = -0.02 + shrug * 0.25;
    // squash & stretch (volume preserving, anchored at the chin)
    const q = clamp(A('squash'), -1, 1) * 0.13, sy = 1 + q, sxz = 1 / Math.sqrt(sy), sc = this.d.head.scale;
    this.face.scale.set(sc * sxz, sc * sy, sc * sxz);
    this.face.position.y = (sy - 1) * this.d.head.sy * sc * 0.9;
    // ---- torso: lean / twist / roll about the waist ----
    const lean = A('torsoLean') * D * 1.5, twist = (A('torsoYaw') + A('headYaw') * 0.12) * D, roll = -A('torsoRoll') * D;
    this.waist.rotation.set(lean, twist, roll, 'YXZ');
    const by = A('torsoY') * 0.0125;
    this.waist.position.set(A('torsoShift') * 0.0125, this.waistY + by, 0);
    this.waist.scale.set(1 - by * 0.25, 1 + by * 0.5, 1 - by * 0.25);
    // arms: rise with the shoulders and keep hanging down while the torso leans and rolls
    for (const arm of this.arms) {
      arm.g.position.y = -0.12 + shrug * 0.95;
      arm.g.rotation.set(-lean * 0.85, 0, arm.s * (0.12 + shrug * 0.9) - roll * 0.85);
    }
    this._updateDyn(dt);
  }
}
