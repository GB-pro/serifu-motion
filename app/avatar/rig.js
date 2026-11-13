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
    this._build();
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

  _mat(color, opts = {}) {
    return new THREE.MeshPhysicalMaterial({ color, roughness: opts.roughness ?? 0.55, metalness: 0, sheen: opts.sheen ?? 0.3, sheenColor: new THREE.Color(opts.sheenColor ?? '#ffd9c7'), clearcoat: opts.clearcoat ?? 0, clearcoatRoughness: 0.4, ...opts.extra });
  }

  _build() {
    const d = this.d;
    const skin = this._mat(d.skin, { roughness: 0.62, sheen: 0.5 });
    this.skinMat = skin;
    // ---- body ----
    this.torso = new THREE.Group(); this.root.add(this.torso);
    const bodyH = d.body.height, sw = d.body.shoulder;
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
    if (d.body.collar) {
      const col = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.06, 12, 32), this._mat(d.body.collar, { roughness: 0.7 }));
      col.rotation.x = Math.PI / 2; col.scale.set(1.2, 1, 1); col.position.y = -0.02; this.torso.add(col);
    }
    // arms
    this.arms = [];
    for (const s of [-1, 1]) {
      const arm = new THREE.Group();
      arm.position.set(s * sw * 0.92, -0.12, 0);
      const up = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, bodyH * 0.55, 6, 16), this._mat(d.body.sleeve || d.body.color, { roughness: 0.8, sheen: 0.8 }));
      up.position.y = -bodyH * 0.35; arm.add(up);
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 14), skin); hand.position.y = -bodyH * 0.7; hand.scale.set(1, 1.2, 0.9); arm.add(hand);
      arm.rotation.z = s * 0.12;
      this.torso.add(arm); this.arms.push(arm);
    }
    // neck
    this.neck = new THREE.Group(); this.neck.position.y = 0.0; this.torso.add(this.neck);
    const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.23, 0.4, 24), skin);
    neckMesh.position.y = 0.1; this.neck.add(neckMesh);
    // ---- head ----
    this.head = new THREE.Group();
    this.head.position.y = 0.12 + d.head.sy * d.head.scale * 0.9;
    this.neck.add(this.head);
    this.headPivotOffset = d.head.sy * d.head.scale * 0.98;
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
    const white = new THREE.MeshPhysicalMaterial({ color: '#fbfaf7', roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.05 });
    const tex = irisTexture(e.color, e.pupil ?? 0.42);
    const irisMat = new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.03 });
    const lidMat = this._mat(new THREE.Color(d.skin).multiplyScalar(0.94).getStyle(), { roughness: 0.6, sheen: 0.5, extra: { side: THREE.DoubleSide } });
    const lashMat = new THREE.MeshStandardMaterial({ color: e.lash || '#1b1210', roughness: 0.6 });
    for (const s of [-1, 1]) {
      const g = new THREE.Group();
      const r = e.r;
      const cx = s * e.sep, cy = e.y;
      const cz = this.surfZ(cx, cy) - r * e.depth;
      g.position.set(cx, cy, cz);
      g.rotation.y = s * (e.splay ?? 0.12);
      this.face.add(g);
      const ball = new THREE.Group(); g.add(ball);
      ball.add(new THREE.Mesh(new THREE.SphereGeometry(r, 40, 28), white));
      const capA = e.iris * 1.18;
      const irisGeo = new THREE.SphereGeometry(r * 1.004, 40, 12, 0, Math.PI * 2, 0, capA);
      irisGeo.rotateX(Math.PI / 2);
      planarUV(irisGeo, r * Math.sin(capA));
      const iris = new THREE.Mesh(irisGeo, irisMat); ball.add(iris);
      // glints (fixed to head, not rotating with eyeball)
      const glMat = new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.95 });
      const gl1 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.13, 12, 8), glMat); gl1.position.set(-r * 0.32 * s * 0 - r * 0.28, r * 0.33, r * 0.93); gl1.scale.z = 0.3; g.add(gl1);
      const gl2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.06, 10, 8), glMat); gl2.position.set(r * 0.25, -r * 0.22, r * 0.96); gl2.scale.z = 0.3; g.add(gl2);
      // lids: hemispherical shells rotating about the eye centre
      const upGeo = new THREE.SphereGeometry(r * 1.03, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2);
      const up = new THREE.Mesh(upGeo, lidMat); const upP = new THREE.Group(); upP.add(up); g.add(upP);
      const lash = new THREE.Mesh(new THREE.TorusGeometry(r * 1.035, r * (e.lashW ?? 0.07), 8, 40, Math.PI), lashMat);
      lash.rotation.x = Math.PI / 2; lash.scale.set(1, 1, 0.6); upP.add(lash);
      if (e.lashFlick) { const fl = new THREE.Mesh(new THREE.ConeGeometry(r * 0.08, r * 0.35, 8), lashMat); fl.position.set(s * r * 0.98, r * 0.05, r * 0.15); fl.rotation.z = -s * 1.1; upP.add(fl); }
      const loGeo = new THREE.SphereGeometry(r * 1.025, 40, 20, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
      const lo = new THREE.Mesh(loGeo, lidMat); const loP = new THREE.Group(); loP.add(lo); g.add(loP);
      this.eyes.push({ g, ball, upP, loP, s });
    }
  }

  _browCurve(s, a) {
    const b = this.d.brows;
    const pts = [];
    const xi = b.inner, xo = b.outer;
    const n = 7;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      let x = lerp(xi, xo, t);
      let y = b.y + Math.sin(t * Math.PI) * b.arch - t * b.tilt;
      // AU1 inner raise (+ slight oblique "sad" shape), AU2 outer raise, AU4 lower + knit
      const inner = (1 - t) * (1 - t), outer = t * t, mid = Math.sin(t * Math.PI);
      y += a.AU1 * 0.16 * (inner * 1.0 + mid * 0.35);
      y += a.AU2 * 0.15 * (outer * 1.0 + mid * 0.45);
      y -= a.AU4 * (0.10 * inner + 0.06 * mid + 0.02 * outer);
      x -= a.AU4 * 0.05 * inner;
      y -= a.AU9 * 0.04 * inner;
      y += (a.AU5 || 0) * 0.02;
      pts.push(this.onFace(s * x, y, 0.025));
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
      const curve = this._browCurve(s, a);
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
      const pa = this.onFace(A[i][0], A[i][1], off), pb = this.onFace(B[i][0], B[i][1], off);
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
    const jaw = clamp(a.AU26 * 1.0 + a.AU16 * 0.25 + (a.AU25 || 0) * 0.15, 0, 1.2);
    const smile = clamp(a.AU12, 0, 1.2), frown = clamp(a.AU15, 0, 1), stretch = clamp(a.AU20, 0, 1);
    const pucker = clamp(a.AU18, 0, 1), funnel = clamp(a.AU22, 0, 1), press = clamp(a.AU24, 0, 1), roll = clamp(a.AU28, 0, 1);
    const upUp = clamp(a.AU10, 0, 1), loDown = clamp(a.AU16, 0, 1), chin = clamp(a.AU17, 0, 1), dimple = clamp(a.AU14, 0, 1);
    const round = clamp(pucker + funnel, 0, 1);
    // corner positions
    const hw = W * (1 + 0.32 * stretch + 0.22 * smile + 0.1 * dimple - 0.42 * round - 0.12 * press - 0.08 * jaw);
    const cornerY = cy + 0.1 * smile - 0.075 * frown + 0.02 * dimple - 0.012 * stretch;
    // vertical opening of the inner contour
    const open = 0.34 * jaw * m.open + 0.035 * funnel + 0.03 * upUp + 0.02 * loDown;
    const upIn = cy + 0.25 * open + 0.03 * upUp - 0.012 * press + 0.01 * smile;
    const loIn = cy - 0.75 * open - 0.035 * loDown * (jaw > 0.05 ? 1 : 0.4) + 0.025 * chin + 0.01 * press - 0.012 * smile * (jaw > 0.1 ? 1 : 0);
    const n = 25;
    const upper = [], lower = [], upOut = [], loOut = [];
    const lipT = m.lipT * (1 - 0.6 * roll) * (1 + 0.5 * round) * (1 - 0.3 * press);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1) * 2 - 1; // -1..1
      const x = t * hw;
      const shape = 1 - Math.pow(Math.abs(t), round > 0.3 ? 2.0 : 2.6); // rounder when puckered
      const smileCurve = (smile * 0.06 - frown * 0.05) * (1 - shape); // corners up
      const yU = lerp(cornerY, upIn, shape) + smileCurve * 0.2;
      const yL = lerp(cornerY, loIn, shape) - 0 * smileCurve;
      // closed mouth: slight smile curve on the seam
      const seam = cy + (smile * 0.05 - frown * 0.045) * Math.pow(Math.abs(t), 2) + (0.008 * press);
      const closedness = clamp(1 - (upIn - loIn) / 0.02, 0, 1);
      const uy = lerp(yU, seam + 0.0015, closedness), ly = lerp(yL, seam - 0.0015, closedness);
      upper.push([x, uy]); lower.push([x, ly]);
      const bow = m.bow * Math.exp(-((Math.abs(t) - 0.18) ** 2) / 0.02) * (1 - round * 0.5);
      upOut.push([x * 1.01, uy + lipT * 0.55 * shape + bow * shape + 0.004]);
      loOut.push([x * 1.01, ly - lipT * 0.85 * Math.pow(shape, 0.8) - 0.004]);
    }
    this._strip(this.mouthInner, upper, lower, 0.006);
    this._strip(this.upperLip, upOut, upper, 0.009);
    this._strip(this.lowerLip, lower, loOut, 0.009);
    // lip line (thin dark line along the seam / inner contour)
    const lw = 0.0055 + 0.004 * (1 - clamp(open * 8));
    this._strip(this.lipLine, upper.map(([x, y]) => [x, y + lw * 0.5]), upper.map(([x, y]) => [x, y - lw * 0.5]), 0.011);
    // teeth & tongue (clipped by stencil to the mouth opening)
    const teethW = hw * 0.85;
    const tt = [], tb = [], bt = [], bb = [], to1 = [], to2 = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1) * 2 - 1, x = t * teethW;
      const yTop = upIn + 0.02 - 0.02 * t * t;
      tt.push([x, yTop + 0.03]); tb.push([x, yTop - 0.045]);
      const yBot = loIn + 0.012 - 0.0 * t * t;
      bt.push([x, yBot + 0.03 - 0.01 * jaw]); bb.push([x, yBot - 0.05]);
      to1.push([x * 0.8, loIn + 0.045 + 0.015 * (1 - t * t)]); to2.push([x * 0.8, loIn - 0.05]);
    }
    this._strip(this.teethTop, tt, tb, 0.008);
    this._strip(this.teethBot, bt, bb, 0.008);
    this._strip(this.tongue, to1, to2, 0.0075);
    this.teethBot.visible = jaw > 0.15;
  }

  // jaw/chin deformation of the head mesh (lower face follows AU26)
  _updateJaw(a) {
    const jaw = clamp(a.AU26 * 1.0 + a.AU16 * 0.2, 0, 1.2) * this.d.mouth.open;
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

  // ---------- apply one emo frame ----------
  // a: {AU1..AU45, gazeX, gazeY, headPitch, headYaw, headRoll, headZ, torsoRoll, torsoShift, torsoShrug, torsoLean}
  apply(a) {
    const A = (k) => (a[k] || 0);
    const au = {};
    for (const k of ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU20', 'AU22', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43', 'AU45']) au[k] = clamp(A(k), 0, 1.3);
    // brows
    this._updateBrows(au);
    // eyes / lids
    const blink = clamp(Math.max(au.AU45, au.AU43 || 0));
    const wide = au.AU5, squint = au.AU7, cheek = au.AU6;
    for (const e of this.eyes) {
      const gx = clamp(A('gazeX'), -1, 1), gy = clamp(A('gazeY'), -1, 1);
      e.ball.rotation.y = gx * 0.45 * e.s * 0 + gx * 0.45;
      e.ball.rotation.x = -gy * 0.35;
      const open = -0.72 - 0.3 * wide + 0.35 * squint + 0.1 * au.AU4 + 0.18 * cheek - 0.12 * gy;
      const upAng = lerp(open, Math.PI / 2 + 0.05, blink);
      e.upP.rotation.x = upAng;
      const lo = 0.55 - 0.42 * squint - 0.38 * cheek - 0.08 * wide + 0.1 * gy;
      e.loP.rotation.x = lerp(lo, -0.15, blink);
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
    // head & torso (degrees -> radians); head pivots at neck
    const D = Math.PI / 180;
    this.head.rotation.set(-A('headPitch') * D, A('headYaw') * D, -A('headRoll') * D, 'YXZ');
    this.head.position.z = A('headZ') * 0.02;
    this.torso.rotation.set(A('torsoLean') * D * 0.6, A('headYaw') * D * 0.12, -A('torsoRoll') * D, 'YXZ');
    this.torso.position.x = A('torsoShift') * 0.01;
    this.arms.forEach((arm, k) => { arm.position.y = -0.12 + A('torsoShrug') * 0.004 * 0; arm.rotation.x = A('torsoLean') * D * -0.3; });
    this.torso.position.y = clamp(A('torsoShrug'), -10, 10) * 0.002;
  }
}
