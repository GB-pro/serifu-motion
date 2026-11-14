// Serifu Motion — sculpting a shape from soft primitives.
//
// Hair, clothes and a torso are described as a handful of rounded primitives (ellipsoids, round cones, half spaces) that
// are merged, cut and intersected with soft edges, the way a sculptor adds and removes clay. The description is turned
// into a mesh when a character is built (surface nets on a regular grid). Lengths are in the units of whoever calls.

// soft minimum and maximum of two distances: k is the width of the rounded edge between the two shapes
const smin = (a, b, k) => { if (k <= 0) return Math.min(a, b); const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * 0.25; };
const smax = (a, b, k) => -smin(-a, -b, k);

function rotation(e) { // rows of the matrix that turns a point into the primitive's own axes (Euler angles x, y, z in radians)
  if (!e) return null;
  const [a, b, c] = e, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
  // R = Rz(c) Ry(b) Rx(a); own axes = R^T (p - centre)
  const R = [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa, sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa, -sb, cb * sa, cb * ca];
  return [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
}

export class Field {
  constructor() { this.ops = []; }
  // mode: 'add' (default), 'cut' (remove), 'keep' (only what is also inside this stays); k: softness of the edge
  _push(op, o = {}) { op.mode = o.mode || 'add'; op.k = o.k ?? 0; this.ops.push(op); return this; }
  // ellipsoid: centre, radii, turned by rot
  ell(c, r, o = {}) { return this._push({ t: 0, c, r, R: rotation(o.rot), bs: Math.max(r[0], r[1], r[2]) }, o); }
  // round cone: from point a with radius ra to point b with radius rb
  cone(a, b, ra, rb, o = {}) { return this._push({ t: 1, a, b, ra, rb, c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], bs: Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / 2 + Math.max(ra, rb) }, o); }
  // half space: the side of the plane through point p that the normal n points away from
  half(p, n, o = {}) { const l = Math.hypot(n[0], n[1], n[2]); return this._push({ t: 2, n: [n[0] / l, n[1] / l, n[2] / l], d: (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) / l }, o); }
  // any function of place that is negative inside
  fn(f, o = {}) { return this._push({ t: 3, f }, o); }
  value(x, y, z) {
    let d = 1e9;
    for (const op of this.ops) {
      let v;
      if (op.t === 0) {
        let px = x - op.c[0], py = y - op.c[1], pz = z - op.c[2];
        if (op.mode === 'add' && d < 1e8) { const far = Math.hypot(px, py, pz) - op.bs; if (far > d + op.k) continue; }
        if (op.R) { const R = op.R, qx = R[0] * px + R[1] * py + R[2] * pz, qy = R[3] * px + R[4] * py + R[5] * pz, qz = R[6] * px + R[7] * py + R[8] * pz; px = qx; py = qy; pz = qz; }
        const r = op.r, ax = px / r[0], ay = py / r[1], az = pz / r[2], k0 = Math.hypot(ax, ay, az), k1 = Math.hypot(ax / r[0], ay / r[1], az / r[2]);
        v = k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(r[0], r[1], r[2]);
      } else if (op.t === 1) {
        const a = op.a, b = op.b, bax = b[0] - a[0], bay = b[1] - a[1], baz = b[2] - a[2], pax = x - a[0], pay = y - a[1], paz = z - a[2];
        const h = Math.min(1, Math.max(0, (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz || 1)));
        v = Math.hypot(pax - bax * h, pay - bay * h, paz - baz * h) - (op.ra + (op.rb - op.ra) * h);
      } else if (op.t === 2) v = op.n[0] * x + op.n[1] * y + op.n[2] * z - op.d;
      else v = op.f(x, y, z);
      d = op.mode === 'add' ? smin(d, v, op.k) : op.mode === 'cut' ? smax(d, -v, op.k) : smax(d, v, op.k);
    }
    return d;
  }
  // box around everything that was added
  bounds(margin = 0.5) {
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (const op of this.ops) { if (op.mode !== 'add' || op.t > 1) continue; for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], op.c[i] - op.bs - op.k - margin); hi[i] = Math.max(hi[i], op.c[i] + op.bs + op.k + margin); } }
    return [lo, hi];
  }
}

// Mesh of the surface where the field is 0. cell: size of a grid cell. Returns { pos, nrm, idx } (smooth normals).
export function polygonise(field, cell, box = field.bounds()) {
  const [lo, hi] = box, nx = Math.ceil((hi[0] - lo[0]) / cell) + 1, ny = Math.ceil((hi[1] - lo[1]) / cell) + 1, nz = Math.ceil((hi[2] - lo[2]) / cell) + 1;
  const V = new Float32Array(nx * ny * nz);
  for (let k = 0, o = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, o++) V[o] = field.value(lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell);
  const sx = 1, sy = nx, sz = nx * ny, cellV = new Int32Array(nx * ny * nz).fill(-1), pos = [];
  const corner = [0, sx, sy, sx + sy, sz, sx + sz, sy + sz, sx + sy + sz], cx = [0, 1, 0, 1, 0, 1, 0, 1], cy = [0, 0, 1, 1, 0, 0, 1, 1], cz = [0, 0, 0, 0, 1, 1, 1, 1];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const o = i + j * sy + k * sz; let mask = 0;
    for (let c = 0; c < 8; c++) if (V[o + corner[c]] < 0) mask |= 1 << c;
    if (mask === 0 || mask === 255) continue;
    let px = 0, py = 0, pz = 0, m = 0;
    for (const [a, b] of edges) {
      const va = V[o + corner[a]], vb = V[o + corner[b]]; if (va < 0 === vb < 0) continue;
      const t = va / (va - vb); px += cx[a] + (cx[b] - cx[a]) * t; py += cy[a] + (cy[b] - cy[a]) * t; pz += cz[a] + (cz[b] - cz[a]) * t; m++;
    }
    cellV[o] = pos.length / 3; pos.push(lo[0] + (i + px / m) * cell, lo[1] + (j + py / m) * cell, lo[2] + (k + pz / m) * cell);
  }
  const idx = [];
  const quad = (a, b, c, d, flip) => { if (a < 0 || b < 0 || c < 0 || d < 0) return; if (flip) idx.push(a, c, b, a, d, c); else idx.push(a, b, c, a, c, d); };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const o = i + j * sy + k * sz, v = V[o] < 0;
    if (v !== V[o + sx] < 0) quad(cellV[o - sy - sz], cellV[o - sz], cellV[o], cellV[o - sy], !v); // edge along x: cells around it in the y-z plane
    if (v !== V[o + sy] < 0) quad(cellV[o - sx - sz], cellV[o - sx], cellV[o], cellV[o - sz], !v);
    if (v !== V[o + sz] < 0) quad(cellV[o - sx - sy], cellV[o - sy], cellV[o], cellV[o - sx], !v);
  }
  const P = Float32Array.from(pos), N = new Float32Array(P.length), e = cell * 0.5;
  for (let i = 0; i < P.length; i += 3) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    const gx = field.value(x + e, y, z) - field.value(x - e, y, z), gy = field.value(x, y + e, z) - field.value(x, y - e, z), gz = field.value(x, y, z + e) - field.value(x, y, z - e), l = Math.hypot(gx, gy, gz) || 1;
    N[i] = gx / l; N[i + 1] = gy / l; N[i + 2] = gz / l;
  }
  return { pos: P, nrm: N, idx: P.length / 3 > 65535 ? Uint32Array.from(idx) : Uint16Array.from(idx) };
}
