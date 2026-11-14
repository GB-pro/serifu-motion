// Serifu Motion — soft shadows that are worked out once, when a character is built.
//
// Where a surface can see little of the surroundings (the corner of an eye, under the hair, under the chin, the top of
// an eyeball under its lid) it gets less of the light that comes from all around. That share is measured for every
// vertex by looking at the character from many directions, and stored on the vertex ("ao", 1 = open, 0 = closed in).
import * as THREE from 'three';

const depthMat = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  uniforms: { near: { value: 0 }, far: { value: 1 } },
  vertexShader: 'uniform float near; uniform float far; varying float vD; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vD = (-mv.z - near) / (far - near); gl_Position = projectionMatrix * mv; }',
  fragmentShader: 'varying float vD; void main() { float d = clamp(vD, 0.0, 0.9999) * 255.0; gl_FragColor = vec4(floor(d) / 255.0, fract(d), 0.0, 1.0); }',
});

// top: the object that holds everything that can throw a shadow. meshes: the meshes whose vertices are measured.
// opts.combine 'max': keep the larger of the old and the new value (a second pose in which a hidden part is in the open).
export function bakeOcclusion(renderer, top, meshes, { size = 512, dirs = 64, offset = 0.3, bias = 0.25, combine = null } = {}) {
  top.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(top), sph = box.getBoundingSphere(new THREE.Sphere()), C = sph.center, R = sph.radius * 1.02;
  const rt = new THREE.WebGLRenderTarget(size, size, { depthBuffer: true }), buf = new Uint8Array(size * size * 4);
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, R, 3 * R), scene = new THREE.Scene(), parent = top.parent;
  depthMat.uniforms.near.value = R; depthMat.uniforms.far.value = 3 * R; scene.overrideMaterial = depthMat; scene.add(top);
  // points and normals of the vertices in the space of `top`'s world
  const T = meshes.map((m) => {
    const g = m.geometry, p = g.attributes.position, nr = g.attributes.normal, n = p.count, P = new Float32Array(n * 3), N = new Float32Array(n * 3), v = new THREE.Vector3(), nm = new THREE.Matrix3().getNormalMatrix(m.matrixWorld);
    for (let i = 0; i < n; i++) { v.fromBufferAttribute(p, i).applyMatrix4(m.matrixWorld); P[i * 3] = v.x; P[i * 3 + 1] = v.y; P[i * 3 + 2] = v.z; v.fromBufferAttribute(nr, i).applyMatrix3(nm).normalize(); N[i * 3] = v.x; N[i * 3 + 1] = v.y; N[i * 3 + 2] = v.z; }
    return { g, n, P, N, vis: new Float32Array(n), tot: new Float32Array(n) };
  });
  const saved = { target: renderer.getRenderTarget(), color: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha(), shadow: renderer.shadowMap.enabled, tone: renderer.toneMapping };
  renderer.shadowMap.enabled = false; renderer.toneMapping = THREE.NoToneMapping; renderer.setRenderTarget(rt); renderer.setClearColor(0xffff00, 1);
  const rx = new THREE.Vector3(), ry = new THREE.Vector3(), rz = new THREE.Vector3(), scale = offset, ga = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < dirs; k++) {
    const y = 1 - (2 * (k + 0.5)) / dirs, r = Math.sqrt(1 - y * y), dx = Math.cos(ga * k) * r, dz = Math.sin(ga * k) * r;
    cam.position.set(C.x + dx * 2 * R, C.y + y * 2 * R, C.z + dz * 2 * R); cam.up.set(0, Math.abs(y) > 0.95 ? 0 : 1, Math.abs(y) > 0.95 ? 1 : 0); cam.lookAt(C); cam.updateMatrixWorld(true);
    renderer.clear(); renderer.render(scene, cam); renderer.readRenderTargetPixels(rt, 0, 0, size, size, buf);
    cam.matrixWorld.extractBasis(rx, ry, rz); // rz points from the character to the camera
    for (const t of T) {
      const { P, N, vis, tot, n } = t;
      for (let i = 0; i < n; i++) {
        const o = i * 3, nd = N[o] * rz.x + N[o + 1] * rz.y + N[o + 2] * rz.z; if (nd < 0.08) continue;
        const qx = P[o] + N[o] * scale - C.x, qy = P[o + 1] + N[o + 1] * scale - C.y, qz = P[o + 2] + N[o + 2] * scale - C.z;
        const u = Math.floor(((qx * rx.x + qy * rx.y + qz * rx.z) / R * 0.5 + 0.5) * size), v = Math.floor(((qx * ry.x + qy * ry.y + qz * ry.z) / R * 0.5 + 0.5) * size);
        tot[i] += nd; if (u < 0 || v < 0 || u >= size || v >= size) { vis[i] += nd; continue; }
        const b = (v * size + u) * 4, seen = ((buf[b] + buf[b + 1] / 255) / 255) * 2 * R, mine = R - (qx * rz.x + qy * rz.y + qz * rz.z);
        if (mine <= seen + bias) vis[i] += nd;
      }
    }
  }
  scene.remove(top); if (parent) parent.add(top);
  renderer.setRenderTarget(saved.target); renderer.setClearColor(saved.color, saved.alpha); renderer.shadowMap.enabled = saved.shadow; renderer.toneMapping = saved.tone; rt.dispose();
  for (const t of T) {
    let a = t.g.attributes.ao; const fresh = !a; if (fresh) { a = new THREE.BufferAttribute(new Float32Array(t.n), 1); t.g.setAttribute('ao', a); }
    for (let i = 0; i < t.n; i++) { const v = t.tot[i] > 0 ? t.vis[i] / t.tot[i] : 1; a.array[i] = combine === 'max' && !fresh ? Math.max(a.array[i], v) : v; }
    a.needsUpdate = true;
  }
}

// evens out the measured values over the mesh (the measurement is grainy): each vertex takes part of its neighbours' mean
export function smoothOcclusion(geometry, passes = 2) {
  const a = geometry.attributes.ao, p = geometry.attributes.position, idx = geometry.index.array, n = p.count, A = a.array;
  const key = new Map(), first = new Int32Array(n);
  for (let i = 0; i < n; i++) { const k = `${Math.round(p.getX(i) * 500)},${Math.round(p.getY(i) * 500)},${Math.round(p.getZ(i) * 500)}`, f = key.get(k); if (f === undefined) { key.set(k, i); first[i] = i; } else first[i] = f; }
  const sum = new Float32Array(n), cnt = new Float32Array(n);
  for (let s = 0; s < passes; s++) {
    for (let i = 0; i < n; i++) if (first[i] !== i) A[first[i]] = Math.min(A[first[i]], A[i]);
    sum.fill(0); cnt.fill(0);
    for (let t = 0; t < idx.length; t += 3) { const i0 = first[idx[t]], i1 = first[idx[t + 1]], i2 = first[idx[t + 2]]; sum[i0] += A[i1] + A[i2]; cnt[i0] += 2; sum[i1] += A[i0] + A[i2]; cnt[i1] += 2; sum[i2] += A[i0] + A[i1]; cnt[i2] += 2; }
    for (let i = 0; i < n; i++) if (first[i] === i && cnt[i]) A[i] = 0.5 * A[i] + 0.5 * (sum[i] / cnt[i]);
    for (let i = 0; i < n; i++) if (first[i] !== i) A[i] = A[first[i]];
  }
  a.needsUpdate = true;
}

// Makes a material use the "ao" of the vertices: less light from all around, a little less from the lamps, and, on
// skin, a warm tone in the closed-in places (light that has passed through skin is red).
export function occluded(material, tint = '#40404a', direct = 0.45) {
  const c = new THREE.Color(tint);
  material.onBeforeCompile = (sh) => {
    sh.uniforms.aoTint = { value: c }; sh.uniforms.aoDirect = { value: direct };
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float ao;\nvarying float vAo;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvAo = ao;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vAo;\nuniform vec3 aoTint;\nuniform float aoDirect;')
      .replace('#include <aomap_fragment>', `#include <aomap_fragment>
      {
        float o = clamp(vAo, 0.0, 1.0); vec3 oc = mix(aoTint, vec3(1.0), o);
        reflectedLight.indirectDiffuse *= oc; reflectedLight.indirectSpecular *= o;
        reflectedLight.directDiffuse *= mix(vec3(1.0), oc, aoDirect); reflectedLight.directSpecular *= mix(1.0, o, aoDirect);
        #ifdef USE_CLEARCOAT
          clearcoatSpecularIndirect *= o;
        #endif
        #ifdef USE_SHEEN
          sheenSpecularIndirect *= o;
        #endif
      }`);
  };
  material.customProgramCacheKey = () => 'ao' + tint + direct;
  return material;
}
