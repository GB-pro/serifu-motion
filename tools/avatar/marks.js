// Landmarks of a built figure, in centimetres of the head at rest: evaluated in figure_dev.html (tools/avatar/evalpage.mjs).
(() => {
  const h = window.fig.head, b = h.base, P = h.rest, S = b.pos, n = b.n, R = b.meta.regions, rg = b.rings;
  const box = (pick) => { const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9]; let c = 0; for (let i = 0; i < n; i++) if (pick(i)) { c++; for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i * 3 + k]); hi[k] = Math.max(hi[k], P[i * 3 + k]); } } return { n: c, lo: lo.map((v) => +v.toFixed(2)), hi: hi.map((v) => +v.toFixed(2)) }; };
  const skin = (i) => R[b.region[i]] === 'face' || R[b.region[i]] === 'head';
  let tip = -1; for (let i = 0; i < n; i++) if (skin(i) && Math.abs(S[i * 3]) < 0.3 && Math.abs(S[i * 3 + 1]) < 1.2 && (tip < 0 || S[i * 3 + 2] > S[tip * 3 + 2])) tip = i;
  const at = (i) => [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]].map((v) => +v.toFixed(2));
  return {
    eyeL: box((i) => skin(i) && rg[i * 3 + 1] === 0), mouth: box((i) => skin(i) && rg[i * 3] === 0), noseTip: at(tip), noseTipSrc: [S[tip * 3], S[tip * 3 + 1], S[tip * 3 + 2]].map((v) => +v.toFixed(2)),
    chin: box((i) => skin(i) && Math.abs(S[i * 3]) < 1.5 && S[i * 3 + 2] > 6 && S[i * 3 + 1] < -6.5 && S[i * 3 + 1] > -8.6),
    head: box((i) => skin(i) && S[i * 3 + 1] + 4.5 + 0.41 * (S[i * 3 + 2] + 5.7) > 0 && !(Math.abs(S[i * 3]) > 7.1 && S[i * 3 + 1] > -2.6 && S[i * 3 + 1] < 5 && S[i * 3 + 2] < 4.6)),
    upperTeeth: box((i) => R[b.region[i]] === 'teeth' && S[i * 3 + 1] > -3.4), lowerTeeth: box((i) => R[b.region[i]] === 'teeth' && S[i * 3 + 1] < -3.6),
    eyeball: (() => { const p = h.eyes[0].mesh.geometry.attributes.position; const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9]; for (let i = 0; i < p.count; i++) { const v = [p.getX(i), p.getY(i), p.getZ(i)]; for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], v[k]); hi[k] = Math.max(hi[k], v[k]); } } return { lo: lo.map((v) => +v.toFixed(2)), hi: hi.map((v) => +v.toFixed(2)) }; })(),
  };
})()
