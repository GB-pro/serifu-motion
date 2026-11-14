// Serifu Motion — the cast: original characters built on the sculpted head (head.js) and body (figure.js).
// They are not based on any existing character. Lengths are centimetres on the published head (hair, brows) or on
// the figure (clothes).
import * as THREE from 'three';
import { loadHeadBase, Field } from './head.js';
import { Figure, cloth, hairMaterial, ring } from './figure.js';

// what all of them share: a round head with the face in low relief, large eyes, a small nose and mouth close under them
function form(o = {}) { // the smooth form of the head, in centimetres on the published head (its middle is at 0, 2.5, 1.5)
  const f = new Field(), w = o.wide ?? 1;
  f.ell([0, 4.5, 0.5], [9.0 * w, 9.5 * (o.tall ?? 1), 10.0]);                                                          // skull
  f.ell([0, -1.6 - (o.drop ?? 0), 2.2], [8.0 * w * (o.jaw ?? 1), 5.2 * (o.long ?? 1), 8.3], { k: o.blend ?? 3.5 }); // cheeks, jaw and chin as one round box: the face is flat in front from the brow to the mouth
  return f;
}
const SHAPE = { smooth: 20, noseSmooth: 60, nostril: 150, noseFilm: 400, neckSmooth: 60, neckCalm: 80, reliefSmooth: 300, even: 100, relax: 30, calm: 30, relief: 0.6,
  eye: 1.7, eyeTall: 1.3, eyeApart: 0.2, eyeDown: 0.4, feature: 1, featureUp: 0.5,
  mouth: 1.25, mouthCurve: 0.5, mouthFlat: 0, mouthRelief: 0.8, noseForm: { r: [1.6, 1.2, 1.5], out: 1.45, k: 0.7, bridge: { r: [0.8, 0.7], k: 1.2 } }, neck: 0.5, neckTop: -5.5, neckFade: 3, ear: 0.9 };
const POSE = { mouthSmile_L: 0.25, mouthSmile_R: 0.25 };
// the two wings of a bow for a Field: centre c, each `long`, half as high as tall[0] at the knot and tall[1] at the end, `thick` through; `back`: how far the ends lean back per cm
const wings = (c, long, tall, thick, round, back = 0) => (x, y, z) => { const u = Math.abs(x - c[0]), t = Math.min(1, u / long), qx = u - long + round, qy = Math.abs(y - c[1]) - (tall[0] + (tall[1] - tall[0]) * Math.pow(t, 0.75)) + round, qz = Math.abs(z - c[2] + back * u) - thick + round; return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(0, Math.max(qx, qy, qz)) - round; };
const matte = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.45, ...o });

export const DESIGNS = {
  hana: {
    name: 'Hana', label: { ja: 'ハナ', en: 'Hana' },
    head: { shape: { ...SHAPE, form: form() }, pose: POSE, skin: '#f0b995', blush: '#f07f6e', blushAmount: 0.32, lip: '#d0706a', iris: '#3f8a58', irisSize: 0.7, hair: '#5a2f1a', lash: { thick: 0.085, wing: 0.42 }, brow: { in: [1.9, 4.72], top: [3.5, 5.1], out: [4.75, 4.65], thick: [0.55, 0.5, 0.16] } },
    body: { shoulder: 7.6, depth: 4.4, color: '#f2c14e' },
    hair: (head) => { // a round bob that ends at the chin and covers the ears, the fringe swept to one side in a few thick locks
      const F = head.shape.form, f = new Field(), m = hairMaterial('#6b3a1f'), S = (y, p, o) => head.on(y, p, o);
      f.fn((x, y, z) => F.value(x, y, z) - 1.2);                        // the hair lies on the whole head, 1.2 cm thick
      f.ell([0, 6.2, -2.4], [10.3, 9.7, 10.6], { k: 2 });               // fuller on top and at the back
      for (const s of [-1, 1]) f.ell([s * 8.7, 0.4, 0.6], [2.9, 6.8, 5.8], { k: 1.6 }); // over the ears
      f.ell([0, 0.8, 11.5], [7.3, 7.8, 7.5], { mode: 'cut', k: 1.2 });  // the window of the face
      f.half([0, -5.4, 0], [0, -1, 0], { mode: 'keep', k: 1.8 });       // it ends at the chin, rounded
      f.ell([0, -8.4, 0.5], [7.4, 5.6, 8.2], { mode: 'cut', k: 1.6 });  // hollow underneath, around the neck
      head.addSculpt(f, m, 0.35, true, [[-14, -9, -15.5], [14, 18.5, 14]]);
      // the fringe: broad locks that run from the top of the head over the hair, are combed across the forehead towards the clip and end in points above the brows
      const fringe = [[-40, 58, -37, 40, -30, 26.5, 1.5], [-28, 60, -24, 40, -15.5, 23, 1.7], [-17, 62, -10, 40, -2, 21, 1.8], [-6, 63, 4, 41, 11.5, 21.5, 1.8], [5, 62, 17, 42, 24.5, 23.5, 1.7], [16, 60, 29, 42, 36, 27, 1.5]]
        .map(([y0, p0, y1, p1, y2, p2, wide]) => ({ path: [[y0, 100], [y0, p0], [y1, p1], [y2, p2]], wide, high: [0.68, 0.3], lift: [-1, -0.45, 0.25], hold: [0.3, 0.74], root: 0.8 }));
      // the sides and the back: long locks from the top down to the ends of the bob, which turn in towards the face
      const sides = []; for (const s of [-1, 1]) for (const [y0, y1, y2, y3, wide] of [[34, 50, 63, 52, 2.1], [62, 76, 86, 74, 2.3], [96, 104, 112, 100, 2.3], [130, 134, 139, 130, 2.4], [162, 164, 166, 160, 2.4]]) sides.push({ path: [[s * y0, 100], [s * y1, 55], [s * y2, 8], [s * y3, -36]], wide, high: [0.7, 0.45], lift: [-1, -0.42, -0.5], hold: [0.25, 0.8], root: 0.8, taper: 6 });
      head.addLocks([...fringe, ...sides], m, { rings: 32, over: f });
      const c = new Field(); c.ell(S(41, 27, 1.55), [1.3, 0.42, 0.5], { rot: [0, 0.7, 0.45] }); head.addSculpt(c, matte('#ff5d7a', { roughness: 0.3 }), 0.12, true); // hair clip
    },
    wear: (fig) => {
      const ny = fig.body.neckY, c = new Field(); c.fn(ring([0, ny + 0.2, -0.9], [3.9, 4.1], 0.62, 0.12, 1.0));
      fig.addBody(c, cloth('#ffffff'), 0.3, [[-6, ny - 2.5, -7], [6, ny + 2.5, 5]]);
      const by = ny - 1.9, bz = fig.front(0, by) + 0.45, bow = new Field(); // a bow under the collar and two buttons
      bow.fn(wings([0, by, bz], 2.7, [0.42, 1.25], 0.42, 0.3, 0.16)); bow.ell([0, by, bz + 0.12], [0.5, 0.62, 0.56], { k: 0.2 });
      fig.addBody(bow, cloth('#ff6f91', { sheen: 0.5, roughness: 0.6 }), 0.11, [[-3.4, by - 1.8, bz - 1.2], [3.4, by + 1.8, bz + 1.3]]);
      for (const y of [ny - 4.6, ny - 7.2]) { const g = new THREE.SphereGeometry(0.42, 18, 12); g.scale(1, 1, 0.45); fig.addProp(new THREE.Mesh(g, matte('#fff6e6', { roughness: 0.35 })), [0, y, fig.front(0, y) + 0.04]); }
    },
  },
};

export async function loadCast() { await loadHeadBase(); }
export function createFigure(key) { return new Figure(DESIGNS[key] || DESIGNS.hana); }
