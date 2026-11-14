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
const matte = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.45, ...o });

export const DESIGNS = {
  hana: {
    name: 'Hana', label: { ja: 'ハナ', en: 'Hana' },
    head: { shape: { ...SHAPE, form: form() }, pose: POSE, skin: '#f0b995', blush: '#f07f6e', blushAmount: 0.32, lip: '#d0706a', iris: '#3f8a58', irisSize: 0.7, hair: '#5a2f1a', lash: { thick: 0.085, wing: 0.42 }, brow: { in: [1.9, 4.72], top: [3.5, 5.1], out: [4.75, 4.65], thick: [0.55, 0.5, 0.16] } },
    body: { shoulder: 7.6, depth: 4.4, color: '#f2c14e' },
    hair: (head) => {
      const f = new Field(), m = hairMaterial('#6b3a1f');
      f.ell([0, 6.0, 0.4], [8.9, 10.3, 10.4]);
      f.half([0, 9.3, 9.6], [0, -1, 0.865], { mode: 'keep', k: 1.0 });
      for (const s of [-1, 1]) f.ell([s * 8.6, 1.3, 1.2], [2.4, 3.6, 2.8], { mode: 'cut', k: 0.6 });
      f.ell([-2.2, 10.4, 8.3], [4.6, 1.9, 2.6], { rot: [0.5, 0.2, 0.25], k: 0.5 });
      f.ell([3.0, 10.9, 7.9], [3.6, 1.7, 2.6], { rot: [0.5, -0.2, -0.3], k: 0.5 });
      head.addSculpt(f, m, 0.4);
    },
    wear: (fig) => {
      const ny = fig.body.neckY, c = new Field(); c.fn(ring([0, ny + 0.2, -0.9], [3.9, 4.1], 0.62, 0.12, 1.0));
      fig.addBody(c, cloth('#ffffff'), 0.3, [[-6, ny - 2.5, -7], [6, ny + 2.5, 5]]);
    },
  },
};

export async function loadCast() { await loadHeadBase(); }
export function createFigure(key) { return new Figure(DESIGNS[key] || DESIGNS.hana); }
