// node tools/dev/facetest.mjs <video file> : what the face tracker gets out of a video (share of pictures with a face, blinks, how far the jaw opens)
import { chromium } from 'playwright'; import fs from 'fs';
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage(); p.on('pageerror', (e) => console.log('ERR', e.message));
await p.goto('http://127.0.0.1:8765/index.html'); await p.waitForSelector('#fileSep', { state: 'attached' });
const data = fs.readFileSync(process.argv[2]).toString('base64');
const r = await p.evaluate(async (b64) => {
  const { trackVideoFace } = await import('./core/facetrack.js');
  const file = new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], 'clip.mp4', { type: 'video/mp4' });
  const T = 600, t0 = performance.now(), f = await trackVideoFace(file, T, '', null);
  if (!f) return { face: null, ms: performance.now() - t0, last: trackVideoFace.last };
  let blinks = 0; for (let t = 1; t < T; t++) if (f.blink[t] > 0.5 && f.blink[t - 1] <= 0.5) blinks++;
  return { last: trackVideoFace.last, ms: Math.round(performance.now() - t0), share: f.share, blinks, jawMax: Math.max(...f.mouth.filter((v, i) => i % 8 === 0)), seen: Array.from(f.seen).join('').replace(/(.)\1*/g, (m, c) => `${c}x${m.length} `) };
}, data);
console.log(JSON.stringify(r)); await b.close();
