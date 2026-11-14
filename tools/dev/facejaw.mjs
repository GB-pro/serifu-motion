// node tools/dev/facejaw.mjs <video> <out.json> : jaw opening per frame of the face in a (cropped) video, for building a who-speaks reference
import { chromium } from 'playwright'; import fs from 'fs';
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage(); p.on('pageerror', (e) => console.log('ERR', e.message)); p.setDefaultTimeout(0);
await p.goto('http://127.0.0.1:8765/index.html'); await p.waitForSelector('#fileSep', { state: 'attached' });
const data = fs.readFileSync(process.argv[2]).toString('base64');
const r = await p.evaluate(async (b64) => {
  const { trackVideoFace } = await import('./core/facetrack.js');
  const file = new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], 'v.mp4', { type: 'video/mp4' });
  const f = await trackVideoFace(file, 6000, '', null);
  return f ? { seen: Array.from(f.seen), jaw: Array.from(f.mouth.filter((v, i) => i % 8 === 0)).map((v) => +v.toFixed(3)) } : null;
}, data);
fs.writeFileSync(process.argv[3], JSON.stringify(r)); console.log('done', r && r.seen.reduce((a, c) => a + c, 0)); await b.close();
