// node tools/dev/facejaw_img.mjs <dir under app/ with 00001.jpg...> <n> <out.json> : jaw opening per image (face tracker in IMAGE mode)
import { chromium } from 'playwright'; import fs from 'fs';
const [dir, n, out] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage(); p.on('pageerror', (e) => console.log('ERR', e.message)); p.setDefaultTimeout(0);
await p.goto('http://127.0.0.1:8765/index.html'); await p.waitForSelector('#fileSep', { state: 'attached' });
const r = await p.evaluate(async ([dir, n]) => {
  const { FilesetResolver, FaceLandmarker } = await import('./vendor/mediapipe/vision_bundle.mjs');
  const fs = await FilesetResolver.forVisionTasks(new URL('vendor/mediapipe/wasm', location.href).href);
  const fl = await FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: 'models/face_landmarker.task', delegate: 'CPU' }, runningMode: 'IMAGE', numFaces: 1, outputFaceBlendshapes: true });
  const jaw = [];
  for (let i = 1; i <= n; i++) {
    const img = new Image(); img.src = `${dir}/${String(i).padStart(5, '0')}.jpg`; await img.decode();
    const res = fl.detect(img); const c = res.faceBlendshapes?.[0]?.categories; jaw.push(c ? +c.find((q) => q.categoryName === 'jawOpen').score.toFixed(3) : -1);
  }
  return jaw;
}, [dir, +n]);
fs.writeFileSync(out, JSON.stringify(r)); console.log('found', r.filter((v) => v >= 0).length, 'of', r.length); await b.close();
