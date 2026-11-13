// node sheet.mjs <spec.json> <out.png>   spec: {mode:'sheet', char, cells, cols, cw, ch, view} | {mode:'strip', char, track:<f32 path>, idx:[...], cols, cw, ch, view}
import { chromium } from 'playwright'; import fs from 'fs';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const spec = JSON.parse(fs.readFileSync(process.argv[2])), out = process.argv[3];
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 800, height: 600 }, deviceScaleFactor: 1 });
p.on('console', m => console.log('console:', m.text())); p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://localhost:8123/sheet.html'); await p.waitForFunction('window.READY', null, { timeout: 60000 });
let wh;
if (spec.mode === 'strip') wh = await p.evaluate(([s, tr]) => strip(s.char, tr, s.idx, s.cols || 8, s.cw || 240, s.ch || 270, s.view || {}, s.labels || null), [spec, Array.from(readF32(spec.track))]);
else wh = await p.evaluate((s) => sheet(s.char, s.cells, s.cols || 4, s.cw || 360, s.ch || 400, s.view || {}), spec);
await p.setViewportSize({ width: wh[0], height: wh[1] });
await p.screenshot({ path: out });
await b.close(); console.log('wrote', out, wh.join('x'));
