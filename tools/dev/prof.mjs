import { chromium } from 'playwright'; import fs from 'fs';
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 480, height: 480 }, deviceScaleFactor: 1 });
p.on('pageerror', e => console.log('ERR', e.message)); p.on('console', m => console.log('console:', m.text()));
await p.goto('http://localhost:8123/render.html'); await p.waitForFunction('window.READY', null, { timeout: 60000 });
await p.evaluate(([a, v]) => setup(a, v), [['hana'], { y: 0.95, ly: 0.7, z: 7.2 }]);
await p.evaluate(t => setTracks(t), [Array.from(readF32(process.argv[2]))]);
const r = await p.evaluate(async () => {
  const st = window.__stage; const out = {};
  const gl = st.renderer.getContext(); const px = new Uint8Array(4);
  const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const time = (f, n = 6) => { const t = performance.now(); for (let i = 0; i < n; i++) f(i); return +((performance.now() - t) / n).toFixed(1); };
  out.render = time(() => { st.render(); sync(); });
  out.toData = time(() => st.canvas.toDataURL('image/jpeg', 0.92));
  st.renderer.shadowMap.enabled = false; st.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; if (o.isLight) o.castShadow = false; });
  st.render(); sync();
  out.noShadow = time(() => { st.render(); sync(); });
  const env = st.scene.environment; st.scene.environment = null; st.scene.traverse(o => { if (o.material) o.material.needsUpdate = true; }); st.render(); sync();
  out.noEnv = time(() => { st.render(); sync(); });
  st.scene.environment = env;
  out.aa = gl.getContextAttributes().antialias + ' samples ' + gl.getParameter(gl.SAMPLES);
  return out;
});
console.log(r);
await b.close();
