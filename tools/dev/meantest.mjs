// node tools/dev/meantest.mjs <audio or video file> <out.png> [nokey] : imports a file in the studio, lets the page transcribe it,
// answers the request to the external AI with an imitation (no key or network needed) and checks the feelings land on the acting track
import { chromium } from 'playwright';
const [file, shot, mode] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage({ viewport: { width: 1200, height: 800 }, locale: 'ja-JP' });
p.on('pageerror', (e) => console.log('ERR', e.message)); p.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console', m.text().slice(0, 300)); });
let asked = null;
await p.route('https://api.anthropic.com/**', async (route) => {
  const req = route.request(); asked = { headers: req.headers(), body: JSON.parse(req.postData()) };
  const lines = JSON.parse(asked.body.messages[0].content).lines, emos = ['HAP', 'SAD', 'FEA'];
  const answer = { lines: lines.map((l, k) => ({ i: l.i, parts: k === 0 ? [{ share: 0.5, emo: 'HAP', intensity: 0.7 }, { share: 0.5, emo: 'SAD', intensity: 0.9 }] : [{ share: 1, emo: emos[k % 3], intensity: 0.8 }] })) };
  await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'access-control-allow-origin': '*' }, body: JSON.stringify({ stop_reason: 'end_turn', content: [{ type: 'text', text: JSON.stringify(answer) }] }) });
});
if (mode !== 'nokey') await p.addInitScript(() => localStorage.setItem('sm.ai.key', 'test-key-not-real'));
await p.goto(process.env.SITE || 'http://127.0.0.1:8765/index.html'); await p.waitForSelector('#fileSep', { state: 'attached', timeout: 60000 });
const t0 = Date.now();
await p.setInputFiles('#fileSep', file);
await p.waitForFunction(() => !document.querySelector('#aiInfo').hidden && document.querySelector('#busy').hidden, null, { timeout: 600000 });
console.log('seconds', ((Date.now() - t0) / 1000).toFixed(1));
console.log('note', await p.textContent('#aiInfo'));
const st = await p.evaluate(() => { const sp = window.app.project.speakers[0]; return { face: sp.face ? { share: sp.face.share, blinkMax: Math.max(...sp.face.blink), jawMax: Math.max(...sp.face.mouth.filter((v, i) => i % 8 === 0)) } : null, script: sp.script, ov: sp.emoOverrides, plan: sp.plan.phrases.map((q) => [+(q.t0 / 30).toFixed(1), q.emo, q.inten, q.fixed]) }; });
console.log(JSON.stringify(st, null, 1).slice(0, 3000));
if (asked) console.log('asked', asked.body.model, Object.keys(asked.headers).filter((h) => /anthropic|x-api/.test(h)), asked.body.messages[0].content.slice(0, 300));
await p.screenshot({ path: shot });
await b.close();
