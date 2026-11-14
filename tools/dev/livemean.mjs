// node tools/dev/livemean.mjs <mic.wav: 48 kHz mono 16-bit> [seconds=26] [delay=6] : the live tab with "read the meaning" on. The wav plays as the
// microphone, the external AI is imitated (it answers after 1.2 s and calls lines with a question mark FEA, others ANG), and the script prints
// what was heard, which feeling was set for which stretch of sound, and how far ahead of the delayed playback each answer arrived.
import { chromium } from 'playwright';
const [wav, sec = '26', delay = '6'] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-audio-capture=' + wav] });
const ctx = await b.newContext({ viewport: { width: 480, height: 420 }, permissions: ['camera', 'microphone'] });
const p = await ctx.newPage(); p.setDefaultTimeout(180000);
p.on('pageerror', (e) => console.log('ERR', e.message)); p.on('console', (m) => { if (m.type() === 'error') console.log('console', m.text().slice(0, 200)); });
let calls = 0, model = '';
await p.route('https://api.anthropic.com/**', async (route) => {
  const body = JSON.parse(route.request().postData()), lines = JSON.parse(body.messages[0].content).lines; calls++; model = body.model;
  await new Promise((r) => setTimeout(r, 1200));
  const answer = { lines: lines.map((l) => ({ i: l.i, parts: [{ share: 1, emo: /\?/.test(l.text) ? 'FEA' : 'ANG', intensity: 0.9 }] })) };
  await route.fulfill({ status: 200, contentType: 'application/json', headers: { 'access-control-allow-origin': '*' }, body: JSON.stringify({ stop_reason: 'end_turn', content: [{ type: 'text', text: JSON.stringify(answer) }] }) });
});
await p.route('**/face_landmarker.task', (r) => r.abort()); // no face tracking here: without a GPU it would take the whole processor
await p.addInitScript(() => { localStorage.setItem('sm.ai.key', 'test-key-not-real'); localStorage.setItem('sm.ai.lang', 'en'); });
await p.goto('http://127.0.0.1:8765/index.html'); await p.waitForFunction('window.app && window.app.timeline');
await p.click('.tabs button[data-tab="live"]'); await p.selectOption('#liveDelay', delay);
await p.click('#liveStart'); await p.waitForFunction('app.live && app.live.running');
await p.evaluate(() => { const l = app.live; window.__m = []; const push = l.marks.push.bind(l.marks); l.marks.push = (m) => { window.__m.push({ ...m, heardAt: l.listener.n, playedAt: l.played }); return push(m); };
  window.__t = []; const tr = app.stt.line.bind(app.stt); app.stt.line = async (x, g) => { const t0 = performance.now(), r = await tr(x, g); window.__t.push([+(x.length / 16000).toFixed(1), Math.round(performance.now() - t0)]); return r; };
  window.__e = []; const force = l.emo.force.bind(l.emo); l.emo.force = (e, i) => { window.__e.push([+(l.played / 16000).toFixed(2), e, i]); return force(e, i); }; });
const w0 = Date.now(); await p.waitForTimeout(+sec * 1000);
const r = await p.evaluate(() => ({ asr: window.__t, marks: window.__m, forced: window.__e, heard: document.querySelector('#liveHeard').textContent, played: app.live.played / 16000, held: app.live.heldN / 16000 }));
for (const m of r.marks) console.log(`${(m.start / 16000).toFixed(2)}-${(m.end / 16000).toFixed(2)} ${m.emo} ${m.intensity} known ${((m.start - m.playedAt) / 16000).toFixed(2)} s before it plays${m.open ? ' (speaker goes on)' : ''} | ${m.text}`);
console.log('wall', ((Date.now() - w0) / 1000).toFixed(1), 'asr [line seconds, ms]', JSON.stringify(r.asr)); console.log('forced', JSON.stringify(r.forced)); console.log('last shown:', r.heard); console.log('requests', calls, model, 'played', r.played.toFixed(1), 'held', r.held.toFixed(1));
await p.screenshot({ path: process.env.SHOT || '/dev/null' }).catch(() => {});
await b.close();
