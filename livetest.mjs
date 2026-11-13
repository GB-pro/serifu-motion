// Live tab smoke test without a phone: a wav file is played as the microphone and, optionally, a video file as the camera
// (Chromium's fake capture devices). It checks that the live path runs in a real browser (microphone -> streaming
// features -> causal network -> live director -> avatar), that the face tracker loads and, with a camera file, that the
// camera's blink and mouth reach the avatar, that a take can be recorded and lands on the timeline, and that no page
// error occurs. It does not replace a check on a real phone (Safari, real camera and microphone, performance).
// node livetest.mjs <outPrefix> <mic.wav: 48 kHz mono 16-bit> [seconds=14] [camera.y4m]   (static server on :8123)
import { chromium } from 'playwright';
const [out, wav, secArg = '14', cam = ''] = process.argv.slice(2);
const args = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required',
  '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-audio-capture=' + wav];
if (cam) args.push('--use-file-for-fake-video-capture=' + cam);
const b = await chromium.launch({ args });
const ctx = await b.newContext({ viewport: { width: +(process.env.VW || 1280), height: +(process.env.VH || 800) }, permissions: ['camera', 'microphone'] }); // a small window renders faster without a GPU
const p = await ctx.newPage(); p.setDefaultTimeout(180000);
let errors = 0;
p.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console:', m.type(), m.text().slice(0, 300)); });
p.on('pageerror', (e) => { errors++; console.log('ERR', e.message); });
p.on('dialog', (d) => { errors++; console.log('DIALOG', d.message()); d.dismiss(); });
await p.goto('http://localhost:8123/index.html');
await p.waitForFunction('window.app && window.app.timeline', null, { timeout: 60000 });
await p.click('.tabs button[data-tab="live"]');
await p.click('#liveStart');
await p.waitForFunction('app.live && app.live.running', null, { timeout: 60000 });
if (cam) await p.waitForFunction('app.live && app.live.cam', null, { timeout: 120000 }).catch(() => console.log('face tracker did not load in time'));
await p.click('#liveRec');
// what the camera reports and what the avatar shows, sampled in the page at every animation frame
await p.evaluate(() => {
  window.__s = [];
  const step = (now) => { const l = app.live; if (!l) return; const f = l.lastFrame, c = l.faceSeen && l.camAU ? l.camAU : null, y = f ? l._compose(f.y) : null;
    window.__s.push({ t: now, face: !!c, cBlink: c ? c.AU45 : null, cJaw: c ? c.AU26 : null, aBlink: y ? y[19] : null, aJaw: y ? y[17] : null }); requestAnimationFrame(step); };
  requestAnimationFrame(step);
});
const seen = [];
for (let i = 0; i < +secArg * 2; i++) {
  await p.waitForTimeout(500);
  seen.push(await p.evaluate(() => (app.live && app.live.emo ? { emo: app.live.emo.dir.emo, voiced: app.live.lastFrame ? app.live.lastFrame.voiced : null, calib: +app.live.emo.feat.calibration.toFixed(2), frames: app.live.recording ? app.live.recording.frames.length : 0, face: !!app.live.faceSeen, cam: !!app.live.cam, rate: app.live.ac.sampleRate } : null)));
}
const last = seen[seen.length - 1] || {};
console.log('played, twice a second:', seen.map((s) => (s ? (s.voiced ? s.emo : s.emo.toLowerCase()) : '-')).join(' '));
console.log('frames recorded', last.frames, 'in', secArg, 's; mic sample rate', last.rate, '; voice calibration', last.calib, '; face tracker loaded', last.cam, '; face seen', last.face);
const s = await p.evaluate(() => window.__s);
if (s.length) {
  const withFace = s.filter((x) => x.face), r = (a, b) => { const n = a.length; if (n < 8) return NaN; const ma = a.reduce((u, v) => u + v, 0) / n, mb = b.reduce((u, v) => u + v, 0) / n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb + 1e-12); };
  const rng = (v) => (v.length ? [Math.min(...v).toFixed(2), Math.max(...v).toFixed(2)] : null);
  console.log('animation frames', s.length, 'in', ((s[s.length - 1].t - s[0].t) / 1000).toFixed(1), 's; with a face', withFace.length);
  if (withFace.length) console.log('camera blink range', rng(withFace.map((x) => x.cBlink)), 'camera jaw range', rng(withFace.map((x) => x.cJaw)),
    '| avatar blink range', rng(withFace.map((x) => x.aBlink)), 'avatar jaw range', rng(withFace.map((x) => x.aJaw)),
    '| r(camera blink, avatar blink)', r(withFace.map((x) => x.cBlink), withFace.map((x) => x.aBlink)).toFixed(2),
    // the mouth passes through the performer and the audio queue, so the avatar follows the camera a few frames later
    'r(camera jaw, avatar jaw) at a lag of 0..4 animation frames', [0, 1, 2, 3, 4].map((k) => r(withFace.slice(0, withFace.length - k).map((x) => x.cJaw), withFace.slice(k).map((x) => x.aJaw)).toFixed(2)).join(' '));
}
await p.screenshot({ path: out + '_live.png' });
await p.click('#liveRec'); await p.waitForTimeout(1200);
const info = await p.evaluate(() => ({ tabStudio: !document.getElementById('studio').hidden, speakers: app.project.speakers.map((sp) => ({ name: sp.name, T: sp.T, recorded: !!sp.recorded, K: sp.base ? sp.base.length / sp.T : 0, finite: sp.base ? Array.from(sp.base).every(Number.isFinite) : false })) }));
console.log('after recording:', JSON.stringify(info));
await p.screenshot({ path: out + '_studio.png' });
console.log('page errors', errors);
await b.close();
