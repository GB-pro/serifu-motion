// UI smoke test: node uitest.mjs <outPrefix> <audio files...>   (static server on :8123)
import { chromium } from 'playwright';
const [out, ...files] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage({ viewport: { width: 1280, height: 800 } }); p.setDefaultTimeout(180000);
let errors = 0;
p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('console:', m.type(), m.text().slice(0, 300)); }); p.on('pageerror', e => { errors++; console.log('ERR', e.message); });
await p.goto('http://localhost:8123/index.html');
await p.waitForFunction('window.app && window.app.timeline', null, { timeout: 60000 });
const t0 = Date.now();
if (files.length) { await p.setInputFiles('#fileSep', files); await p.waitForFunction(`app.project.speakers.length >= ${files.length} && document.getElementById('busy').hidden`, null, { timeout: 240000 }); }
console.log('import+generate', ((Date.now() - t0) / 1000).toFixed(1), 's');
const info = await p.evaluate(() => app.project.speakers.map(s => ({ name: s.name, T: s.T, K: s.base.length / s.T, style: s.style, phrases: s.plan && s.plan.phrases.map(q => q.emo + (q.emo2 ? '+' + q.emo2 : '') + ':' + q.inten) })));
console.log(JSON.stringify(info));
await p.evaluate(() => app.seek(3.4)); await p.waitForTimeout(400);
await p.screenshot({ path: out + '_1.png' });
// click the second phrase block in the acting lane and set it to "sad"
const pos = await p.evaluate(() => { const tl = app.timeline, l = tl.layout.find(x => x.type === 'emo'), ph = l.sp.plan.phrases[1], r = tl.canvas.getBoundingClientRect(); return { x: r.left + tl.t2x((ph.t0 + ph.t1) / 2 / 30), y: r.top + l.y + 10, before: ph.emo }; });
await p.mouse.click(pos.x, pos.y); await p.waitForTimeout(300);
const open = await p.evaluate(() => document.getElementById('emoDlg').open);
console.log('dialog open after click:', open, 'phrase was', pos.before);
if (open) {
  await p.click('#emoChoices button[data-emo="SAD"]'); await p.screenshot({ path: out + '_2.png' });
  await p.click('#emoDlg button[value="apply"]'); await p.waitForTimeout(400);
  console.log('after override:', JSON.stringify(await p.evaluate(() => app.project.speakers[0].plan.phrases.map(q => q.emo + (q.fixed ? '*' : '')))));
}
// style switch and expressiveness, undo
const t1 = Date.now();
await p.evaluate(() => { const sp = app.project.speakers[0]; sp.expressiveness = 1.4; app.redirect(sp); });
console.log('redirect ms', Date.now() - t1);
await p.evaluate(() => app.seek(5.6)); await p.waitForTimeout(300);
await p.screenshot({ path: out + '_3.png' });
await p.evaluate(() => app.undo()); await p.waitForTimeout(200);
console.log('after undo:', JSON.stringify(await p.evaluate(() => app.project.speakers[0].plan.phrases.map(q => q.emo + (q.fixed ? '*' : '')))));
// export + save/load round trip
const rt = await p.evaluate(async () => { const pj = app.saveProject(); const n0 = JSON.stringify(pj).length; await app.loadProject(pj); const sp = app.project.speakers[0]; return { bytes: n0, T: sp.T, phrases: sp.plan.phrases.length, exp: JSON.stringify(Object.keys((await import('./core/project.js')).exportEmo(app.project, app.composer).speakers[0])) }; });
console.log('roundtrip', JSON.stringify(rt));
console.log('page errors', errors);
await b.close();
