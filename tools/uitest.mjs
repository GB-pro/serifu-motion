import { chromium } from 'playwright';
const [out, ...files] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
p.on('console', m => { if (m.type() !== 'debug') console.log('console:', m.text()); }); p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://localhost:8123/index.html');
await p.waitForFunction('window.app && window.app.timeline', null, { timeout: 60000 });
if (files.length) { await p.setInputFiles('#fileSep', files); await p.waitForFunction(`app.project.speakers.length >= ${files.length} && document.getElementById('busy').hidden`, null, { timeout: 120000 }); }
await p.evaluate(() => app.seek(1.0));
await p.waitForTimeout(500);
await p.screenshot({ path: out }); await b.close();
