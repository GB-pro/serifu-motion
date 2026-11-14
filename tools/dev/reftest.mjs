// node tools/dev/reftest.mjs <video file> <out.png> : imports a video file in the studio and checks the reference window follows the playhead
import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage({ viewport: { width: 1100, height: 760 } });
p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://127.0.0.1:8765/index.html'); await p.waitForSelector('#fileSep', { state: 'attached', timeout: 60000 });
await p.setInputFiles('#fileSep', process.argv[2]); await p.check('input[name=loadMode][value=sep]'); await p.click('#startBtn');
await p.waitForFunction(() => !document.querySelector('#refBox').hidden && document.querySelector('#busy').hidden !== false, null, { timeout: 120000 });
await p.waitForTimeout(1500);
const st = async () => p.evaluate(() => { const v = document.querySelector('#refVideo'); return { w: v.videoWidth, t: +v.currentTime.toFixed(2), paused: v.paused, clock: document.querySelector('#clock').textContent }; });
console.log('loaded', await st());
await p.click('#play'); await p.waitForTimeout(2000); console.log('playing', await st());
await p.click('#play'); await p.waitForTimeout(400); console.log('paused', await st());
await p.screenshot({ path: process.argv[3] });
await p.click('#refHide'); console.log('hidden', await p.evaluate(() => [document.querySelector('#refBox').hidden, document.querySelector('#refShow').hidden]));
await b.close();
