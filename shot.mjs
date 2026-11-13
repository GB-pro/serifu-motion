import { chromium } from 'playwright';
const [url, out, w = 1200, h = 700] = process.argv.slice(2);
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: +w, height: +h } });
p.on('console', m => console.log('console:', m.text())); p.on('pageerror', e => console.log('ERR', e.message));
await p.goto(url); await p.waitForFunction('window.READY', null, { timeout: 60000 }); await p.screenshot({ path: out }); await b.close();
