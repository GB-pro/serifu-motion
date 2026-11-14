import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 400, height: 400 } });
p.on('console', m => console.log('console:', m.text())); p.on('pageerror', e => console.log('ERR', e.message, e.stack));
await p.goto(process.argv[2]); await p.waitForTimeout(6000); await b.close();
