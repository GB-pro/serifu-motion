// node tools/avatar/evalpage.mjs "<url>" "<js expression evaluated in the page>"
import { chromium } from 'playwright';
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 400, height: 400 } });
p.on('pageerror', e => console.log('ERR', e.message));
await p.goto(process.argv[2]); await p.waitForFunction('window.READY', null, { timeout: 60000 });
console.log(JSON.stringify(await p.evaluate(process.argv[3]), null, 1)); await b.close();
