// Contact sheet of the head for development: one page, several sets of weights, one image.
//   node tools/avatar/sheet.mjs <out.png> "<query of head_dev.html>" <cell px> <columns> name=w1:v,w2:v[@yaw,pitch] ...
// A cell can override the query: name=...|cam=30,0,62  (then the page is loaded again for that cell).
import { chromium } from 'playwright';
const [out, query, cell = '360', cols = '4', ...cells] = process.argv.slice(2);
const port = process.env.PORT || 8765, C = +cell, page = process.env.PAGE || 'head_dev.html'; // PAGE=figure_dev.html for a whole character
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: C, height: C } });
p.on('pageerror', (e) => console.log('ERR', e.message)); p.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });
const kv = (s) => Object.fromEntries((s || '').split(',').filter(Boolean).map((x) => { const [k, v] = x.split(':'); return [k, +v]; }));
let loaded = null; const shots = [];
for (const c of cells) {
  const [head, extra] = c.split('|'), eq = head.indexOf('='), name = eq < 0 ? head : head.slice(0, eq), rest = eq < 0 ? '' : head.slice(eq + 1);
  const [ws, look] = rest.split('@'), qp = new URLSearchParams(query); for (const [k, v] of new URLSearchParams(extra || '')) qp.set(k, v); const q = qp.toString();
  if (loaded !== q) { await p.goto(`http://127.0.0.1:${port}/${page}?${q}`); await p.waitForFunction('window.READY', null, { timeout: 60000 }); loaded = q; }
  await p.evaluate(([w, l]) => window.setW(w, l), [kv(ws), look ? look.split(',').map(Number) : [0, 0]]);
  shots.push({ name, data: (await p.screenshot({ type: 'jpeg', quality: 92 })).toString('base64') });
}
const n = +cols, rows = Math.ceil(shots.length / n);
const g = await b.newPage({ viewport: { width: C * n, height: C * rows } });
await g.setContent(`<body style="margin:0;display:grid;grid-template-columns:repeat(${n},${C}px);font:13px sans-serif">${shots.map((s) => `<div style="position:relative;width:${C}px;height:${C}px"><img src="data:image/jpeg;base64,${s.data}" width="${C}" height="${C}"><span style="position:absolute;left:6px;top:4px;background:#fffc;padding:1px 5px;border-radius:3px">${s.name}</span></div>`).join('')}</body>`);
await g.screenshot({ path: out, type: out.endsWith('.png') ? 'png' : 'jpeg' }); await b.close();
