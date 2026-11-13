// Render avatar frames from emo tracks to a video. spec.json: {w,h,avatars:[...],tracks:[path.f32...],audio?:wav, out:mp4, frames?:N, raw?:bool}
import { chromium } from 'playwright'; import fs from 'fs'; import { execFileSync } from 'child_process'; import path from 'path';
const spec = JSON.parse(fs.readFileSync(process.argv[2]));
const tracks = spec.tracks.map(p => Array.from(new Float32Array(fs.readFileSync(p).buffer.slice(0))));
const K = 30; const N = spec.frames || Math.max(...tracks.map(t => t.length / K));
const tmp = fs.mkdtempSync('/tmp/claude-0/rf');
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: spec.w, height: spec.h }, deviceScaleFactor: 1 });
p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://localhost:8123/render.html'); await p.waitForFunction('window.READY', null, { timeout: 60000 });
await p.evaluate(a => setup(a), spec.avatars);
await p.evaluate(t => { window._T = t; }, tracks);
for (let i = 0; i < N; i++) {
  const d = await p.evaluate(([i, raw]) => renderFrame(window._T, i, { raw }), [i, !!spec.raw]);
  fs.writeFileSync(path.join(tmp, String(i).padStart(5, '0') + '.jpg'), Buffer.from(d.split(',')[1], 'base64'));
}
await b.close();
const args = ['-v', 'error', '-y', '-framerate', '30', '-i', path.join(tmp, '%05d.jpg')];
if (spec.audio) args.push('-i', spec.audio, '-shortest');
args.push('-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20'); if (spec.audio) args.push('-c:a', 'aac');
args.push(spec.out); execFileSync('ffmpeg', args); fs.rmSync(tmp, { recursive: true });
console.log('wrote', spec.out, N, 'frames');
