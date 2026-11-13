// Render avatars from rig-space emo tracks to video.
// node render.mjs <spec.json>
// spec: { w, h, view?, jobs: [{ avatars: [...], tracks: [f32 path per avatar], out: mp4, audio?: wav, frames?: N, lead?: frames of rest before the track }] }
//   (a single job may also be given at the top level; a job may give framesDir instead of / in addition to out)
import { chromium } from 'playwright'; import fs from 'fs'; import { execFileSync } from 'child_process'; import path from 'path';
// Node may hand back small files as views into a shared pool: copy exactly the file's bytes.
function readF32(p) { const b = fs.readFileSync(p); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
const spec = JSON.parse(fs.readFileSync(process.argv[2]));
const jobs = spec.jobs || [spec];
const K = 40;
const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: spec.w, height: spec.h }, deviceScaleFactor: 1 });
p.on('pageerror', e => console.log('ERR', e.message));
await p.goto('http://localhost:8123/render.html'); await p.waitForFunction('window.READY', null, { timeout: 60000 });
const t0 = Date.now(); let total = 0;
for (const job of jobs) {
  const tracks = job.tracks.map(f => Array.from(readF32(f)));
  const N = job.frames || Math.max(...tracks.map(t => t.length / K));
  // framesDir: keep the JPEG frames there (for compositing) instead of a temporary folder
  const tmp = job.framesDir || fs.mkdtempSync('/tmp/claude-0/rf'); if (job.framesDir) fs.mkdirSync(tmp, { recursive: true });
  await p.evaluate(([a, v]) => setup(a, v), [job.avatars, job.view || spec.view || null]);
  await p.evaluate(t => setTracks(t), tracks);
  // ranges: [[a, b], ...] draw only these frame ranges (the frames between are still simulated)
  const want = (i) => !job.ranges || job.ranges.some(([a, b]) => i >= a && i < b);
  let n = 0;
  for (let i = 0; i < N; i++) {
    if (!want(i)) { let j = i; while (j < N && !want(j)) j++; await p.evaluate(([a, b]) => stepFrames(a, b), [i, j]); i = j - 1; continue; }
    const d = await p.evaluate(i => renderFrame(i), i); n++;
    fs.writeFileSync(path.join(tmp, String(i).padStart(5, '0') + '.jpg'), Buffer.from(d.split(',')[1], 'base64'));
  }
  if (job.out) {
    const args = ['-v', 'error', '-y', '-framerate', '30', '-i', path.join(tmp, '%05d.jpg')];
    if (job.audio) args.push('-i', job.audio, '-shortest');
    args.push('-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', String(job.crf ?? 18)); if (job.audio) args.push('-c:a', 'aac');
    args.push(job.out); execFileSync('ffmpeg', args);
  }
  if (!job.framesDir) fs.rmSync(tmp, { recursive: true });
  total += n; console.log('wrote', job.out || tmp, n, 'frames');
}
await b.close();
console.log('rendered', total, 'frames in', ((Date.now() - t0) / 1000).toFixed(1), 's');
