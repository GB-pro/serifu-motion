// node tools/dev/playout_test.mjs : the live playout (LiveSession.tick) with frames arriving in bursts and a 60 Hz screen: the value shown must rise evenly
import assert from 'assert';
globalThis.window = {}; const { LiveSession } = await import('../../app/rt/live.js'); const { K } = await import('../../app/core/au.js');
const s = new LiveSession({ rtModel: null, video: { readyState: 0 } }); s.running = true; s.emo = { meta: {} }; s.style = 'anime';
let made = 0; const shown = [];
for (let i = 0; i < 600; i++) { // 10 s of screen pictures
  const now = i * 1000 / 60;
  if (i % 8 === 0) for (let j = 0; j < 4; j++) { const y = new Float32Array(K); y[0] = made++; s.queue.push({ y, emo: [1, 0, 0, 0, 0, 0], voiced: 1 }); } // 4 frames every 8 pictures = 30 a second, in bursts
  const f = s.tick(now); if (f) shown.push(f.y[0]);
}
const d = shown.slice(30).map((v, i, a) => (i ? v - a[i - 1] : 0.5)).slice(1), max = Math.max(...d), min = Math.min(...d), lag = made - shown[shown.length - 1];
console.log('step per picture: min', min.toFixed(3), 'max', max.toFixed(3), '(even would be 0.5); behind the newest frame by', lag.toFixed(1), 'frames');
assert.ok(min > 0.4 && max < 0.6 && lag < 8); console.log('playout: ok');
