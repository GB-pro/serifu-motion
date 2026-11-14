// Serifu Motion — the face in an imported video: the blink and the mouth (the same two things the camera gives in
// live mode), and where the head turns and the eyes look (toward the other person or away). The video is played through once, muted and faster than real time,
// and every picture is handed to the face tracker.
//   const face = await trackVideoFace(file, T, base, onProgress)  ->  { seen: Uint8Array(T), blink: Float32Array(T), mouth: Float32Array(T * 8), pose: Float32Array(T * 5) } or null
import { FPS } from './features.js';
import { blendshapesToAU } from './au.js';
export const FACE_MOUTH = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];

// per picture and face, besides the blink and the mouth: where the head points and where the eyes look, as seen on
// the screen (FACE_POSE; degrees for the head: yaw > 0 toward the right of the picture, pitch > 0 down, roll > 0 the
// top of the head toward the left; gaze -1..1: x > 0 toward the right of the picture, y > 0 up)
export const FACE_POSE = ['headYaw', 'headPitch', 'headRoll', 'gazeX', 'gazeY'];

const trackers = new Map(), best = new Map(); // best: the faster way to run (GPU or CPU) found for n faces on this computer
async function getTracker(base, n = 1, delegate = 'GPU') {
  const key = n + ':' + delegate; if (trackers.has(key)) return trackers.get(key);
  const abs = (p) => new URL(base + p, document.baseURI).href;
  const { FilesetResolver, FaceLandmarker } = await import(abs('vendor/mediapipe/vision_bundle.mjs'));
  const fs = await FilesetResolver.forVisionTasks(abs('vendor/mediapipe/wasm'));
  const opts = (d) => ({ baseOptions: { modelAssetPath: abs('models/face_landmarker.task'), delegate: d }, runningMode: 'VIDEO', numFaces: n, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
  let tr; try { tr = await FaceLandmarker.createFromOptions(fs, opts(delegate)); } catch (e) { if (delegate === 'CPU') throw e; delegate = 'CPU'; tr = await FaceLandmarker.createFromOptions(fs, opts('CPU')); }
  const r = { tr, delegate, stamp: 0 }; trackers.set(key, r); trackers.set(n + ':' + delegate, r); return r;
}

const R2D = 180 / Math.PI, clamp = (x, a, b) => Math.min(b, Math.max(a, x));
// head direction from the face's transformation matrix (column major 4x4; columns 0..2 are the face's axes in camera space)
export function headPose(m) {
  let zx = m[8], zy = m[9], zz = m[10]; if (zz < 0) { zx = -zx; zy = -zy; zz = -zz; } // the face's forward axis, toward the camera
  let xx = m[0], xy = m[1]; if (xx < 0) { xx = -xx; xy = -xy; }
  return [Math.atan2(zx, zz) * R2D, Math.atan2(-zy, Math.hypot(zx, zz)) * R2D, Math.atan2(xy, xx) * R2D];
}
// gaze from the eye blendshapes ("Left" is the person's own left eye, which is on the right of the picture)
export function gazeOf(bs) {
  const g = (n) => bs[n] || 0;
  const x = (g('eyeLookOutLeft') + g('eyeLookInRight') - g('eyeLookInLeft') - g('eyeLookOutRight')) / 2;
  const y = (g('eyeLookUpLeft') + g('eyeLookUpRight') - g('eyeLookDownLeft') - g('eyeLookDownRight')) / 2;
  return [clamp(x * 1.8, -1, 1), clamp(y * 1.6, -1, 1)];
}

// faces: how many people to follow. With 1 the answer is one track or null (as before); with more it is an array of
// tracks ordered from the left of the picture to the right (each { seen, blink, mouth, pose, x, share }), possibly
// shorter than asked when fewer people show.
export async function trackVideoFace(file, T, base = '', onProgress = null, faces = 1) {
  const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.preload = 'auto';
  const url = URL.createObjectURL(file);
  try {
    await new Promise((ok, no) => { v.onloadeddata = ok; v.onerror = () => no(new Error('not a video')); v.src = url; });
    if (!v.videoWidth) return null; // sound only
    // the tracker runs on the graphics chip when it can; on some computers (a weak graphics chip, or none) the processor
    // is faster, so after a few pictures the other way is tried once and the faster one is kept
    let cam = await getTracker(base, faces, trackVideoFace.cpu ? 'CPU' : best.get(faces) || 'GPU'), trial = null, costA = 0, nLook = 0;
    const NM = FACE_MOUTH.length, NP = FACE_POSE.length;
    const dets = new Array(T).fill(null), got = new Uint8Array(T);
    let cost = 0; // cost: how long the tracker takes for one picture (ms)
    const look = (time) => {
      const t = Math.round(time * FPS); if (t < 0 || t >= T || got[t]) return;
      got[t] = 1;
      let r = null; const t0 = performance.now(); try { r = cam.tr.detectForVideo(v, (cam.stamp += 34)); } catch (e) { /* this picture is skipped */ }
      const dt = performance.now() - t0; cost = cost ? 0.8 * cost + 0.2 * dt : dt; nLook++;
      if (!best.has(faces) && !trial && nLook === 8 && cam.delegate === 'GPU' && cost > 30) {
        trial = getTracker(base, faces, 'CPU').then((c) => { costA = cost; cam = c; cost = 0; nLook = 100; }).catch(() => best.set(faces, 'GPU'));
      } else if (trial && nLook === 108) { if (cost > costA) { getTracker(base, faces, 'GPU').then((g) => { cam = g; cost = costA; }); best.set(faces, 'GPU'); } else best.set(faces, 'CPU'); }
      else if (!best.has(faces) && nLook === 8 && !trial) best.set(faces, cam.delegate);
      if (!r || !r.faceBlendshapes || !r.faceBlendshapes.length) return;
      dets[t] = r.faceBlendshapes.map((fb, i) => {
        const bs = {}; for (const c of fb.categories) bs[c.categoryName] = c.score;
        const au = blendshapesToAU(bs), lm = r.faceLandmarks && r.faceLandmarks[i];
        let x = 0.5; if (lm && lm.length) { let s = 0; for (const q of lm) s += q.x; x = s / lm.length; }
        const mtx = r.facialTransformationMatrixes && r.facialTransformationMatrixes[i], hp = mtx ? headPose(mtx.data) : [0, 0, 0], gz = gazeOf(bs);
        return { x, blink: Math.max(0, Math.min(1, (au.AU45 - 0.12) / 0.55)), mouth: FACE_MOUTH.map((id) => au[id] || 0), pose: [hp[0], hp[1], hp[2], gz[0], gz[1]] }; // the tracker reports about 0.12 with open eyes
      });
    };
    const dur = Math.min(T / FPS, v.duration || T / FPS), N = Math.min(T, Math.floor(dur * FPS));
    // first pass: play it through at double speed and look at every picture the browser hands over
    if (v.requestVideoFrameCallback) await new Promise((done) => {
      let over = false; let end = () => { if (!over) { over = true; done(); } };
      // the playback runs as fast as the tracker can look at every second picture: much faster than seeking to the
      // missed ones afterwards (each seek decodes the video again from the last key picture)
      // the browser stops playing a video in a tab that is not shown: pause with it, carry on when the tab is back, and
      // if the pictures stop coming for another reason, go on with what was seen (the second pass fills the gaps)
      let last = performance.now(), hiddenAt = 0;
      const onVis = () => { if (document.hidden) { hiddenAt = performance.now(); v.pause(); onProgress && onProgress(0.5 * Math.min(1, v.currentTime / dur), 'hidden'); } else { last = performance.now(); v.play().catch(() => {}); } };
      document.addEventListener('visibilitychange', onVis);
      const dog = setInterval(() => { if (over || document.hidden) return; const idle = performance.now() - last; if (idle > 3000 && v.paused) v.play().catch(() => {}); if (idle > 8000) end(); }, 1000);
      const end0 = end; end = () => { clearInterval(dog); document.removeEventListener('visibilitychange', onVis); end0(); }; v.onended = end; v.onerror = end;
      const cb = (now, meta) => { last = performance.now(); look(meta.mediaTime); if (cost) v.playbackRate = Math.max(0.1, Math.min(2, (2 / FPS) / (cost / 1000) * 0.9)); onProgress && onProgress(0.5 * Math.min(1, meta.mediaTime / dur)); if (!v.ended && meta.mediaTime < dur - 0.05) v.requestVideoFrameCallback(cb); else end(); };
      v.requestVideoFrameCallback(cb); v.playbackRate = 2; v.play().catch(end);
    });
    v.pause(); v.onended = null;
    // second pass: where the tracker could not keep up with the playback, go to every second picture that is still
    // missing and look at it (a blink lasts 4 to 6 pictures, so every second one is enough to catch it)
    const todo = []; for (let t = 0; t < N; t += 2) if (!got[t] && !got[t + 1]) todo.push(t);
    const until = performance.now() + (trackVideoFace.maxSeconds || Math.max(45, 1.5 * dur)) * 1000; // a slow device stops here and uses what it has seen
    for (let k = 0; k < todo.length && (performance.now() < until || trackVideoFace.maxSeconds === Infinity); k++) {
      const t = todo[k];
      if (document.hidden) { onProgress && onProgress(0.5 + 0.5 * k / todo.length, 'hidden'); await new Promise((ok) => { const f = () => { if (!document.hidden) { document.removeEventListener('visibilitychange', f); ok(); } }; document.addEventListener('visibilitychange', f); }); }
      await new Promise((ok) => { const tm = setTimeout(ok, 2000); v.onseeked = () => { clearTimeout(tm); ok(); }; v.currentTime = Math.min(dur - 0.01, t / FPS + 0.001); });
      look(t / FPS); onProgress && onProgress(0.5 + 0.5 * (k + 1) / todo.length);
    }
    v.onseeked = null;
    // who is who: the people sit in their places, so the faces are told apart by where they are across the picture
    const xs = []; for (const d of dets) if (d) for (const f of d) xs.push(f.x);
    xs.sort((a, b) => a - b);
    let C = []; for (let j = 0; j < faces; j++) C.push(xs.length ? xs[Math.min(xs.length - 1, Math.floor((j + 0.5) / faces * xs.length))] : 0.5);
    for (let it = 0; it < 20 && xs.length; it++) { const S = C.map(() => [0, 0]); for (const x of xs) { let b = 0; for (let j = 1; j < C.length; j++) if (Math.abs(x - C[j]) < Math.abs(x - C[b])) b = j; S[b][0] += x; S[b][1]++; } C = C.map((c, j) => (S[j][1] ? S[j][0] / S[j][1] : c)); }
    const tracks = C.map((cx) => ({ cx, seen: new Uint8Array(T), blink: new Float32Array(T), mouth: new Float32Array(T * NM), pose: new Float32Array(T * NP) }));
    for (let t = 0; t < T; t++) if (dets[t]) {
      const taken = new Set();
      for (const f of dets[t].slice().sort((a, b) => Math.min(...C.map((c) => Math.abs(a.x - c))) - Math.min(...C.map((c) => Math.abs(b.x - c))))) {
        let b = -1; for (let j = 0; j < C.length; j++) if (!taken.has(j) && (b < 0 || Math.abs(f.x - C[j]) < Math.abs(f.x - C[b]))) b = j;
        if (b < 0) continue; taken.add(b); const tr = tracks[b];
        tr.seen[t] = 1; tr.blink[t] = f.blink; tr.mouth.set(f.mouth, t * NM); tr.pose.set(f.pose, t * NP);
      }
    }
    trackVideoFace.last = { cost: Math.round(cost), delegate: cam.delegate, looked: got.reduce((x, y) => x + y, 0), found: tracks.reduce((s, tr) => s + tr.seen.reduce((x, y) => x + y, 0), 0) }; // for the tests
    const out = [];
    for (const tr of tracks) {
      const { seen, blink, mouth, pose } = tr;
      // pictures nobody looked at lie between two that were: they take the values in between
      for (let t = 0; t < T; t++) if (!got[t]) {
        let a = t - 1, b = t + 1; while (a >= 0 && !got[a] && t - a < 3) a--; while (b < T && !got[b] && b - t < 3) b++;
        const A = a >= 0 && got[a] && seen[a] === 1, B = b < T && got[b] && seen[b] === 1; if (!A && !B) continue;
        const u = A && B ? (t - a) / (b - a) : A ? 0 : 1, ia = A ? a : b, ib = B ? b : a;
        seen[t] = 2; blink[t] = blink[ia] + (blink[ib] - blink[ia]) * u;
        for (let j = 0; j < NM; j++) mouth[t * NM + j] = mouth[ia * NM + j] + (mouth[ib * NM + j] - mouth[ia * NM + j]) * u;
        for (let j = 0; j < NP; j++) pose[t * NP + j] = pose[ia * NP + j] + (pose[ib * NP + j] - pose[ia * NP + j]) * u;
      }
      for (let t = 0; t < T; t++) if (seen[t]) seen[t] = 1;
      // a face that shows for a moment only is not followed (a cut, a picture on the wall)
      for (let t = 0; t < T;) { if (seen[t]) { let e = t; while (e < T && seen[e]) e++; if (e - t < 10) seen.fill(0, t, e); t = e; } else t++; }
      let n = 0; for (let t = 0; t < T; t++) n += seen[t];
      if (n >= 15) out.push({ seen, blink, mouth, pose, x: tr.cx, share: n / T });
    }
    return faces === 1 ? out[0] || null : out;
  } finally { v.removeAttribute('src'); v.load(); URL.revokeObjectURL(url); }
}

// which face belongs to which voice: a speaker's mouth moves while they speak. segments: [{ start, end, speaker }];
// returns for each speaker 0..k-1 the index of its face in tracks, or -1.
export function matchFaces(tracks, segments, k, T) {
  const NM = FACE_MOUTH.length, talk = Array.from({ length: k }, () => new Uint8Array(T));
  for (const s of segments) if (s.speaker >= 0 && s.speaker < k) talk[s.speaker].fill(1, Math.max(0, Math.floor(s.start * FPS)), Math.min(T, Math.ceil(s.end * FPS)));
  // how much the mouth moves (change of the jaw opening) while each speaker talks and the others do not
  const score = tracks.map((tr) => talk.map((tk, j) => {
    let a = 0, na = 0, b = 0, nb = 0;
    for (let t = 1; t < T; t++) { if (!tr.seen[t] || !tr.seen[t - 1]) continue; const m = Math.abs(tr.mouth[t * NM] - tr.mouth[(t - 1) * NM]);
      const alone = tk[t] && talk.every((o, i) => i === j || !o[t]); const none = talk.every((o) => !o[t]) || (!tk[t] && talk.some((o) => o[t]));
      if (alone) { a += m; na++; } else if (none) { b += m; nb++; } }
    return na > 20 && nb > 20 ? a / na - b / nb : -1;
  }));
  // best pairs first
  const res = new Array(k).fill(-1), used = new Set(), pairs = [];
  score.forEach((row, i) => row.forEach((v, j) => pairs.push([v, i, j])));
  pairs.sort((p, q) => q[0] - p[0]);
  for (const [v, i, j] of pairs) if (v > 0 && res[j] < 0 && !used.has(i)) { res[j] = i; used.add(i); }
  // as many faces as voices: the ones left over go together (a face seen too little to judge still belongs to someone)
  if (tracks.length === k) { const free = tracks.map((_, i) => i).filter((i) => !used.has(i)); res.forEach((r, j) => { if (r < 0 && free.length === res.filter((q) => q < 0).length) res[j] = free.shift(); }); }
  matchFaces.last = score; return res;
}
