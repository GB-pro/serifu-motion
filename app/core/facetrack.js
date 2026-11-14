// Serifu Motion — the blink and the mouth of the face in an imported video (the same two things the camera gives in
// live mode; nothing else of the face is used). The video is played through once, muted and faster than real time,
// and every picture is handed to the face tracker.
//   const face = await trackVideoFace(file, T, base, onProgress)  ->  { seen: Uint8Array(T), blink: Float32Array(T), mouth: Float32Array(T * 8) } or null
import { FPS } from './features.js';
import { blendshapesToAU } from './au.js';
export const FACE_MOUTH = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];

let tracker = null;
async function getTracker(base) {
  if (tracker) return tracker;
  const abs = (p) => new URL(base + p, document.baseURI).href;
  const { FilesetResolver, FaceLandmarker } = await import(abs('vendor/mediapipe/vision_bundle.mjs'));
  const fs = await FilesetResolver.forVisionTasks(abs('vendor/mediapipe/wasm'));
  const opts = (delegate) => ({ baseOptions: { modelAssetPath: abs('models/face_landmarker.task'), delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true });
  try { tracker = await FaceLandmarker.createFromOptions(fs, opts('GPU')); } catch (e) { tracker = await FaceLandmarker.createFromOptions(fs, opts('CPU')); }
  return tracker;
}

export async function trackVideoFace(file, T, base = '', onProgress = null) {
  const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.preload = 'auto';
  const url = URL.createObjectURL(file);
  try {
    await new Promise((ok, no) => { v.onloadeddata = ok; v.onerror = () => no(new Error('not a video')); v.src = url; });
    if (!v.videoWidth) return null; // sound only
    const cam = await getTracker(base);
    const seen = new Uint8Array(T), blink = new Float32Array(T), mouth = new Float32Array(T * FACE_MOUTH.length), got = new Uint8Array(T);
    let stamp = 0;
    const look = (time) => {
      const t = Math.round(time * FPS); if (t < 0 || t >= T || got[t]) return;
      got[t] = 1;
      let cats = null; try { const r = cam.detectForVideo(v, (stamp += 34)); cats = r.faceBlendshapes && r.faceBlendshapes[0] && r.faceBlendshapes[0].categories; } catch (e) { /* this picture is skipped */ }
      if (!cats) return;
      const bs = {}; for (const c of cats) bs[c.categoryName] = c.score;
      const au = blendshapesToAU(bs);
      seen[t] = 1; blink[t] = Math.max(0, Math.min(1, (au.AU45 - 0.12) / 0.55)); // the tracker reports about 0.12 with open eyes
      FACE_MOUTH.forEach((id, j) => { mouth[t * FACE_MOUTH.length + j] = au[id] || 0; });
    };
    const dur = Math.min(T / FPS, v.duration || T / FPS), N = Math.min(T, Math.floor(dur * FPS));
    // first pass: play it through at double speed and look at every picture the browser hands over
    if (v.requestVideoFrameCallback) await new Promise((done) => {
      let over = false; const end = () => { if (!over) { over = true; done(); } }; v.onended = end; v.onerror = end;
      const cb = (now, meta) => { look(meta.mediaTime); onProgress && onProgress(0.5 * Math.min(1, meta.mediaTime / dur)); if (!v.ended && meta.mediaTime < dur - 0.05) v.requestVideoFrameCallback(cb); else end(); };
      v.requestVideoFrameCallback(cb); v.playbackRate = 2; v.play().catch(end);
    });
    v.pause(); v.onended = null;
    // second pass: where the tracker could not keep up with the playback, go to every second picture that is still
    // missing and look at it (a blink lasts 4 to 6 pictures, so every second one is enough to catch it)
    const todo = []; for (let t = 0; t < N; t += 2) if (!got[t] && !got[t + 1]) todo.push(t);
    for (let k = 0; k < todo.length; k++) {
      const t = todo[k];
      await new Promise((ok) => { const tm = setTimeout(ok, 2000); v.onseeked = () => { clearTimeout(tm); ok(); }; v.currentTime = Math.min(dur - 0.01, t / FPS + 0.001); });
      look(t / FPS); onProgress && onProgress(0.5 + 0.5 * (k + 1) / todo.length);
    }
    v.onseeked = null;
    trackVideoFace.last = { looked: got.reduce((x, y) => x + y, 0), found: seen.reduce((x, y) => x + y, 0) }; // for the tests
    // pictures nobody looked at lie between two that were: they take the values in between
    const NM = FACE_MOUTH.length;
    for (let t = 0; t < T; t++) if (!got[t]) {
      let a = t - 1, b = t + 1; while (a >= 0 && !got[a] && t - a < 3) a--; while (b < T && !got[b] && b - t < 3) b++;
      const A = a >= 0 && got[a] && seen[a], B = b < T && got[b] && seen[b]; if (!A && !B) continue;
      const u = A && B ? (t - a) / (b - a) : A ? 0 : 1, ia = A ? a : b, ib = B ? b : a;
      seen[t] = 2; blink[t] = blink[ia] + (blink[ib] - blink[ia]) * u; for (let j = 0; j < NM; j++) mouth[t * NM + j] = mouth[ia * NM + j] + (mouth[ib * NM + j] - mouth[ia * NM + j]) * u;
    }
    for (let t = 0; t < T; t++) if (seen[t]) seen[t] = 1;
    // a face that shows for a moment only is not followed (a cut, a picture on the wall)
    for (let t = 0; t < T;) { if (seen[t]) { let e = t; while (e < T && seen[e]) e++; if (e - t < 10) seen.fill(0, t, e); t = e; } else t++; }
    let n = 0; for (let t = 0; t < T; t++) n += seen[t];
    return n >= 15 ? { seen, blink, mouth, share: n / T } : null;
  } finally { v.removeAttribute('src'); v.load(); URL.revokeObjectURL(url); }
}
