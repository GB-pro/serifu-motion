// Serifu Motion — realtime: mic drives expression/head/torso through the causal model,
// camera (MediaPipe Face Landmarker) drives ONLY blinks and mouth shape.
import { LiveEmo } from '../core/generator.js';
import { visemesFromCamera } from '../core/animate.js';
import { K, IDX, IDS, EMOS, blendshapesToAU } from '../core/au.js';
import { LineListener } from './listener.js';

const WORKLET = `class Tap extends AudioWorkletProcessor{process(i){const c=i[0];if(c&&c[0])this.port.postMessage(c[0].slice(0));return true}}registerProcessor('sm-tap',Tap);`;
const CAM_MOUTH = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];

export class LiveSession {
  // delay (seconds, 0 = none) with meaning = { transcribe, understand }: the character plays that much later than the
  // voice, and what is said in the meantime is understood so that the feeling of the words is on the face in time
  constructor({ rtModel, rtExtra = [], onFrame, onStatus, video, base = './', profile = null, style = 'anime', delay = 0, meaning = null, onHeard = null }) {
    this.delay = meaning ? delay : 0; this.hold = []; this.heldN = 0; this.played = 0; this.marks = []; this.mark = null;
    if (this.delay > 0) this.listener = new LineListener({ ...meaning, maxAge: this.delay, onText: (text) => onHeard && onHeard({ text }), onMark: (m) => { this.marks.push(m); if (this.marks.length > 40) this.marks.shift(); onHeard && onHeard(m); } });
    this.rtModel = rtModel; this.rtExtra = rtExtra; this.onFrame = onFrame; this.onStatus = onStatus || (() => {});
    this.profile = profile; this.style = style;
    this.video = video; this.base = base;
    this.useCamMouth = true; this.useCamBlink = true;
    this.queue = []; this.cam = null; this.faceSeen = false; this.recording = null;
  }
  async start() {
    this.emo = new LiveEmo(this.rtModel, { seed: Date.now() & 0xffff, profile: this.profile, style: this.style, extra: this.rtExtra });
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    });
    this.video.srcObject = this.stream; this.video.muted = true; this.video.playsInline = true; await this.video.play().catch(() => {});
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ac = new AC();
    await this.ac.resume();
    const src = this.ac.createMediaStreamSource(this.stream);
    this.ratio = this.ac.sampleRate / 16000; this.acc = 0; this.accN = 0; this.phase = 0;
    const onChunk = (buf) => this._audio(buf);
    try {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await this.ac.audioWorklet.addModule(url);
      this.node = new AudioWorkletNode(this.ac, 'sm-tap'); this.node.port.onmessage = (e) => onChunk(e.data);
      src.connect(this.node);
      const sink = this.ac.createGain(); sink.gain.value = 0; this.node.connect(sink); sink.connect(this.ac.destination);
    } catch (e) {
      this.node = this.ac.createScriptProcessor(1024, 1, 1); this.node.onaudioprocess = (ev) => onChunk(ev.inputBuffer.getChannelData(0).slice(0));
      src.connect(this.node); const sink = this.ac.createGain(); sink.gain.value = 0; this.node.connect(sink); sink.connect(this.ac.destination);
    }
    this._loadFace();
    this.running = true;
    this.lastFrame = null;
  }
  async _loadFace() {
    try {
      // paths are resolved against the page: a dynamic import would otherwise be resolved against this file's folder
      const abs = (p) => new URL(this.base + p, document.baseURI).href;
      const { FilesetResolver, FaceLandmarker } = await import(abs('vendor/mediapipe/vision_bundle.mjs'));
      const fs = await FilesetResolver.forVisionTasks(abs('vendor/mediapipe/wasm'));
      const opts = (delegate) => ({ baseOptions: { modelAssetPath: abs('models/face_landmarker.task'), delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true });
      try { this.cam = await FaceLandmarker.createFromOptions(fs, opts('GPU')); } catch (e) { this.cam = await FaceLandmarker.createFromOptions(fs, opts('CPU')); }
      this.onStatus('cam-ready');
    } catch (e) { console.warn('face landmarker unavailable', e); this.onStatus('cam-failed'); }
  }
  _audio(buf) {
    // downsample to 16 kHz by box averaging
    const out = [];
    for (let i = 0; i < buf.length; i++) {
      this.acc += buf[i]; this.accN++; this.phase += 1;
      if (this.phase >= this.ratio) { out.push(this.acc / this.accN); this.acc = 0; this.accN = 0; this.phase -= this.ratio; }
    }
    const x = Float32Array.from(out), au = this.faceSeen ? this.camAU : null;
    if (!(this.delay > 0)) return this._perform(x, au);
    // with a delay: the listener hears it now, the performer gets it (and the face the camera saw with it) later
    this.listener.push(x); this.hold.push({ x, au }); this.heldN += x.length;
    while (this.hold.length && this.heldN - this.hold[0].x.length >= this.delay * 16000) {
      const h = this.hold.shift(); this.heldN -= h.x.length;
      // the feeling known for this stretch of sound; a piece that was cut while the speaker went on keeps its feeling until the next one is known
      const at = this.played + h.x.length / 2; let m = null; for (const k of this.marks) if (at >= k.start && (at < k.end || (k.open && at < k.end + 3 * 16000))) m = k;
      if (m !== this.mark) { this.mark = m; this.emo.force(m ? m.emo : null, m ? m.intensity : 0); }
      this.played += h.x.length; this._perform(h.x, h.au);
    }
  }
  _perform(x, au) {
    this.lateAU = au;
    if (this.recording) this.recording.audio.push(x);
    // The camera gives two things only: the mouth shape and the blink. In the anime style the mouth shape goes INTO
    // the performer (so the emotion still shapes the lips and accents still open the jaw); the blink replaces the
    // performer's own blinks while a face is visible.
    const camVis = au && this.useCamMouth ? visemesFromCamera(au, this.emo.meta, this._vis || (this._vis = {})) : null;
    this.emo.setAutoBlink(!(au && this.useCamBlink));
    for (const f of this.emo.push(x, camVis)) {
      this.queue.push(f);
      if (this.recording) { this.recording.frames.push(this._compose(f.y)); this.recording.emo.push(f.emo.slice()); }
    }
  }
  setProfile(profile) { this.profile = profile; if (this.emo) this.emo.setProfile(profile); }
  setStyle(style) { this.style = style; if (this.emo) this.emo.setStyle(style); }
  _compose(src) {
    const y = Float32Array.from(src), au = this.delay > 0 ? this.lateAU : this.faceSeen ? this.camAU : null;
    if (au) {
      if (this.useCamBlink) { // a clear close or open only (hysteresis), played smoothly: the raw value flickers (glasses, light)
        const raw = Math.max(0, Math.min(1, (au.AU45 - 0.12) / 0.55)); // the tracker reports about 0.12 with open eyes
        this._blT = raw > 0.6 ? 1 : raw < 0.35 ? 0 : (this._blT || 0); this._bl = (this._bl || 0) + (this._blT - (this._bl || 0)) * 0.55; y[IDX.AU45] = this._bl; }
      if (this.useCamMouth && this.style === 'natural') { const v = visemesFromCamera(au, this.emo.meta, this._vis2 || (this._vis2 = {})); for (const id of CAM_MOUTH) y[IDX[id]] = Math.min(1, v[id]); }
    }
    return y;
  }
  // called every animation frame; returns the latest composed emo frame (Float32Array K) or null
  tick(now) {
    if (!this.running) return null;
    // The frames are made 30 times a second, in small bursts as the sound arrives; the screen is drawn 60 or 120 times
    // a second. Showing the newest frame as it comes makes the face move in steps, so the frames are played out evenly,
    // about four frames behind the newest, and every picture is a blend of the two frames it falls between.
    if (this.queue.length > 12) this.queue.splice(0, this.queue.length - 4); // fell far behind (the tab was in the background)
    const dt = this._last ? Math.min(0.1, (now - this._last) / 1000) : 0; this._last = now;
    if (!this.lastFrame) { if (this.queue.length < 4) return null; this.lastFrame = this.queue.shift(); this.u = 0; }
    // the speed leans gently towards keeping about four frames in hand: a little faster when they pile up, slower when they run low
    this.rate = (this.rate || 1) + 0.02 * (Math.min(1.12, Math.max(0.88, 1 + 0.03 * (this.queue.length + (1 - this.u) - 4.5))) - (this.rate || 1));
    this.u += dt * 30 * this.rate;
    while (this.u >= 1 && this.queue.length) { this.lastFrame = this.queue.shift(); this.u -= 1; }
    const nxt = this.queue[0], u = nxt ? Math.min(1, this.u) : 0; if (!nxt) this.u = Math.min(this.u, 1);
    // camera: blink + mouth only
    if (this.cam && this.video.readyState >= 2) {
      try {
        const r = this.cam.detectForVideo(this.video, now);
        const cats = r.faceBlendshapes && r.faceBlendshapes[0] && r.faceBlendshapes[0].categories;
        this.faceSeen = !!cats;
        if (cats) {
          const bs = {}; for (const c of cats) bs[c.categoryName] = c.score;
          this.camAU = blendshapesToAU(bs);
        }
      } catch (e) { this.faceSeen = false; }
    }
    let src = this.lastFrame.y; if (nxt && u > 0) { src = this._mix || (this._mix = new Float32Array(K)); const a = this.lastFrame.y, b = nxt.y; for (let k = 0; k < K; k++) src[k] = a[k] + (b[k] - a[k]) * u; }
    const y = this._compose(src);
    // the meter shows what is being played: the feeling of the words while one is known, otherwise the reading of the voice
    const emo = this.mark ? EMOS.map((e) => (e === this.mark.emo ? Math.min(1, this.mark.intensity) : 0)) : this.lastFrame.emo;
    return { y, emo, voiced: this.lastFrame.voiced };
  }
  startRecording() { this.recording = { audio: [], frames: [], emo: [] }; }
  stopRecording() {
    const r = this.recording; this.recording = null; if (!r) return null;
    const n = r.audio.reduce((s, a) => s + a.length, 0), audio = new Float32Array(n);
    let o = 0; for (const a of r.audio) { audio.set(a, o); o += a.length; }
    const T = r.frames.length, base = new Float32Array(T * K), emo = new Float32Array(T * 6);
    r.frames.forEach((f, i) => base.set(f, i * K)); r.emo.forEach((e, i) => emo.set(e, i * 6));
    return { audio16k: audio, T, base, emo };
  }
  stop() {
    this.running = false;
    try { this.stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* ignore */ }
    try { this.ac.close(); } catch (e) { /* ignore */ }
  }
}
