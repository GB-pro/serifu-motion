// Serifu Motion — realtime: mic drives expression/head/torso through the causal model,
// camera (MediaPipe Face Landmarker) drives ONLY blinks and mouth shape.
import { LiveEmo } from '../core/generator.js';
import { visemesFromCamera } from '../core/animate.js';
import { K, IDX, IDS, blendshapesToAU } from '../core/au.js';

const WORKLET = `class Tap extends AudioWorkletProcessor{process(i){const c=i[0];if(c&&c[0])this.port.postMessage(c[0].slice(0));return true}}registerProcessor('sm-tap',Tap);`;
const CAM_MOUTH = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];

export class LiveSession {
  constructor({ rtModel, onFrame, onStatus, video, base = './', profile = null, style = 'anime' }) {
    this.rtModel = rtModel; this.onFrame = onFrame; this.onStatus = onStatus || (() => {});
    this.profile = profile; this.style = style;
    this.video = video; this.base = base;
    this.useCamMouth = true; this.useCamBlink = true;
    this.queue = []; this.cam = null; this.faceSeen = false; this.recording = null;
  }
  async start() {
    this.emo = new LiveEmo(this.rtModel, { seed: Date.now() & 0xffff, profile: this.profile, style: this.style });
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
      const { FilesetResolver, FaceLandmarker } = await import(this.base + 'vendor/mediapipe/vision_bundle.mjs');
      const fs = await FilesetResolver.forVisionTasks(this.base + 'vendor/mediapipe/wasm');
      const opts = (delegate) => ({ baseOptions: { modelAssetPath: this.base + 'models/face_landmarker.task', delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true });
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
    const x = Float32Array.from(out);
    if (this.recording) this.recording.audio.push(x);
    // The camera gives two things only: the mouth shape and the blink. In the anime style the mouth shape goes INTO
    // the performer (so the emotion still shapes the lips and accents still open the jaw); the blink replaces the
    // performer's own blinks while a face is visible.
    const au = this.faceSeen ? this.camAU : null;
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
    const y = Float32Array.from(src), au = this.faceSeen ? this.camAU : null;
    if (au) {
      if (this.useCamBlink) y[IDX.AU45] = Math.max(0, Math.min(1, (au.AU45 - 0.12) / 0.55)); // the tracker reports about 0.12 with open eyes
      if (this.useCamMouth && this.style === 'natural') { const v = visemesFromCamera(au, this.emo.meta, this._vis2 || (this._vis2 = {})); for (const id of CAM_MOUTH) y[IDX[id]] = Math.min(1, v[id]); }
    }
    return y;
  }
  // called every animation frame; returns the latest composed emo frame (Float32Array K) or null
  tick(now) {
    if (!this.running) return null;
    let f = null; while (this.queue.length) f = this.queue.shift(); // newest generated frame
    if (f) this.lastFrame = f;
    if (!this.lastFrame) return null;
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
    const y = this._compose(this.lastFrame.y);
    return { y, emo: this.lastFrame.emo, voiced: this.lastFrame.voiced };
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
