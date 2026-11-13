// Serifu Motion — application shell
import { EmoGenerator } from './core/generator.js';
import { IDS, K, EMOS, CHANNELS, GROUPS } from './core/au.js';
import { SR, FPS } from './core/features.js';
import { newSpeaker, analyzeSpeaker, generateSpeaker, diarize, Composer, insertGesture, exportEmo, exportCSV, GESTURES } from './core/project.js';
import { DESIGNS } from './avatar/characters.js';
import { Stage } from './ui/stage.js';
import { Timeline, EMO_COLORS } from './ui/timeline.js';
import { LANGS, detectLang, setLang, t, chName, getLang } from './ui/i18n.js';
import { LiveSession } from './rt/live.js';

const $ = (s) => document.querySelector(s);
const AVATARS = Object.keys(DESIGNS);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class App {
  constructor() {
    this.project = { speakers: [] };
    this.selected = null; this.undoStack = []; this.redoStack = [];
    this.playing = false; this.time = 0;
  }
  async init() {
    setLang(detectLang());
    const sel = $('#lang'); for (const [k, v] of Object.entries(LANGS)) sel.add(new Option(v, k)); sel.value = getLang();
    sel.onchange = () => { setLang(sel.value); this.i18n(); };
    this.i18n();
    this.busy(t('analyzing'));
    const [off, rt] = await Promise.all([fetch('models/emo_offline.json').then((r) => r.json()), fetch('models/emo_realtime.json').then((r) => r.json())]);
    this.rtModel = rt;
    this.gen = new EmoGenerator(off, rt);
    this.composer = new Composer(this.gen.meta, DESIGNS);
    $('#modelInfo').textContent = `${t('model')}: ${this.gen.meta.dataset} · ${this.gen.meta.trained_clips} clips`;
    this.stage = new Stage($('#stage'), { preserve: true });
    this.timeline = new Timeline($('#timeline'), this);
    this.bindUI();
    this.busy(null);
    this.renderPanel();
    this.loop();
    if (new URLSearchParams(location.search).has('demo')) this.loadDemo();
  }
  i18n() {
    document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    if (this.timeline) { this.renderPanel(); this.timeline.draw(); }
  }
  busy(msg) { $('#busy').hidden = !msg; $('#busyText').textContent = msg || ''; }

  bindUI() {
    document.querySelectorAll('.tabs button').forEach((b) => b.onclick = () => this.tab(b.dataset.tab));
    $('#play').onclick = () => this.togglePlay();
    $('#fileSep').onchange = (e) => this.addSeparate([...e.target.files]).finally(() => (e.target.value = ''));
    $('#fileMix').onchange = (e) => this.addMixed(e.target.files[0], +$('#nSpk').value).finally(() => (e.target.value = ''));
    $('#fileProj').onchange = async (e) => { const f = e.target.files[0]; if (f) this.loadProject(JSON.parse(await f.text())); e.target.value = ''; };
    $('#demoBtn').onclick = () => this.loadDemo();
    $('#undo').onclick = () => this.undo(); $('#redo').onclick = () => this.redo();
    $('#zin').onclick = () => this.timeline.zoom(1.3); $('#zout').onclick = () => this.timeline.zoom(1 / 1.3);
    $('#exportBtn').onclick = () => { $('#exportMenu').hidden = !$('#exportMenu').hidden; };
    document.querySelectorAll('[data-exp]').forEach((b) => b.onclick = () => { $('#exportMenu').hidden = true; this.export(b.dataset.exp); });
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); this.togglePlay(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); }
    });
    const stageEl = $('#stagewrap') || $('.stagewrap');
    stageEl.addEventListener('dragover', (e) => e.preventDefault());
    stageEl.addEventListener('drop', (e) => { e.preventDefault(); this.addSeparate([...e.dataTransfer.files]); });
    $('#timeline').addEventListener('dragover', (e) => e.preventDefault());
    $('#timeline').addEventListener('drop', (e) => { e.preventDefault(); this.addSeparate([...e.dataTransfer.files]); });
    // emotion dialog
    const ch = $('#emoChoices');
    for (const e of EMOS) { const b = document.createElement('button'); b.type = 'button'; b.dataset.emo = e; b.style.background = EMO_COLORS[e]; b.onclick = () => { ch.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); }; ch.appendChild(b); }
    $('#emoInt').oninput = () => { $('#emoIntV').textContent = Math.round($('#emoInt').value * 100) + '%'; };
    this.bindLive();
  }
  tab(name) {
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
    $('#studio').hidden = name !== 'studio'; $('#live').hidden = name !== 'live';
    if (name === 'live' && !this.liveStage) { this.liveStage = new Stage($('#liveStage')); this.liveStage.setCharacters([$('#liveAvatar').value]); }
    if (name === 'studio') { this.timeline.draw(); }
    if (name !== 'live' && this.live) this.stopLive();
  }

  // ---------- audio import ----------
  async decode(file) {
    const AC = window.AudioContext || window.webkitAudioContext; const ac = new AC();
    const buf = await ac.decodeAudioData(await file.arrayBuffer()); ac.close();
    const n = Math.ceil(buf.duration * SR);
    const oc = new OfflineAudioContext(1, n, SR); const s = oc.createBufferSource(); s.buffer = buf; s.connect(oc.destination); s.start();
    const r = await oc.startRendering();
    return r.getChannelData(0).slice(0);
  }
  async addSeparate(files) {
    files = files.filter((f) => /audio|video/.test(f.type) || /\.(wav|mp3|m4a|aac|ogg|flac|mp4|mov|webm)$/i.test(f.name));
    if (!files.length) return;
    this.pushUndo();
    for (const f of files) {
      this.busy(t('analyzing') + ' ' + f.name);
      await sleep(30);
      const x = await this.decode(f);
      const i = this.project.speakers.length;
      const sp = newSpeaker(i, f.name.replace(/\.[^.]+$/, '').slice(0, 24), AVATARS[i % AVATARS.length]);
      sp.audio16k = x;
      await this.process(sp);
      this.project.speakers.push(sp);
    }
    this.padSpeakers();
    this.afterChange(); this.busy(null);
  }
  async addMixed(file, k) {
    if (!file) return;
    this.pushUndo();
    this.busy(t('analyzing')); await sleep(30);
    const x = await this.decode(file);
    const { segments } = diarize(x, k);
    const n0 = this.project.speakers.length;
    for (let j = 0; j < k; j++) {
      const a = new Float32Array(x.length);
      for (const s of segments) if (s.speaker === j) {
        const s0 = Math.floor(s.start * SR), s1 = Math.min(x.length, Math.ceil(s.end * SR)), fade = 800;
        for (let i = s0; i < s1; i++) { const w = Math.min(1, (i - s0) / fade, (s1 - i) / fade); a[i] = x[i] * w; }
      }
      const i = n0 + j;
      const sp = newSpeaker(i, `${t('speaker')} ${String.fromCharCode(65 + j)}`, AVATARS[i % AVATARS.length]);
      sp.audio16k = a;
      this.busy(t('generating') + ` (${j + 1}/${k})`); await sleep(30);
      await this.process(sp);
      this.project.speakers.push(sp);
    }
    this.padSpeakers();
    this.afterChange(); this.busy(null);
  }
  async process(sp) {
    analyzeSpeaker(sp); await sleep(10);
    generateSpeaker(sp, this.gen, 7 + this.project.speakers.length);
    this.recompose(sp);
  }
  padSpeakers() { // make all speaker tracks the same length (silence after shorter ones) so the dialogue stays aligned
    const T = Math.max(...this.project.speakers.map((s) => s.T));
    for (const sp of this.project.speakers) if (sp.T < T && sp.audio16k) {
      const a = new Float32Array(Math.ceil(T / FPS * SR)); a.set(sp.audio16k); sp.audio16k = a;
      if (sp.base && !sp.recorded) { analyzeSpeaker(sp); generateSpeaker(sp, this.gen, 7); }
      else if (sp.base) { const b = new Float32Array(T * K); b.set(sp.base); for (let t2 = sp.T; t2 < T; t2++) b.set(sp.base.subarray((sp.T - 1) * K, sp.T * K), t2 * K); const e = new Float32Array(T * 6); e.set(sp.emo); sp.base = b; sp.emo = e; sp.T = T; }
      this.recompose(sp);
    }
  }
  recompose(sp) {
    if (!sp.base) return;
    const f = sp.final && sp.final.length === sp.T * K ? sp.final : new Float32Array(sp.T * K);
    const tmp = new Float32Array(K);
    for (let i = 0; i < sp.T; i++) { this.composer.frame(sp, i, tmp); f.set(tmp, i * K); }
    sp.final = f;
  }
  afterChange() {
    if (!this.selected || !this.project.speakers.includes(this.selected)) this.selected = this.project.speakers[0] || null;
    this.stage.setCharacters(this.project.speakers.map((s) => s.avatar));
    this.playBuf = null;
    this.renderPanel(); this.timeline.draw(); this.applyFrame(this.time);
  }

  // ---------- panel ----------
  selectSpeaker(sp) { this.selected = sp; this.renderPanel(); this.timeline.draw(); }
  renderPanel() {
    const list = $('#spList'); list.innerHTML = '';
    for (const sp of this.project.speakers) {
      const d = document.createElement('div'); d.className = 'sp' + (sp === this.selected ? ' on' : '');
      d.innerHTML = `<span class="dot" style="background:${sp.color}"></span>`;
      const name = document.createElement('input'); name.value = sp.name; name.onchange = () => { sp.name = name.value; this.timeline.draw(); };
      const av = document.createElement('select'); for (const k of AVATARS) av.add(new Option(DESIGNS[k].label[getLang()] || DESIGNS[k].label.en, k)); av.value = sp.avatar;
      av.onchange = () => { sp.avatar = av.value; this.stage.setCharacters(this.project.speakers.map((s) => s.avatar)); this.applyFrame(this.time); };
      d.append(name, av); d.onclick = (e) => { if (e.target === d || e.target.className === 'dot') this.selectSpeaker(sp); };
      name.onfocus = () => { if (this.selected !== sp) this.selectSpeaker(sp); };
      list.appendChild(d);
    }
    const det = $('#spDetail'); det.innerHTML = '';
    const sp = this.selected; if (!sp) return;
    det.className = 'detail';
    const ex = document.createElement('label'); ex.className = 'mini'; ex.innerHTML = `${t('expressiveness')} <input type="range" min="0.3" max="2" step="0.05" value="${sp.expressiveness}"> <output>${Math.round(sp.expressiveness * 100)}%</output>`;
    const exIn = ex.querySelector('input'); exIn.oninput = () => { sp.expressiveness = +exIn.value; ex.querySelector('output').textContent = Math.round(sp.expressiveness * 100) + '%'; this.recompose(sp); this.timeline.draw(); this.applyFrame(this.time); };
    exIn.onpointerdown = () => this.pushUndo();
    det.appendChild(ex);
    const h1 = document.createElement('h3'); h1.textContent = t('gestures'); det.appendChild(h1);
    const chips = document.createElement('div'); chips.className = 'chips';
    for (const g of Object.keys(GESTURES)) { const b = document.createElement('button'); b.textContent = t(g); b.onclick = () => { this.pushUndo(); insertGesture(sp, g, this.time); this.onEdit(sp); }; chips.appendChild(b); }
    det.appendChild(chips);
    const h2 = document.createElement('h3'); h2.textContent = t('channels'); det.appendChild(h2);
    const cl = document.createElement('div'); cl.className = 'chlist';
    for (const c of CHANNELS) {
      const l = document.createElement('label'); const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = sp.visible.includes(c.id);
      cb.onchange = () => { sp.visible = CHANNELS.map((x) => x.id).filter((id) => (id === c.id ? cb.checked : sp.visible.includes(id))); this.timeline.draw(); };
      l.title = chName(c); l.append(cb, document.createTextNode(`${c.id.startsWith('AU') ? c.id + ' ' : ''}${chName(c)}`)); cl.appendChild(l);
    }
    det.appendChild(cl);
    const row = document.createElement('div'); row.className = 'row wrap'; row.style.marginTop = '10px';
    const clr = document.createElement('button'); clr.className = 'ghost'; clr.textContent = t('clearEdits'); clr.onclick = () => { this.pushUndo(); sp.edits = {}; sp.emoOverrides = []; this.onEdit(sp); };
    const regen = document.createElement('button'); regen.className = 'ghost'; regen.textContent = t('regenerate'); regen.disabled = !!sp.recorded;
    regen.onclick = () => { this.pushUndo(); generateSpeaker(sp, this.gen, (Math.random() * 1e6) | 0); this.onEdit(sp); };
    const del = document.createElement('button'); del.className = 'ghost danger'; del.textContent = t('delSpeaker'); del.onclick = () => { this.pushUndo(); this.project.speakers = this.project.speakers.filter((s) => s !== sp); this.afterChange(); };
    row.append(clr, regen, del); det.appendChild(row);
  }

  // ---------- editing ----------
  snapshot() { return this.project.speakers.map((s) => ({ s, edits: JSON.parse(JSON.stringify(s.edits)), ov: JSON.parse(JSON.stringify(s.emoOverrides)), ex: s.expressiveness, base: s.base, emo: s.emo, avatar: s.avatar })); }
  pushUndo() { this.undoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); if (this.undoStack.length > 80) this.undoStack.shift(); this.redoStack = []; }
  restore(st) { this.project.speakers = st.speakers.slice(); for (const x of st.snap) { x.s.edits = x.edits; x.s.emoOverrides = x.ov; x.s.expressiveness = x.ex; x.s.base = x.base; x.s.emo = x.emo; x.s.avatar = x.avatar; this.recompose(x.s); } this.afterChange(); }
  undo() { const st = this.undoStack.pop(); if (!st) return; this.redoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); this.restore(st); }
  redo() { const st = this.redoStack.pop(); if (!st) return; this.undoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); this.restore(st); }
  onEdit(sp) { this.recompose(sp); this.timeline.draw(); this.applyFrame(this.time); }
  openEmoDialog(sp, a, b, existing) {
    const dlg = $('#emoDlg');
    $('#emoRange').textContent = `${sp.name} · ${a.toFixed(2)}s – ${b.toFixed(2)}s`;
    const cur = existing ? existing.emo : 'HAP';
    $('#emoChoices').querySelectorAll('button').forEach((x) => { x.textContent = t(x.dataset.emo); x.classList.toggle('on', x.dataset.emo === cur); });
    $('#emoInt').value = existing ? existing.intensity : 0.8; $('#emoInt').oninput();
    $('#emoRemove').hidden = !existing;
    dlg.onclose = () => {
      if (dlg.returnValue === 'apply') {
        this.pushUndo();
        const emo = $('#emoChoices').querySelector('.on')?.dataset.emo || 'HAP';
        if (existing) { existing.emo = emo; existing.intensity = +$('#emoInt').value; }
        else sp.emoOverrides.push({ start: a, end: b, emo, intensity: +$('#emoInt').value });
        this.onEdit(sp);
      } else if (dlg.returnValue === 'remove' && existing) { this.pushUndo(); sp.emoOverrides = sp.emoOverrides.filter((o) => o !== existing); this.onEdit(sp); }
    };
    dlg.showModal();
  }

  // ---------- playback ----------
  buildPlayBuffer() {
    const sp = this.project.speakers; if (!sp.length) return null;
    const n = Math.max(...sp.map((s) => s.audio16k ? s.audio16k.length : 0));
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ac = this.ac || new AC();
    const buf = this.ac.createBuffer(1, n, SR); const d = buf.getChannelData(0);
    for (const s of sp) if (s.audio16k) for (let i = 0; i < s.audio16k.length; i++) d[i] += s.audio16k[i];
    return buf;
  }
  duration() { return Math.max(0, ...this.project.speakers.map((s) => s.T / FPS)); }
  togglePlay() { this.playing ? this.pause() : this.play(); }
  play(from = this.time, dest = null) {
    if (!this.project.speakers.length) return;
    if (from >= this.duration() - 0.05) from = 0;
    if (!this.playBuf) this.playBuf = this.buildPlayBuffer();
    this.ac.resume();
    this.src = this.ac.createBufferSource(); this.src.buffer = this.playBuf;
    this.src.connect(dest || this.ac.destination);
    if (dest) this.src.connect(this.ac.destination);
    this.src.start(0, from); this.startAt = this.ac.currentTime - from; this.playing = true;
    this.src.onended = () => { if (this.playing && this.ac.currentTime - this.startAt >= this.duration() - 0.1) { this.playing = false; $('#play').textContent = '▶'; this.onEnded && this.onEnded(); } };
    $('#play').textContent = '❚❚';
  }
  pause() { if (this.src) { this.src.onended = null; try { this.src.stop(); } catch (e) { /* already stopped */ } } this.playing = false; $('#play').textContent = '▶'; }
  seek(t2) { const was = this.playing; if (was) this.pause(); this.time = Math.min(t2, this.duration()); this.applyFrame(this.time); this.timeline.setTime(this.time); this.clock(); if (was) this.play(); }
  clock() { const m = Math.floor(this.time / 60), s = this.time - m * 60; $('#clock').textContent = `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`; }
  applyFrame(time) {
    const fi = time * FPS, i0 = Math.floor(fi), u = fi - i0;
    this.project.speakers.forEach((sp, i) => {
      if (!sp.final) return;
      const a = Math.min(sp.T - 1, i0), b = Math.min(sp.T - 1, i0 + 1);
      const f = new Float32Array(K); for (let k = 0; k < K; k++) f[k] = sp.final[a * K + k] * (1 - u) + sp.final[b * K + k] * u;
      this.stage.apply(i, this.composer.styled(f, sp.avatar));
    });
  }
  loop() {
    const step = (now) => {
      if (!$('#studio').hidden) {
        if (this.playing) { this.time = Math.min(this.duration(), this.ac.currentTime - this.startAt); this.applyFrame(this.time); this.timeline.setTime(this.time); this.clock(); }
        this.stage.render();
      } else if (this.liveStage) this.liveTick(now);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---------- export / save ----------
  download(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000); }
  async export(kind) {
    if (!this.project.speakers.length) return;
    if (kind === 'json') this.download(new Blob([JSON.stringify(exportEmo(this.project, this.composer))], { type: 'application/json' }), 'serifu-motion-emo.json');
    if (kind === 'csv') this.download(new Blob([exportCSV(this.project, this.composer)], { type: 'text/csv' }), 'serifu-motion-au.csv');
    if (kind === 'project') this.download(new Blob([JSON.stringify(this.saveProject())], { type: 'application/json' }), 'serifu-motion-project.json');
    if (kind === 'video') await this.recordVideo();
  }
  async recordVideo() {
    if (!this.playBuf) this.playBuf = this.buildPlayBuffer();
    const dest = this.ac.createMediaStreamDestination();
    const stream = new MediaStream([...this.stage.canvas.captureStream(30).getVideoTracks(), ...dest.stream.getAudioTracks()]);
    const mime = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm'].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6e6 } : undefined);
    const chunks = []; rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((r) => (rec.onstop = r));
    this.pause(); this.time = 0; rec.start(200); this.busy(t('recording') + '…');
    $('#busy').onclick = () => { this.pause(); rec.stop(); };
    this.onEnded = () => rec.stop();
    this.play(0, dest);
    await done; this.onEnded = null; this.busy(null); $('#busy').onclick = null;
    this.download(new Blob(chunks, { type: rec.mimeType || 'video/webm' }), 'serifu-motion.' + ((rec.mimeType || '').includes('mp4') ? 'mp4' : 'webm'));
  }
  saveProject() {
    const f32 = (a) => b64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
    const i16 = (a) => { const o = new Int16Array(a.length); for (let i = 0; i < a.length; i++) o[i] = Math.max(-32767, Math.min(32767, a[i] * 32767)); return b64(new Uint8Array(o.buffer)); };
    return { format: 'serifu-motion-project/1', speakers: this.project.speakers.map((s) => ({ name: s.name, avatar: s.avatar, color: s.color, expressiveness: s.expressiveness, visible: s.visible, edits: s.edits, emoOverrides: s.emoOverrides, recorded: !!s.recorded, T: s.T, audio: i16(s.audio16k), base: f32(s.base), emo: f32(s.emo) })) };
  }
  loadProject(p) {
    if (p.format !== 'serifu-motion-project/1') return;
    this.pushUndo(); this.pause();
    this.project.speakers = p.speakers.map((d, i) => {
      const sp = newSpeaker(i, d.name, d.avatar);
      Object.assign(sp, { color: d.color || sp.color, expressiveness: d.expressiveness ?? 1, visible: d.visible || sp.visible, edits: d.edits || {}, emoOverrides: d.emoOverrides || [], recorded: d.recorded, T: d.T });
      const ia = new Int16Array(unb64(d.audio).buffer); sp.audio16k = Float32Array.from(ia, (v) => v / 32767);
      sp.base = new Float32Array(unb64(d.base).buffer); sp.emo = new Float32Array(unb64(d.emo).buffer);
      this.recompose(sp); return sp;
    });
    this.time = 0; this.afterChange(); this.timeline.setTime(0);
  }
  async loadDemo() {
    this.busy(t('analyzing'));
    try { const p = await fetch('demo/demo.json').then((r) => r.json()); this.loadProject(p); } finally { this.busy(null); }
  }

  // ---------- live ----------
  bindLive() {
    const sel = $('#liveAvatar'); for (const k of AVATARS) sel.add(new Option(DESIGNS[k].label[getLang()] || DESIGNS[k].label.en, k));
    sel.onchange = () => this.liveStage && this.liveStage.setCharacters([sel.value]);
    $('#liveStart').onclick = () => (this.live ? this.stopLive() : this.startLive());
    $('#liveRec').onclick = () => this.toggleRec();
    $('#camMouth').onchange = (e) => this.live && (this.live.useCamMouth = e.target.checked);
    $('#camBlink').onchange = (e) => this.live && (this.live.useCamBlink = e.target.checked);
  }
  async startLive() {
    try {
      this.live = new LiveSession({ rtModel: this.rtModel, video: $('#camPreview'), onStatus: () => {} });
      this.live.useCamMouth = $('#camMouth').checked; this.live.useCamBlink = $('#camBlink').checked;
      await this.live.start();
      $('#liveStart').textContent = t('stopLive'); $('#liveRec').disabled = false;
    } catch (e) { alert(e.message || e); this.live = null; }
  }
  stopLive() { if (this.recLive) this.toggleRec(); this.live?.stop(); this.live = null; $('#liveStart').textContent = t('startLive'); $('#liveRec').disabled = true; $('#emoMeter').innerHTML = ''; $('#faceStatus').textContent = ''; }
  toggleRec() {
    if (!this.live) return;
    if (!this.recLive) { this.live.startRecording(); this.recLive = true; $('#liveRec').textContent = t('stopRecord'); $('#recBadge').hidden = false; $('#recBadge').textContent = '● ' + t('recording'); return; }
    this.recLive = false; $('#liveRec').textContent = t('record'); $('#recBadge').hidden = true;
    const r = this.live.stopRecording(); if (!r || r.T < 10) return;
    this.pushUndo();
    const i = this.project.speakers.length;
    const sp = newSpeaker(i, `Live ${i + 1}`, $('#liveAvatar').value);
    Object.assign(sp, { audio16k: r.audio16k, T: r.T, base: r.base, emo: r.emo, recorded: true });
    this.recompose(sp); this.project.speakers.push(sp); this.padSpeakers(); this.selected = sp;
    this.stopLive(); this.tab('studio'); this.afterChange();
  }
  liveTick(now) {
    if (this.live) {
      const f = this.live.tick(now);
      if (f) {
        const sp = { base: f.y, T: 1, emo: null, edits: {}, emoOverrides: [], expressiveness: 1 };
        const out = this.composer.frame(sp, 0);
        this.liveStage.apply(0, this.composer.styled(out, $('#liveAvatar').value));
        const m = $('#emoMeter'); if (!this._meterT || now - this._meterT > 150) { this._meterT = now; m.innerHTML = EMOS.map((e, j) => `<div><span style="width:52px">${t(e)}</span><i style="width:${Math.round(f.emo[j] * 160)}px;background:${EMO_COLORS[e]}"></i></div>`).join(''); }
        $('#faceStatus').textContent = this.live.cam ? (this.live.faceSeen ? t('faceFound') : t('noFace')) : '';
      }
    }
    this.liveStage.render();
  }
}
function b64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function unb64(s) { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }

window.app = new App();
window.app.init().catch((e) => { console.error(e); document.getElementById('busyText').textContent = 'Error: ' + e.message; });
