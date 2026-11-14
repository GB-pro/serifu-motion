// Serifu Motion — application shell
import { EmoGenerator } from './core/generator.js';
import { IDS, K, EMOS, CHANNELS, GROUPS, frameToObj } from './core/au.js';
import { SR, FPS } from './core/features.js';
const SPEECH_LANGS = { ja: '日本語', en: 'English', zh: '中文', ko: '한국어', es: 'Español', fr: 'Français', de: 'Deutsch', it: 'Italiano', pt: 'Português', ru: 'Русский', id: 'Bahasa Indonesia', th: 'ไทย', vi: 'Tiếng Việt', hi: 'हिन्दी', ar: 'العربية' };
import { trackVideoFace, matchFaces } from './core/facetrack.js';
import { SpeechToText, speechLines } from './core/transcribe.js';
import { understandRemote, understandLocal, toOverrides, DEFAULT_MODEL, LIVE_MODEL } from './core/understand.js';
import { newSpeaker, analyzeSpeaker, readSpeaker, directSpeaker, isDirected, upgradeTrack, diarize, Composer, insertGesture, exportEmo, exportCSV, GESTURES, STYLES } from './core/project.js';
import { DESIGNS } from './avatar/characters.js';
import { Stage } from './ui/stage.js';
import { Timeline, EMO_COLORS } from './ui/timeline.js';
import { LANGS, detectLang, setLang, t, chName, getLang } from './ui/i18n.js';
import { LiveSession } from './rt/live.js';

const $ = (s) => document.querySelector(s);
const AVATARS = Object.keys(DESIGNS);
const isVideo = (f) => /video/.test(f.type) || /\.(mp4|mov|webm|m4v)$/i.test(f.name);
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
    const J = (u) => fetch(u).then((r) => r.json());
    // three offline networks read the speech as a committee (live: one or two realtime networks); the acting profiles give each avatar its own way of moving
    const [off, rt, acting, rtB, ...extra] = await Promise.all([J('models/emo_offline.json'), J('models/emo_realtime.json'), J('avatar/acting.json'), J('models/emo_realtime_b.json').catch(() => null), J('models/emo_offline_b.json').catch(() => null), J('models/emo_offline_c.json').catch(() => null)]);
    this.rtModel = rt; this.rtExtra = [rtB].filter(Boolean); this.acting = acting;
    this.gen = new EmoGenerator(off, rt, extra.filter(Boolean));
    this.composer = new Composer(this.gen.meta);
    $('#modelInfo').textContent = `${t('model')}: ${this.gen.meta.dataset} · ${this.gen.meta.trained_clips} clips`;
    this.stage = new Stage($('#stage'), { preserve: true });
    this.timeline = new Timeline($('#timeline'), this);
    this.bindUI(); this.bindAI();
    this.busy(null);
    this.renderPanel();
    this.loop();
    if (new URLSearchParams(location.search).has('demo')) this.loadDemo();
  }
  i18n() {
    document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
    if (this.timeline) { this.renderPanel(); this.timeline.draw(); }
  }
  busy(msg) { $('#busy').hidden = !msg; $('#busyText').textContent = msg || ''; }

  bindUI() {
    document.querySelectorAll('.tabs button').forEach((b) => b.onclick = () => this.tab(b.dataset.tab));
    $('#play').onclick = () => this.togglePlay();
    // files wait in a list until the reading is set up and started with the button
    $('#fileSep').onchange = (e) => { this.addPending([...e.target.files]); e.target.value = ''; };
    document.querySelectorAll('input[name=loadMode]').forEach((r) => (r.onchange = () => this.renderPending()));
    $('#startBtn').onclick = () => this.startAnalysis();
    this.pending = []; this.renderPending();
    $('#fileProj').onchange = async (e) => { const f = e.target.files[0]; if (f) await this.loadProject(JSON.parse(await f.text())); e.target.value = ''; };
    $('#demoBtn').onclick = () => this.loadDemo();
    $('#undo').onclick = () => this.undo(); $('#redo').onclick = () => this.redo();
    $('#zin').onclick = () => this.timeline.zoom(1.3); $('#zout').onclick = () => this.timeline.zoom(1 / 1.3);
    $('#exportBtn').onclick = () => { $('#exportMenu').hidden = !$('#exportMenu').hidden; };
    document.querySelectorAll('[data-exp]').forEach((b) => b.onclick = () => { $('#exportMenu').hidden = true; this.export(b.dataset.exp); });
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.code === 'Space') { e.preventDefault(); this.togglePlay(); }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); }
      if (e.ctrlKey || e.metaKey || e.altKey || $('#studio').hidden || document.querySelector('dialog[open]')) return;
      // the acting track from the keyboard: 1-6 set the feeling of the selected band, left / right move along the bands
      const sel = this.timeline.sel;
      if (/^[1-6]$/.test(e.key) && sel) { e.preventDefault(); this.setBand(sel, EMOS[+e.key - 1]); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); this.timeline.step(e.key === 'ArrowRight' ? 1 : -1); }
      else if (e.key === 'Escape') this.timeline.select(null);
      else if ((e.key === 'Delete' || e.key === 'Backspace') && sel) { const o = this.markAt(sel.sp, (sel.a + sel.b) / 2); if (o) { e.preventDefault(); this.pushUndo(); sel.sp.emoOverrides = sel.sp.emoOverrides.filter((q) => q !== o); this.redirect(sel.sp); this.onEdit(sel.sp); } }
      else if (e.key === 'c' || e.key === 'C') this.setTool(this.timeline.tool === 'cut' ? 'select' : 'cut');
    });
    const stageEl = $('#stagewrap') || $('.stagewrap');
    stageEl.addEventListener('dragover', (e) => e.preventDefault());
    stageEl.addEventListener('drop', (e) => { e.preventDefault(); this.addPending([...e.dataTransfer.files]); });
    $('#timeline').addEventListener('dragover', (e) => e.preventDefault());
    $('#timeline').addEventListener('drop', (e) => { e.preventDefault(); this.addPending([...e.dataTransfer.files]); });
    // emotion dialog
    const ch = $('#emoChoices');
    for (const e of EMOS) { const b = document.createElement('button'); b.type = 'button'; b.dataset.emo = e; b.style.background = EMO_COLORS[e]; b.onclick = () => { ch.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); }; ch.appendChild(b); }
    // the properties of the selected band (non-modal; the same choices as the dialog, with their number keys)
    const be = $('#bandEmos');
    EMOS.forEach((e, i) => { const b = document.createElement('button'); b.type = 'button'; b.dataset.emo = e; b.style.background = EMO_COLORS[e]; b.onclick = () => this.timeline.sel && this.setBand(this.timeline.sel, e); be.appendChild(b); });
    $('#bandInt').oninput = () => { $('#bandIntV').textContent = Math.round($('#bandInt').value * 100) + '%'; };
    $('#bandInt').onchange = () => { const s = this.timeline.sel; if (s) this.setBand(s, s.emo, +$('#bandInt').value); };
    $('#bandInt').onpointerdown = () => {};
    $('#vzin').onclick = () => this.timeline.vzoom(1.25); $('#vzout').onclick = () => this.timeline.vzoom(1 / 1.25);
    $('#toolSel').onclick = () => this.setTool('select'); $('#toolCut').onclick = () => this.setTool('cut');
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
  // ---------- files waiting to be read ----------
  addPending(files) {
    files = files.filter((f) => /audio|video/.test(f.type) || /\.(wav|mp3|m4a|aac|ogg|flac|mp4|mov|webm|m4v)$/i.test(f.name));
    if (!files.length) return;
    this.pending.push(...files); this.renderPending();
  }
  renderPending() {
    const ul = $('#pendList'); ul.innerHTML = '';
    if (!this.pending.length) { const li = document.createElement('li'); li.className = 'none'; li.textContent = t('pendNone'); ul.appendChild(li); }
    this.pending.forEach((f, i) => { const li = document.createElement('li'), n = document.createElement('span'), x = document.createElement('button');
      n.textContent = (isVideo(f) ? '🎬 ' : '🎵 ') + f.name; n.title = f.name; x.textContent = '✕'; x.title = 'remove';
      x.onclick = () => { this.pending.splice(i, 1); this.renderPending(); }; li.append(n, x); ul.appendChild(li); });
    const mode = (document.querySelector('input[name=loadMode]:checked') || {}).value;
    $('#nSpkBox').hidden = mode !== 'mix';
    $('#startBtn').disabled = !this.pending.length || !mode || this.analyzing;
    $('#startBtn').title = mode ? '' : t('chooseMode');
  }
  async startAnalysis() {
    const mode = (document.querySelector('input[name=loadMode]:checked') || {}).value;
    if (!this.pending.length || !mode || this.analyzing) return;
    const files = this.pending.splice(0); this.analyzing = true; this.renderPending();
    try {
      if (mode === 'sep') await this.addSeparate(files);
      else for (const f of files) await this.addMixed(f, +$('#nSpk').value || 2);
    } finally { this.analyzing = false; this.renderPending(); }
  }
  // how long the import took, kept on screen until the next one
  loadInfo(stage, extra = '') {
    const el = $('#faceInfo'), s = (ms) => Math.max(1, Math.round(ms / 1000));
    if (stage === 'start') { this.loadT = { t0: performance.now() }; el.hidden = true; return; }
    const L = this.loadT; if (!L) return;
    if (stage === 'voice') L.voice = performance.now() - L.t0;
    if (stage === 'face') L.face = performance.now() - L.t0 - L.voice;
    const parts = [`${t('loadVoice')} ${s(L.voice)} ${t('sec')}`];
    if (L.face != null) parts.push(`${t('loadFace')} ${s(L.face)} ${t('sec')}`);
    el.textContent = (L.face != null || stage === 'done' ? `${t('loadDone')} ${s((L.voice || 0) + (L.face || 0))} ${t('sec')}（${parts.join(' + ')}）` : `${parts[0]}。${extra}`);
    el.hidden = false;
  }
  async addSeparate(files) {
    files = files.filter((f) => /audio|video/.test(f.type) || /\.(wav|mp3|m4a|aac|ogg|flac|mp4|mov|webm)$/i.test(f.name));
    if (!files.length) return;
    this.pushUndo(); this.loadInfo('start');
    for (const f of files) {
      this.busy(t('analyzing') + ' ' + f.name);
      await sleep(30);
      const x = await this.decode(f); this.setReference(f);
      const i = this.project.speakers.length;
      const sp = newSpeaker(i, f.name.replace(/\.[^.]+$/, '').slice(0, 24), AVATARS[i % AVATARS.length]);
      sp.audio16k = x;
      analyzeSpeaker(sp);
      await this.process(sp);
      this.project.speakers.push(sp);
      // a video of the speaker: its face is read in the background and applied when done (see watchFaces)
      if (isVideo(f)) this.watchFaces(f, sp.T, [sp], null, 1);
    }
    this.padSpeakers();
    this.afterChange(); this.busy(null);
    this.loadInfo(files.some(isVideo) ? 'voice' : 'voice', files.some(isVideo) ? t('faceTracking') : ''); if (!files.some(isVideo)) this.loadInfo('done');
    await this.understand(this.project.speakers.slice(-files.length), true);
  }
  async addMixed(file, k) {
    if (!file) return;
    this.pushUndo(); this.loadInfo('start');
    this.busy(t('analyzing')); await sleep(30);
    const x = await this.decode(file); this.setReference(file);
    // who speaks when: the voiceprint network in the worker; the older split by pitch and timbre if it cannot run
    let segments;
    try {
      this.stt = this.stt || new SpeechToText();
      segments = await this.stt.speakers(x, k, (m) => this.busy(m.what === 'speakers' ? `${t('spkLoading')} ${Math.round(m.loaded / 1e6)} / ${Math.round(m.total / 1e6)} MB` : `${t('spkSplitting')} ${Math.round(m.u * 100)}%`));
    } catch (e) { console.warn('speaker split fell back:', e); segments = diarize(x, k).segments; }
    const n0 = this.project.speakers.length;
    for (let j = 0; j < k; j++) {
      const a = k === 1 ? x.slice() : new Float32Array(x.length);
      if (k > 1) for (const s of segments) if (s.speaker === j) {
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
    this.loadInfo('voice', isVideo(file) ? t('faceTracking') : ''); if (!isVideo(file)) this.loadInfo('done');
    // a video of the conversation: each person's face is read in the background, and each voice gets the face whose
    // mouth moves while that voice speaks
    if (isVideo(file)) this.watchFaces(file, Math.max(1, Math.floor(x.length / SR * FPS)), this.project.speakers.slice(n0, n0 + k), segments, k);
    await this.understand(this.project.speakers.slice(n0), true);
  }
  // Reading the faces in a video takes a while (the face tracker runs on every picture), so it does not hold up the
  // work: the speakers are made from the voice first, and the faces are applied when they are ready. One video at a time.
  watchFaces(file, T, sps, segments, k) {
    const show = (msg) => { if (msg === null) return; this.loadInfo('progress', msg); };
    this.faceQueue = (this.faceQueue || Promise.resolve()).then(async () => {
      const t0 = performance.now();
      show(t('faceTracking')); this.faceBusy = true;
      try {
        const res = await trackVideoFace(file, T, '', (u) => show(`${t('faceTracking')} ${Math.round(u * 100)}%`), k);
        const tracks = k === 1 ? (res ? [res] : []) : res || [];
        const faceOf = k === 1 ? [tracks[0] || null] : (tracks.length ? matchFaces(tracks, segments, k, T).map((i) => (i >= 0 ? tracks[i] : null)) : []);
        console.info('faces', tracks.map((q) => q.x.toFixed(2)), 'took', ((performance.now() - t0) / 1000).toFixed(1), 's', trackVideoFace.last);
        let any = false;
        sps.forEach((sp, j) => { if (faceOf[j] && this.project.speakers.includes(sp)) { sp.face = faceOf[j]; this.redirect(sp); any = true; } });
        if (any) { this.timeline.draw(); this.applyFrame(this.time); }
        this.loadInfo('face');
      } catch (e) { console.warn('face tracking', e); this.loadInfo('done'); }
      this.faceBusy = false;
    });
    return this.faceQueue;
  }
  // A file that carries video: the video is shown small over the stage as a reference and follows the playhead.
  // It is only looked at: its picture is not copied into the project.
  setReference(file) {
    if (!file || !(/^video\//.test(file.type) || /\.(mp4|mov|webm|m4v)$/i.test(file.name))) return;
    const v = $('#refVideo'); if (this.refUrl) URL.revokeObjectURL(this.refUrl);
    this.refUrl = URL.createObjectURL(file); v.src = this.refUrl; v.muted = true; v.playsInline = true;
    v.onloadedmetadata = () => { if (!v.videoWidth) { this.clearReference(); return; } $('#refBox').hidden = false; $('#refShow').hidden = true; this.syncReference(true); }; // a file with sound only has no picture
    v.onerror = () => this.clearReference();
    $('#refHide').onclick = () => { $('#refBox').hidden = true; $('#refShow').hidden = false; };
    $('#refShow').onclick = () => { $('#refBox').hidden = false; $('#refShow').hidden = true; this.syncReference(true); };
  }
  clearReference() { const v = $('#refVideo'); v.pause(); v.removeAttribute('src'); v.load(); if (this.refUrl) URL.revokeObjectURL(this.refUrl); this.refUrl = null; $('#refBox').hidden = true; $('#refShow').hidden = true; }
  syncReference(force) {
    const v = $('#refVideo'); if (!this.refUrl || !v.videoWidth || $('#refBox').hidden) return;
    const t2 = Math.min(this.time, (v.duration || 1e9) - 0.05);
    if (this.playing) { if (v.paused) { v.currentTime = t2; v.play().catch(() => {}); } else if (Math.abs(v.currentTime - t2) > 0.25) v.currentTime = t2; }
    else { if (!v.paused) v.pause(); if (force || Math.abs(v.currentTime - t2) > 0.02) v.currentTime = t2; }
  }
  // ---------- what is said: speech to text, then the meaning of the lines decides the feelings ----------
  aiSettings() { const g = (k) => { try { return localStorage.getItem(k) || ''; } catch { return ''; } }; return { on: g('sm.ai.on') !== '0', key: g('sm.ai.key'), model: g('sm.ai.model') || DEFAULT_MODEL, lang: g('sm.ai.lang') }; }
  bindAI() {
    const dlg = $('#aiDlg'), put = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch { /* private mode: kept for this visit only */ } };
    $('#aiBtn').onclick = () => { const s = this.aiSettings(); $('#aiOn').checked = s.on; $('#aiKey').value = s.key; $('#aiModel').value = s.model; $('#aiLang').value = s.lang; dlg.showModal(); };
    { const sl = $('#aiLang'); sl.add(new Option(t('langAuto'), '')); for (const [k, v] of Object.entries(SPEECH_LANGS)) sl.add(new Option(v, k)); }
    dlg.addEventListener('close', () => {
      if (dlg.returnValue !== 'apply') return;
      put('sm.ai.on', $('#aiOn').checked ? '' : '0'); put('sm.ai.key', $('#aiKey').value.trim()); put('sm.ai.lang', $('#aiLang').value); put('sm.ai.model', $('#aiModel').value.trim() === DEFAULT_MODEL ? '' : $('#aiModel').value.trim());
    });
    $('#readBtn').onclick = () => this.understand(this.project.speakers, false);
  }
  note(msg, warn = false) { const el = $('#aiInfo'); if (msg && this.project.speakers.some((sp) => sp.face)) msg += ' ' + t('faceUsed') + '。'; el.textContent = msg || ''; el.classList.toggle('warn', !!warn); el.hidden = !msg; }
  // Reads the lines of these speakers (auto: after an import, only when the setting is on) and re-directs them.
  async understand(speakers, auto) {
    const set = this.aiSettings(); speakers = speakers.filter((sp) => !sp.recorded && sp.audio16k && sp.features);
    if (!speakers.length || (auto && !set.on) || this.reading) return;
    this.reading = true;
    try {
      this.stt = this.stt || new SpeechToText();
      this.busy(t('sttLoading')); await this.stt.load((a, b) => this.busy(`${t('sttLoading')} ${Math.round(a / 1e6)} / ${Math.round(b / 1e6)} MB`));
      const jobs = speakers.map((sp) => ({ sp, lines: speechLines(sp) })), total = jobs.reduce((s, j) => s + j.lines.length, 0); let k = 0;
      for (const j of jobs) {
        this.stt.forget(); j.sp.script = [];
        for (const l of j.lines) { this.busy(`${t('sttLine')} (${++k}/${total})`); const r = await this.stt.line(l.samples, set.lang || undefined); if (r.text) j.sp.script.push({ start: l.start, end: l.end, text: r.text, lang: r.lang, conf: +r.conf.toFixed(2) }); }
      }
      // the whole dialogue in the order it is spoken, so each line is understood in the light of the others
      const all = this.project.speakers.flatMap((sp) => (sp.script || []).map((l) => ({ ...l, sp, speaker: sp.name }))).sort((a, b) => a.start - b.start);
      if (!all.length) { this.note(t('noWords'), true); return; }
      let parts = null, why = '';
      if (set.key) {
        this.busy(t('understanding'));
        try { parts = await understandRemote(all, { key: set.key, model: set.model }); } catch (e) { why = `${t('aiFailed')} (${t('aiErr_' + e.code) || e.code}${e.code === 'http' || e.code === 'answer' ? ': ' + e.message : ''})`; console.warn('understanding', e); }
      }
      const remote = !!parts; if (!parts) parts = understandLocal(all);
      this.pushUndo();
      for (const sp of this.project.speakers) {
        const idx = all.map((l, i) => (l.sp === sp ? i : -1)).filter((i) => i >= 0); if (!idx.length) continue;
        sp.emoOverrides = toOverrides(idx.map((i) => all[i]), idx.map((i) => parts[i]), sp.emoOverrides); this.redirect(sp);
      }
      this.afterChange();
      this.note(remote ? t('readByAI') : `${t('readOnDevice')}${why ? ' ' + why : set.key ? '' : ' ' + t('noKeyHint')}`, !remote);
    } catch (e) { console.error(e); this.note(`${t('sttFailed')} (${e.message})`, true); }
    finally { this.reading = false; this.busy(null); }
  }
  async process(sp) {
    analyzeSpeaker(sp); await sleep(10);
    readSpeaker(sp, this.gen); await sleep(10);
    this.redirect(sp);
  }
  // rebuild the performance from the stored reading (after a change of emotion, avatar, style or strength)
  redirect(sp) {
    if (!sp.recorded && sp.audio16k) directSpeaker(sp, this.gen, this.acting);
    this.recompose(sp);
  }
  padSpeakers() { // make all speaker tracks the same length (silence after shorter ones) so the dialogue stays aligned
    const T = Math.max(...this.project.speakers.map((s) => s.T));
    for (const sp of this.project.speakers) if (sp.T < T && sp.audio16k) {
      const a = new Float32Array(Math.ceil(T / FPS * SR)); a.set(sp.audio16k); sp.audio16k = a;
      if (sp.base && !sp.recorded) { analyzeSpeaker(sp); readSpeaker(sp, this.gen); directSpeaker(sp, this.gen, this.acting); }
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
    this.playBuf = null; if (this.timeline) this.timeline.reselect();
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
      av.onchange = () => { this.pushUndo(); sp.avatar = av.value; this.stage.setCharacters(this.project.speakers.map((s) => s.avatar)); this.redirect(sp); this.timeline.draw(); this.applyFrame(this.time); };
      d.append(name, av); d.onclick = (e) => { if (e.target === d || e.target.className === 'dot') this.selectSpeaker(sp); };
      name.onfocus = () => { if (this.selected !== sp) this.selectSpeaker(sp); };
      list.appendChild(d);
    }
    const det = $('#spDetail'); det.innerHTML = '';
    const sp = this.selected; if (!sp) return;
    det.className = 'detail';
    const ex = document.createElement('label'); ex.className = 'mini'; ex.innerHTML = `${t('expressiveness')} <input type="range" min="0.3" max="2" step="0.05" value="${sp.expressiveness}"> <output>${Math.round(sp.expressiveness * 100)}%</output>`;
    const exIn = ex.querySelector('input'); exIn.oninput = () => { sp.expressiveness = +exIn.value; ex.querySelector('output').textContent = Math.round(sp.expressiveness * 100) + '%'; this.redirect(sp); this.timeline.draw(); this.applyFrame(this.time); };
    exIn.onpointerdown = () => this.pushUndo();
    det.appendChild(ex);
    // acting style: the animation layer (default) or the plain linear rendition of the learned motion
    if (!sp.recorded) {
      const st = document.createElement('label'); st.className = 'mini'; st.append(document.createTextNode(t('style') + ' '));
      const ss = document.createElement('select'); for (const k of STYLES) ss.add(new Option(t('style_' + k), k)); ss.value = sp.style;
      ss.onchange = () => { this.pushUndo(); sp.style = ss.value; this.redirect(sp); this.timeline.draw(); this.applyFrame(this.time); };
      st.appendChild(ss); det.appendChild(st);
    }
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
    const clr = document.createElement('button'); clr.className = 'ghost'; clr.textContent = t('clearEdits'); clr.onclick = () => { this.pushUndo(); sp.edits = {}; sp.emoOverrides = []; this.redirect(sp); this.onEdit(sp); };
    const regen = document.createElement('button'); regen.className = 'ghost'; regen.textContent = t('regenerate'); regen.disabled = !!sp.recorded;
    regen.title = t('regenerateHint');
    regen.onclick = () => { this.pushUndo(); sp.seed = (Math.random() * 1e6) | 0; this.redirect(sp); this.onEdit(sp); };
    const del = document.createElement('button'); del.className = 'ghost danger'; del.textContent = t('delSpeaker'); del.onclick = () => { this.pushUndo(); this.project.speakers = this.project.speakers.filter((s) => s !== sp); this.afterChange(); };
    row.append(clr, regen, del); det.appendChild(row);
  }

  // ---------- editing ----------
  snapshot() { return this.project.speakers.map((s) => ({ s, edits: JSON.parse(JSON.stringify(s.edits)), ov: JSON.parse(JSON.stringify(s.emoOverrides)), ex: s.expressiveness, base: s.base, emo: s.emo, plan: s.plan, avatar: s.avatar, style: s.style, seed: s.seed })); }
  pushUndo() { this.undoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); if (this.undoStack.length > 80) this.undoStack.shift(); this.redoStack = []; }
  restore(st) { this.project.speakers = st.speakers.slice(); for (const x of st.snap) { x.s.edits = x.edits; x.s.emoOverrides = x.ov; x.s.expressiveness = x.ex; x.s.base = x.base; x.s.emo = x.emo; x.s.plan = x.plan; x.s.avatar = x.avatar; x.s.style = x.style; x.s.seed = x.seed; this.recompose(x.s); } this.afterChange(); }
  undo() { const st = this.undoStack.pop(); if (!st) return; this.redoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); this.restore(st); }
  redo() { const st = this.redoStack.pop(); if (!st) return; this.undoStack.push({ speakers: this.project.speakers.slice(), snap: this.snapshot() }); this.restore(st); }
  onEdit(sp) { this.recompose(sp); this.timeline.draw(); this.applyFrame(this.time); }
  setTool(tool) { this.timeline.tool = tool; $('#toolSel').classList.toggle('on', tool === 'select'); $('#toolCut').classList.toggle('on', tool === 'cut'); }
  // the emotion mark that covers time t of a speaker, if any
  markAt(sp, t) { return sp.emoOverrides.find((o) => t >= o.start && t <= o.end) || null; }
  onSelectBand(sel) {
    const box = $('#bandProps'); box.hidden = !sel; if (!sel) return;
    $('#bandRange').textContent = `${sel.sp.name} · ${sel.a.toFixed(2)}s – ${sel.b.toFixed(2)}s`;
    $('#bandEmos').querySelectorAll('button').forEach((x, i) => { x.innerHTML = `<span class="k">${i + 1}</span>${t(x.dataset.emo)}`; x.classList.toggle('on', x.dataset.emo === sel.emo); });
    $('#bandInt').value = sel.inten; $('#bandInt').oninput();
    if (this.selected !== sel.sp) { this.selected = sel.sp; }
  }
  // give a band a feeling (and strength): the mark covering it is changed, or a new one is laid over the band
  setBand(sel, emo, inten = null) {
    const sp = sel.sp, mid = (sel.a + sel.b) / 2; this.pushUndo();
    const o = this.markAt(sp, mid), I = inten ?? (o ? o.intensity : sel.inten ?? 0.8);
    if (o) { o.emo = emo; o.intensity = I; delete o.ai; }
    else sp.emoOverrides.push({ start: +(sel.a - (isDirected(sp) ? 0.05 : 0)).toFixed(3), end: +(sel.b + (isDirected(sp) ? 0.05 : 0)).toFixed(3), emo, intensity: I });
    this.redirect(sp); this.onEdit(sp); this.timeline.reselect(); this.timeline.draw();
  }
  // the scissors: the band under time t becomes two marks with the same feeling, cut at t
  cutBand(sp, t) {
    const band = this.timeline.bandAt(sp, t); if (!band || t - band.a < 0.5 || band.b - t < 0.5) return; // each half must be long enough to act
    this.pushUndo();
    const o = this.markAt(sp, t), pad = isDirected(sp) ? 0.05 : 0;
    const a = o ? o.start : band.a - pad, b = o ? o.end : band.b + pad, emo = o ? o.emo : band.emo, I = o ? o.intensity : band.inten;
    if (o) sp.emoOverrides = sp.emoOverrides.filter((q) => q !== o);
    sp.emoOverrides.push({ start: +a.toFixed(3), end: +t.toFixed(3), emo, intensity: I, cut: true }, { start: +t.toFixed(3), end: +b.toFixed(3), emo, intensity: I, cut: true });
    this.redirect(sp); this.onEdit(sp);
    const left = this.timeline.bandAt(sp, t - 0.1); this.timeline.select(left ? { sp, ...left } : null);
  }
  openEmoDialog(sp, a, b, existing, suggested = null, suggestedInt = null) {
    const dlg = $('#emoDlg');
    $('#emoRange').textContent = `${sp.name} · ${a.toFixed(2)}s – ${b.toFixed(2)}s`;
    const cur = existing ? existing.emo : (suggested || 'HAP');
    $('#emoChoices').querySelectorAll('button').forEach((x) => { x.textContent = t(x.dataset.emo); x.classList.toggle('on', x.dataset.emo === cur); });
    $('#emoInt').value = existing ? existing.intensity : (suggestedInt ?? 0.8); $('#emoInt').oninput();
    $('#emoRemove').hidden = !existing;
    dlg.onclose = () => {
      if (dlg.returnValue === 'apply') {
        this.pushUndo();
        const emo = $('#emoChoices').querySelector('.on')?.dataset.emo || 'HAP';
        if (existing) { existing.emo = emo; existing.intensity = +$('#emoInt').value; }
        else sp.emoOverrides.push({ start: a, end: b, emo, intensity: +$('#emoInt').value });
        this.redirect(sp); this.onEdit(sp);
      } else if (dlg.returnValue === 'remove' && existing) { this.pushUndo(); sp.emoOverrides = sp.emoOverrides.filter((o) => o !== existing); this.redirect(sp); this.onEdit(sp); }
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
  pause() { const rv = $('#refVideo'); if (rv && !rv.paused) rv.pause(); if (this.src) { this.src.onended = null; try { this.src.stop(); } catch (e) { /* already stopped */ } } this.playing = false; $('#play').textContent = '▶'; }
  seek(t2) { const was = this.playing; if (was) this.pause(); this.time = Math.min(t2, this.duration()); this.applyFrame(this.time); this.timeline.setTime(this.time); this.clock(); if (was) this.play(); }
  clock() { const m = Math.floor(this.time / 60), s = this.time - m * 60; $('#clock').textContent = `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`; }
  applyFrame(time) {
    this.syncReference();
    const fi = time * FPS, i0 = Math.floor(fi), u = fi - i0;
    this.project.speakers.forEach((sp, i) => {
      if (!sp.final) return;
      const a = Math.min(sp.T - 1, i0), b = Math.min(sp.T - 1, i0 + 1);
      const f = new Float32Array(K); for (let k = 0; k < K; k++) f[k] = sp.final[a * K + k] * (1 - u) + sp.final[b * K + k] * u;
      this.stage.apply(i, frameToObj(f), this.playing ? this._dt || 0 : 0); // hair and accessories follow through only while playing
    });
  }
  loop() {
    let last = 0;
    const step = (now) => {
      this._dt = Math.min(0.1, last ? (now - last) / 1000 : 0); last = now;
      if (!$('#studio').hidden) {
        if (this.playing) { this.time = Math.min(this.duration(), this.ac.currentTime - this.startAt); this.applyFrame(this.time); this.timeline.setTime(this.time); this.clock(); }
        // while a video's faces are being read and nothing plays, the stage is drawn less often: the face tracker and
        // the drawing share the computer, and the reading finishes much sooner this way
        if (this.playing || !this.faceBusy || now - (this._lastDraw || 0) > 500) { this.stage.render(); this._lastDraw = now; }
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
    // generated takes are rebuilt from their audio when the project is opened (same seed, same result); only live
    // recordings carry their frames, because the camera's blinks and mouth cannot be regenerated
    return { format: 'serifu-motion-project/2', speakers: this.project.speakers.map((s) => ({ name: s.name, avatar: s.avatar, color: s.color, expressiveness: s.expressiveness, style: s.style, seed: s.seed, visible: s.visible, edits: s.edits, emoOverrides: s.emoOverrides, script: s.script || [], recorded: !!s.recorded, T: s.T, audio: i16(s.audio16k), ...(s.recorded ? { base: f32(s.base), emo: f32(s.emo) } : {}) })) };
  }
  async loadProject(p) {
    const v1 = p.format === 'serifu-motion-project/1';
    if (!v1 && p.format !== 'serifu-motion-project/2') return;
    this.pushUndo(); this.pause();
    const speakers = [];
    for (let i = 0; i < p.speakers.length; i++) {
      const d = p.speakers[i], sp = newSpeaker(i, d.name, d.avatar);
      Object.assign(sp, { color: d.color || sp.color, expressiveness: d.expressiveness ?? 1, style: d.style || 'anime', seed: d.seed ?? sp.seed, visible: (d.visible || sp.visible).filter((id) => IDS.includes(id)), edits: d.edits || {}, emoOverrides: d.emoOverrides || [], script: d.script || [], recorded: !!d.recorded, T: d.T });
      const ia = new Int16Array(unb64(d.audio).buffer); sp.audio16k = Float32Array.from(ia, (v) => v / 32767);
      if (sp.recorded && d.base) { sp.base = upgradeTrack(new Float32Array(unb64(d.base).buffer), d.T, this.gen.meta); sp.emo = new Float32Array(unb64(d.emo).buffer); this.recompose(sp); }
      else { sp.recorded = false; this.busy(t('generating') + ` (${i + 1}/${p.speakers.length})`); await sleep(20); await this.process(sp); }
      speakers.push(sp);
    }
    this.project.speakers = speakers; this.busy(null);
    this.time = 0; this.afterChange(); this.timeline.setTime(0);
  }
  async loadDemo() {
    this.busy(t('analyzing'));
    try { const p = await fetch('demo/demo.json').then((r) => r.json()); await this.loadProject(p); } finally { this.busy(null); }
  }

  // ---------- live ----------
  bindLive() {
    const sel = $('#liveAvatar'); for (const k of AVATARS) sel.add(new Option(DESIGNS[k].label[getLang()] || DESIGNS[k].label.en, k));
    sel.onchange = () => { if (this.liveStage) this.liveStage.setCharacters([sel.value]); if (this.live) this.live.setProfile(this.acting[sel.value]); };
    const ls = $('#liveStyle'); if (ls) { for (const k of STYLES) ls.add(new Option(t('style_' + k), k)); ls.onchange = () => this.live && this.live.setStyle(ls.value); }
    $('#liveStart').onclick = () => (this.live ? this.stopLive() : this.startLive());
    $('#liveRec').onclick = () => this.toggleRec();
    $('#camMouth').onchange = (e) => this.live && (this.live.useCamMouth = e.target.checked);
    $('#camBlink').onchange = (e) => this.live && (this.live.useCamBlink = e.target.checked);
  }
  async startLive() {
    try {
      // with "meaning" on, the character plays a few seconds late and the words spoken in the meantime decide the feeling
      let meaning = null; const set = this.aiSettings(), heard = $('#liveHeard'); heard.textContent = '';
      if ($('#liveMeaning').checked) {
        try {
          this.stt = this.stt || new SpeechToText();
          this.busy(t('sttLoading')); await this.stt.load((a, b) => this.busy(`${t('sttLoading')} ${Math.round(a / 1e6)} / ${Math.round(b / 1e6)} MB`)); this.stt.forget();
          const model = set.model === DEFAULT_MODEL ? LIVE_MODEL : set.model; let failed = 0;
          const lang = set.lang || (SPEECH_LANGS[getLang()] ? getLang() : undefined); // a few words are too little to tell the language from, so live starts from the chosen one (or the language of the page)
          meaning = { transcribe: (x) => this.stt.line(x, lang), understand: async (lines) => {
            if (set.key && failed < 3) { try { const r = await understandRemote(lines, { key: set.key, model }); failed = 0; return r; } catch (e) { failed++; console.warn('understanding', e); heard.dataset.why = `${t('aiFailed')} (${t('aiErr_' + e.code)})`; } }
            return understandLocal(lines);
          } };
        } catch (e) { console.warn(e); heard.textContent = `${t('sttFailed')} (${e.message})`; } finally { this.busy(null); }
      }
      const onHeard = (m) => { heard.textContent = (m.emo ? `${t(m.emo)} ${Math.round(m.intensity * 100)}% · ` : '') + m.text + (heard.dataset.why ? ` · ${heard.dataset.why}` : ''); };
      this.live = new LiveSession({ rtModel: this.rtModel, rtExtra: this.rtExtra, video: $('#camPreview'), onStatus: () => {}, profile: this.acting[$('#liveAvatar').value], style: $('#liveStyle') ? $('#liveStyle').value : 'anime', delay: +$('#liveDelay').value, meaning, onHeard });
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
        this.liveStage.apply(0, frameToObj(f.y), this._dt || 0);
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
