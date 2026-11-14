// Serifu Motion — 3D stage that hosts one avatar per speaker.
import * as THREE from 'three';
import { RoomEnvironment } from '../vendor/addons/RoomEnvironment.js';
import { createCharacter } from '../avatar/characters.js';
import { loadCast, createFigure, DESIGNS as CAST } from '../avatar/cast.js';

// The characters of the new cast stand on a sculpted head whose base is loaded once; one that is not ready (or not yet
// remade) is shown as its first figure.
let castReady = false;
try { await loadCast(); castReady = true; } catch (e) { console.warn('new cast not loaded, the first figures are used', e); }

export class Stage {
  constructor(container, opts = {}) {
    this.el = container;
    this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, stencil: true, preserveDrawingBuffer: !!opts.preserve, alpha: false });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    const shadows = opts.shadows !== false; // the floor shadow costs about half of the frame time on a software renderer
    this.renderer.shadowMap.enabled = shadows; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    const scene = (this.scene = new THREE.Scene());
    scene.background = this._bgTexture();
    const pm = new THREE.PMREMGenerator(this.renderer);
    scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.add(new THREE.HemisphereLight('#ffffff', '#c9b8a8', 0.55));
    const key = new THREE.DirectionalLight('#fff1e0', 1.7); key.position.set(2.5, 4, 5); key.castShadow = shadows;
    key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -8; key.shadow.camera.right = 8; key.shadow.camera.top = 6; key.shadow.camera.bottom = -3; key.shadow.bias = -0.0005;
    scene.add(key);
    const rim = new THREE.DirectionalLight('#cfe4ff', 1.1); rim.position.set(-4, 3, -4); scene.add(rim);
    const fill = new THREE.DirectionalLight('#ffe9f0', 0.4); fill.position.set(-3, 1, 3); scene.add(fill);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(12, 64), new THREE.ShadowMaterial({ opacity: 0.18 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -1.4; floor.receiveShadow = true; scene.add(floor);
    this.camera = new THREE.PerspectiveCamera(28, 1, 0.1, 100);
    this.chars = [];
    this._resize();
    new ResizeObserver(() => this._resize()).observe(container);
  }
  _bgTexture() {
    const c = document.createElement('canvas'); c.width = 16; c.height = 256;
    const g = c.getContext('2d'); const grd = g.createLinearGradient(0, 0, 0, 256);
    grd.addColorStop(0, '#bfe0f2'); grd.addColorStop(0.65, '#e9f3f8'); grd.addColorStop(1, '#f6efe6');
    g.fillStyle = grd; g.fillRect(0, 0, 16, 256);
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  setCharacters(keys) {
    const sig = keys.join(',');
    if (sig === this._sig) return; this._sig = sig;
    for (const c of this.chars) this.scene.remove(c.root);
    const pool = this._pool || (this._pool = {}), used = {}; // a figure takes a moment to build: each is kept for the next time
    this.chars = keys.map((k) => {
      const id = k + '#' + (used[k] = (used[k] || 0) + 1); if (pool[id]) return pool[id];
      let c; if (castReady && CAST[k] && !this.opts.firstFigures) { c = createFigure(k); if (this.opts.bake !== false) c.bake(this.renderer); } else c = createCharacter(k);
      return (pool[id] = c);
    });
    const n = this.chars.length, gap = 2.5;
    this.chars.forEach((c, i) => {
      c.root.position.x = (i - (n - 1) / 2) * gap;
      c.root.rotation.y = n > 1 ? -((i - (n - 1) / 2) / Math.max(1, (n - 1) / 2)) * 0.28 : 0;
      this.scene.add(c.root);
    });
    this._frame();
  }
  _frame() {
    const n = Math.max(1, this.chars.length), aspect = this.camera.aspect;
    const fov = this.camera.fov * Math.PI / 180;
    const width = n * 2.5 + 0.6, height = 2.9;
    const dist = Math.max(height / 2 / Math.tan(fov / 2), width / 2 / Math.tan(fov / 2) / aspect) + 1.0;
    this.camera.position.set(0, 1.05, dist); this.camera.lookAt(0, 0.75, 0);
    // optional fixed framing (used for close-ups and offline rendering): { y, ly, z }
    if (this.view) { this.camera.position.set(this.view.x || 0, this.view.y ?? 1.05, this.view.z ?? dist); this.camera.lookAt(this.view.x || 0, this.view.ly ?? 0.75, 0); }
  }
  setView(view) { this.view = view || null; this._frame(); }
  _resize() {
    const w = this.el.clientWidth || 640, h = this.el.clientHeight || 360;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%'; this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this._frame();
  }
  // frameObj: rig-space channels { id: value }; dt > 0 advances the follow-through simulation (hair, accessories)
  apply(i, frameObj, dt = 0) { if (this.chars[i]) this.chars[i].apply(frameObj, dt); }
  render() { this.renderer.render(this.scene, this.camera); }
  get canvas() { return this.renderer.domElement; }
}
