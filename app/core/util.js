// Serifu Motion — small shared helpers (seeded random numbers, clamping, easing).
export function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
export const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smooth = (u) => { u = clamp(u); return u * u * (3 - 2 * u); };
export function softmax(a) { const mx = Math.max(...a); const e = a.map((v) => Math.exp(v - mx)); const s = e.reduce((p, q) => p + q, 0); return e.map((v) => v / s); }
