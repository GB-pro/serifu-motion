// Serifu Motion — temporal conv net inference (weights trained in train/train.py).
function b64ToF32(b64) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

export class EmoNet {
  constructor(json) {
    this.meta = json.meta;
    const W = {};
    for (const [k, v] of Object.entries(json.weights)) W[k] = { shape: v.shape, data: b64ToF32(v.b64) };
    this.W = W;
    this.C = this.meta.C; this.dil = this.meta.dil; this.Din = this.meta.feat_dim;
    this.Dout = W['out.weight'].shape[0];
  }
  // pointwise conv: y[o] = b[o] + sum_i w[o,i] x[i]
  static pw(w, b, x, Dout, Din, out) {
    for (let o = 0; o < Dout; o++) { let s = b[o]; const off = o * Din; for (let i = 0; i < Din; i++) s += w[off + i] * x[i]; out[o] = s; }
    return out;
  }
  // full-sequence inference. X: Float32Array(T*Din) -> Float32Array(T*Dout)
  run(X, T) {
    const { C, dil, Din, Dout, W } = this;
    const causal = this.meta.causal;
    let H = new Float32Array(T * C);
    const xin = new Float32Array(Din), tmp = new Float32Array(C);
    for (let t = 0; t < T; t++) { for (let i = 0; i < Din; i++) xin[i] = X[t * Din + i]; EmoNet.pw(W['inp.weight'].data, W['inp.bias'].data, xin, C, Din, tmp); H.set(tmp, t * C); }
    const z = new Float32Array(C), g = new Float32Array(C);
    for (let l = 0; l < dil.length; l++) {
      const d = dil[l], w = W[`convs.${l}.weight`].data, b = W[`convs.${l}.bias`].data;
      const mw = W[`mix.${l}.weight`].data, mb = W[`mix.${l}.bias`].data;
      const Hn = new Float32Array(T * C);
      for (let t = 0; t < T; t++) {
        // taps: causal -> t-2d, t-d, t ; symmetric -> t-d, t, t+d
        const taps = causal ? [t - 2 * d, t - d, t] : [t - d, t, t + d];
        for (let o = 0; o < C; o++) {
          let s = b[o];
          const ob = o * C * 3;
          for (let k = 0; k < 3; k++) {
            const tt = taps[k]; if (tt < 0 || tt >= T) continue;
            const hb = tt * C;
            for (let i = 0; i < C; i++) s += w[ob + i * 3 + k] * H[hb + i];
          }
          g[o] = gelu(s);
        }
        EmoNet.pw(mw, mb, g, C, C, z);
        for (let o = 0; o < C; o++) Hn[t * C + o] = H[t * C + o] + z[o];
      }
      H = Hn;
    }
    const Y = new Float32Array(T * Dout), hh = new Float32Array(C), yo = new Float32Array(Dout);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < C; i++) hh[i] = gelu(H[t * C + i]);
      EmoNet.pw(W['out.weight'].data, W['out.bias'].data, hh, Dout, C, yo); Y.set(yo, t * Dout);
    }
    return Y;
  }
}

// Incremental causal inference for realtime use: one feature frame in, one output frame out (delayed by lookahead).
export class StreamingEmoNet {
  constructor(net) {
    if (!net.meta.causal) throw new Error('streaming requires causal model');
    this.net = net;
    const { C, dil } = net;
    this.hist = dil.map((d) => ({ buf: new Float32Array((2 * d + 1) * C), len: 2 * d + 1, pos: 0 }));
    this.tmp = new Float32Array(C); this.g = new Float32Array(C); this.z = new Float32Array(C); this.out = new Float32Array(net.Dout);
  }
  step(x) {
    const { C, dil, Din, Dout, W } = this.net;
    let h = new Float32Array(C);
    EmoNet.pw(W['inp.weight'].data, W['inp.bias'].data, x, C, Din, h);
    for (let l = 0; l < dil.length; l++) {
      const d = dil[l], H = this.hist[l];
      H.buf.set(h, H.pos * C);
      const cur = H.pos;
      H.pos = (H.pos + 1) % H.len;
      const w = W[`convs.${l}.weight`].data, b = W[`convs.${l}.bias`].data;
      const idx = [(cur - 2 * d + 2 * H.len) % H.len, (cur - d + H.len) % H.len, cur];
      for (let o = 0; o < C; o++) {
        let s = b[o]; const ob = o * C * 3;
        for (let k = 0; k < 3; k++) { const hb = idx[k] * C; for (let i = 0; i < C; i++) s += w[ob + i * 3 + k] * H.buf[hb + i]; }
        this.g[o] = gelu(s);
      }
      EmoNet.pw(W[`mix.${l}.weight`].data, W[`mix.${l}.bias`].data, this.g, C, C, this.z);
      const hn = new Float32Array(C);
      for (let o = 0; o < C; o++) hn[o] = h[o] + this.z[o];
      h = hn;
    }
    for (let i = 0; i < C; i++) this.tmp[i] = gelu(h[i]);
    return EmoNet.pw(W['out.weight'].data, W['out.bias'].data, this.tmp, Dout, C, this.out);
  }
}
