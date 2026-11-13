"""Lay the recorded actor, the linear rendition and the animation-layer rendition side by side.
python compose_compare.py <take.json> <flvDir> <framesRoot> <out.mp4> [main|fix]
  framesRoot/<name>.linear/%05d.jpg, <name>.auto/..., <name>.fix/...  (rendered by render.mjs)
  main: actor | linear | anime (automatic reading), whole take
  fix : actor | anime (automatic, misread) | anime (emotion given), only the lines the network misread
"""
import sys, os, json, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

take_p, flv_dir, root, out = sys.argv[1:5]
mode = sys.argv[5] if len(sys.argv) > 5 else 'main'
tk = json.load(open(take_p)); name = tk['name']; T = tk['T']
FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
F = lambda s: ImageFont.truetype(FONT, s)
f13, f15, f18, f22, f26 = F(13), F(15), F(18), F(22), F(26)
JA = {'NEU': '平静', 'HAP': '喜び', 'SAD': '悲しみ', 'ANG': '怒り', 'FEA': '恐れ', 'DIS': '嫌悪'}
COL = {'NEU': (150, 158, 170), 'HAP': (245, 190, 60), 'SAD': (86, 140, 235), 'ANG': (235, 86, 76), 'FEA': (170, 110, 235), 'DIS': (90, 180, 110)}
PW, PH, TOP, CAP, TL = 480, 480, 46, 62, 66
W, H = PW * 3, TOP + PH + CAP + TL
BG, FG, DIM = (22, 25, 31), (240, 242, 246), (150, 156, 168)
AV = {'hana': 'ハナ', 'kenta': 'ケンタ', 'moku': 'モク', 'mimi': 'ミミ'}

def actor_frames(clip):
    n = clip['f1'] - clip['f0']
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', os.path.join(flv_dir, clip['name'] + '.flv'), '-vf', 'fps=30,scale=480:360', '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    fr = np.frombuffer(p, np.uint8).reshape(-1, 360, 480, 3)
    return [Image.fromarray(x) for x in fr]

clips = tk['clips']
for c in clips: c['frames'] = actor_frames(c)
plan = tk['plan']['phrases']; plan_fix = tk['planFix']['phrases']

def clip_at(f):
    cur = None
    for c in clips:
        if f >= c['f0']: cur = c
    return cur or clips[0]

def avatar(kind, f):
    p = os.path.join(root, f'{name}.{kind}', f'{min(f, T - 1):05d}.jpg')
    return Image.open(p).convert('RGB').resize((PW, PH), Image.LANCZOS)

def phrase_label(ph): return JA[ph['emo']] + (' + ' + JA[ph['emo2']] if ph.get('emo2') else '')

def timeline(d, y, f, f_lo, f_hi, phrases):
    x0, x1 = 16, W - 16; sx = (x1 - x0) / max(1, f_hi - f_lo)
    X = lambda t: x0 + (t - f_lo) * sx
    d.rectangle([x0, y + 22, x1, y + 48], fill=(38, 42, 50))
    for c in clips:
        if c['f1'] < f_lo or c['f0'] > f_hi: continue
        d.line([X(max(c['f0'], f_lo)), y + 4, X(max(c['f0'], f_lo)), y + 50], fill=(90, 96, 108))
        d.text((X(max(c['f0'], f_lo)) + 5, y + 3), '俳優への指示: ' + JA[c['label']], font=f13, fill=COL[c['label']])
    for ph in phrases:
        if ph['t1'] < f_lo or ph['t0'] > f_hi: continue
        a, b = X(max(ph['tp'], f_lo)), X(min(ph['t1'], f_hi))
        d.rectangle([a, y + 24, b, y + 46], fill=COL[ph['emo']])
        if ph.get('emo2'): d.rectangle([a, y + 38, b, y + 46], fill=COL[ph['emo2']])
        d.text((a + 4, y + 27), phrase_label(ph), font=f13, fill=(20, 20, 24))
        for ac in ph['accents']:
            xa = X(ac['t']); s = 6 if ac['key'] else 4
            d.polygon([(xa - s, y + 58), (xa + s, y + 58), (xa, y + 48)], fill=FG if ac['key'] else DIM)
    xp = X(f); d.line([xp, y + 16, xp, y + 62], fill=(255, 255, 255), width=2)
    d.text((x1 - 300, y + 50), '色: 読み取った感情 / ▲: アクセント(強調)', font=f13, fill=DIM)

def frame(f, kinds, heads, subs, f_lo, f_hi, title, phrases):
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    c = clip_at(f)
    d.text((16, 11), title, font=f22, fill=FG)
    # actor
    im.paste((10, 11, 14), [0, TOP, PW, TOP + PH])
    i = f - c['f0']; fr = c['frames'][min(max(i, 0), len(c['frames']) - 1)]
    if i >= len(c['frames']) or i < 0: fr = Image.eval(fr, lambda v: v // 3)
    im.paste(fr, (0, TOP + 60))
    for j, k in enumerate(kinds): im.paste(avatar(k, f), (PW * (j + 1), TOP))
    for j in (1, 2): d.line([PW * j, TOP, PW * j, TOP + PH + CAP], fill=BG, width=3)
    y = TOP + PH + 8
    d.text((16, y), '元の俳優', font=f22, fill=FG); d.text((16, y + 30), '演技の指示: ', font=f15, fill=DIM); d.text((110, y + 26), JA[c['label']], font=f22, fill=COL[c['label']])
    d.text((250, y + 6), 'CREMA-D', font=f13, fill=DIM); d.text((250, y + 22), '(学習に使っていない俳優)', font=f13, fill=DIM)
    for j in range(2):
        x = PW * (j + 1) + 16
        d.text((x, y), heads[j], font=f22, fill=FG)
        s = subs[j](c) if callable(subs[j]) else subs[j]
        if isinstance(s, tuple): d.text((x, y + 30), s[0], font=f15, fill=DIM); d.text((x + d.textlength(s[0], font=f15) + 6, y + 30), s[1], font=f15, fill=s[2])
        else: d.text((x, y + 30), s, font=f15, fill=DIM)
    timeline(d, TOP + PH + CAP, f, f_lo, f_hi, phrases)
    return im

def read_sub(c):
    ok = c['ok']
    return (f"音声からの読み取り: {' / '.join(dict.fromkeys(JA[p.split('+')[0]] + ('+' + JA[p.split('+')[1]] if '+' in p else '') for p in c['played']))} ({round(c['conf'] * 100)}%)", '一致' if ok else '読み違い', (110, 200, 120) if ok else (240, 110, 100))

def encode(frames_iter, n, audio_segments, out):
    tmp = out + '.tmp.wav'
    w = wave.open(os.path.join(os.path.dirname(take_p), name + '.wav')); sr = w.getframerate(); pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16); w.close()
    seg = np.concatenate([np.pad(pcm[int(a * sr / 30):int(b * sr / 30)], (0, max(0, int((b - a) * sr / 30) - len(pcm[int(a * sr / 30):int(b * sr / 30)])))) for a, b in audio_segments])
    ww = wave.open(tmp, 'w'); ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr); ww.writeframes(seg.astype(np.int16).tobytes()); ww.close()
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-i', tmp, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
                          '-c:a', 'aac', '-ar', '44100', '-b:a', '128k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for im in frames_iter: p.stdin.write(im.tobytes())
    p.stdin.close(); p.wait(); os.remove(tmp)
    print('wrote', out, n, 'frames')

av = AV.get(tk['avatar'], tk['avatar'])
if mode == 'main':
    title = f'Serifu Motion  同じセリフ音声から作った動きの比較   アバター: {av}'
    it = (frame(f, ['linear', 'auto'], ['リニア版 (前回の方式)', 'アニメ解釈版 (今回)'], ['学習した動きをそのまま線形に反映', read_sub], 0, T, title, plan) for f in range(T))
    encode(it, T, [(0, T)], out)
else:
    bad = [c for c in clips if not c['ok']]
    if not bad: print('no misread lines in', name); sys.exit(0)
    title = f'補足: 読み違えたセリフで、感情を手動で指定した場合   アバター: {av}'
    segs = [(max(0, c['f0'] - 6), min(T, c['f1'] + 14)) for c in bad]
    def gen():
        for a, b in segs:
            for f in range(a, b):
                yield frame(f, ['auto', 'fix'], ['アニメ解釈版 (自動)', 'アニメ解釈版 (感情を指定)'], [read_sub, lambda c: ('タイムラインで感情を指定:', JA[c['label']], COL[c['label']])], a, b, title, [p for p in plan_fix])
    encode(gen(), sum(b - a for a, b in segs), segs, out)
