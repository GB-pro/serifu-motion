"""Verification video: original clip | avatar driven by the extracted numbers | per-frame AU/head bars."""
import sys, os, subprocess, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataset import NAMES
names, gt_dir, flv_dir, out = sys.argv[1].split(','), sys.argv[2], sys.argv[3], sys.argv[4]
FONT = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
f14, f18 = ImageFont.truetype(FONT, 14), ImageFont.truetype(FONT, 18)
EMO_JA = {'NEU': '平静', 'HAP': '喜び', 'SAD': '悲しみ', 'ANG': '怒り', 'FEA': '恐れ', 'DIS': '嫌悪'}
AUS = [('AU1', '眉内上げ'), ('AU2', '眉外上げ'), ('AU4', '眉寄せ'), ('AU5', '目見開き'), ('AU6', '頬上げ'), ('AU7', 'まぶた緊張'), ('AU9', '鼻しわ'),
       ('AU10', '上唇上げ'), ('AU12', '口角上げ'), ('AU15', '口角下げ'), ('AU20', '唇横引き'), ('AU26', '顎下げ'), ('AU45', '瞬き')]
HEAD = [('headPitch', '頭 上下'), ('headYaw', '頭 左右'), ('headRoll', '頭 傾き')]
tmp = '/tmp/claude-0/vv'; os.makedirs(tmp, exist_ok=True)
parts = []
for n in names:
    Y = np.fromfile(os.path.join(gt_dir, n + '.gt.f32'), np.float32).reshape(-1, len(NAMES))
    av = os.path.join(gt_dir, n + '.avatar.mp4')
    cd = os.path.join(tmp, n); os.makedirs(cd, exist_ok=True)
    emo = n.split('_')[2]
    for t, y in enumerate(Y):
        im = Image.new('RGB', (480, 360), (24, 27, 33)); d = ImageDraw.Draw(im)
        d.text((10, 6), f'抽出した数値（学習データ） {EMO_JA[emo]}', font=f18, fill=(255, 255, 255))
        for i, (k, ja) in enumerate(AUS):
            v = float(np.clip(y[NAMES.index(k)], 0, 1)); yy = 36 + i * 19
            d.text((8, yy), f'{k} {ja}', font=f14, fill=(200, 205, 215))
            d.rectangle([170, yy + 3, 170 + 290, yy + 14], fill=(50, 55, 65)); d.rectangle([170, yy + 3, 170 + int(290 * v), yy + 14], fill=(255, 122, 89))
        for i, (k, ja) in enumerate(HEAD):
            v = float(np.clip(y[NAMES.index(k)], -20, 20)); yy = 290 + i * 21
            d.text((8, yy), f'{ja} {v:+.1f}°', font=f14, fill=(200, 205, 215))
            d.rectangle([170, yy + 3, 460, yy + 14], fill=(50, 55, 65)); c = 315
            d.rectangle([min(c, c + int(v / 20 * 145)), yy + 3, max(c, c + int(v / 20 * 145)), yy + 14], fill=(79, 140, 255)); d.line([c, yy + 1, c, yy + 16], fill=(230, 230, 230))
        im.save(os.path.join(cd, f'{t:05d}.png'))
    seg = os.path.join(tmp, n + '.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(flv_dir, n + '.flv'), '-i', av, '-framerate', '30', '-i', os.path.join(cd, '%05d.png'),
                    '-filter_complex', '[0:v]fps=30,scale=480:360,setpts=PTS-STARTPTS[a];[1:v]scale=480:360[b];[2:v]scale=480:360[c];[a][b][c]hstack=3,format=yuv420p[v]',
                    '-map', '[v]', '-map', '0:a', '-shortest', '-c:v', 'libx264', '-crf', '21', '-c:a', 'aac', '-ar', '44100', seg], check=True)
    parts.append(seg)
with open(os.path.join(tmp, 'list.txt'), 'w') as f:
    for p in parts: f.write(f"file '{p}'\n")
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', os.path.join(tmp, 'list.txt'), '-c', 'copy', out], check=True)
print('wrote', out)
