"""Write actor-normalized real motion (GT emo) for given clips as .gt.f32 (same space as generated)."""
import sys, os, numpy as np
from dataset import load_all, actor_baselines, apply_baseline
feat_dir, out_dir, names = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
clips = load_all(feat_dir); base, gneu = actor_baselines(clips); apply_baseline(clips, base, gneu)
os.makedirs(out_dir, exist_ok=True)
for c in clips:
    if c['name'] in names:
        c['Yn'].astype(np.float32).tofile(os.path.join(out_dir, c['name'] + '.gt.f32')); print(c['name'], len(c['Yn']))
