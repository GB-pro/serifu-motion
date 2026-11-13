"""Build the comparison video (actor | linear | animation layer) for held-out CREMA-D actors.
python tools/build_compare.py <workDir> <out.mp4> [steps=gen,render,compose,join] [models=comma list]
The lines are chosen by a fixed rule (no hand picking): each of four held-out actors says six different sentences, one per emotion.
"""
import sys, os, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work, out = sys.argv[1], sys.argv[2]
steps = (sys.argv[3] if len(sys.argv) > 3 else 'gen,render,compose,join').split(',')
models = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, 'models', 'emo_offline.json')
DATA = os.path.expanduser('~/data/crema')
SENTS = ['DFA', 'IEO', 'IOM', 'ITH', 'ITS', 'IWL', 'IWW', 'MTI', 'TAI', 'TIE', 'TSI', 'WSI']
ORDER = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
TAKES = [('1089', 'hana'), ('1085', 'kenta'), ('1083', 'moku'), ('1091', 'mimi')]  # held-out actors (never seen in training)
VIEW = {'y': 0.92, 'ly': 0.62, 'z': 8.2}
os.makedirs(work, exist_ok=True)
names = []
for j, (actor, avatar) in enumerate(TAKES):
    clips = [f'{actor}_{SENTS[(2 * i + 3 * j + 1) % 12]}_{e}_XX' for i, e in enumerate(ORDER)]
    # the sentence IEO was recorded at stated intensity levels: take the medium one where no unmarked take exists
    clips = [c if os.path.exists(os.path.join(DATA, 'feat', c + '.wav')) else c[:-2] + 'MD' for c in clips]
    name = f'take_{actor}_{avatar}'; names.append(name)
    if 'gen' in steps:
        subprocess.run(['node', os.path.join(ROOT, 'tools', 'make_compare.mjs'), models, os.path.join(DATA, 'feat'), work, name, avatar, ','.join(clips)], check=True)
if 'render' in steps:
    jobs = []
    for name in names:
        tk = json.load(open(os.path.join(work, name + '.json')))
        for kind in ['linear', 'auto', 'fix']:
            if kind == 'fix' and all(c['ok'] for c in tk['clips']): continue
            job = {'avatars': [tk['avatar']], 'tracks': [os.path.join(work, f'{name}.{kind}.f32')], 'framesDir': os.path.join(work, 'frames', f'{name}.{kind}')}
            if kind == 'fix': job['ranges'] = [[max(0, c['f0'] - 6), min(tk['T'], c['f1'] + 14)] for c in tk['clips'] if not c['ok']]
            jobs.append(job)
    jobs.sort(key=lambda j: -sum(b - a for a, b in j.get('ranges', [[0, 600]])))
    NP = int(os.environ.get('NP', '3')); procs = []
    for i in range(NP):
        spec = os.path.join(work, f'render_{i}.json'); json.dump({'w': 480, 'h': 480, 'view': VIEW, 'jobs': jobs[i::NP]}, open(spec, 'w'))
        procs.append(subprocess.Popen(['node', os.path.join(ROOT, 'render.mjs'), spec]))
    for p in procs: p.wait()
parts = []
if 'compose' in steps or 'join' in steps:
    for mode in ['main', 'fix']:
        for name in names:
            o = os.path.join(work, f'{name}.{mode}.mp4')
            if 'compose' in steps:
                if os.path.exists(o): os.remove(o)
                subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'compose_compare.py'), os.path.join(work, name + '.json'), os.path.join(DATA, 'flv'), os.path.join(work, 'frames'), o, mode], check=True)
            if os.path.exists(o): parts.append(o)
if 'join' in steps:
    lst = os.path.join(work, 'list.txt'); open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '44100', '-b:a', '128k', '-movflags', '+faststart', out], check=True)
    print('wrote', out)
