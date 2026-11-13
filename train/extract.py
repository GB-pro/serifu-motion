"""CREMA-D video -> per-frame FACS-ish blendshapes, head pose, torso pose (numbers only).
Also writes 16 kHz mono wav for the shared JS feature extractor.
Usage: python extract.py <flv_dir> <out_dir> [n_workers]
"""
import os, sys, subprocess, math, glob
import numpy as np
from multiprocessing import Pool

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
W, H = 480, 360


def rot_to_euler(m):
    # m: 4x4 facial transformation matrix (column-major semantics from mediapipe -> numpy row-major)
    R = np.asarray(m)[:3, :3]
    # yaw (Y), pitch (X), roll (Z)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(-R[2, 0], sy)
    roll = math.atan2(R[1, 0], R[0, 0])
    return pitch, yaw, roll


_FL = _PL = None
_TS = 0


def process(args):
    src, out_dir = args
    name = os.path.splitext(os.path.basename(src))[0]
    out_npz = os.path.join(out_dir, name + '.npz')
    out_wav = os.path.join(out_dir, name + '.wav')
    if os.path.exists(out_npz) and os.path.exists(out_wav):
        return name, 'skip'
    global _FL, _PL, _TS
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', '16000', out_wav], check=True)
        p = subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps=30,scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           capture_output=True, check=True)
        frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
        fopt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'face_landmarker.task')),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        popt = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'pose_landmarker_lite.task')),
                                            running_mode=vision.RunningMode.VIDEO, num_poses=1)
        bs_names = None
        BS, HEAD, TORSO, OK = [], [], [], []
        if _FL is None:
            _FL = vision.FaceLandmarker.create_from_options(fopt)
            _PL = vision.PoseLandmarker.create_from_options(popt)
        fl, pl = _FL, _PL
        _TS += 5000  # gap between clips resets tracking
        if True:
            q = None
            for i, fr in enumerate(frames):
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr))
                ts = _TS + int(i * 1000 / 30)
                r = fl.detect_for_video(img, ts)
                if i % 3 == 0:
                    q = pl.detect_for_video(img, ts)
                if r.face_blendshapes:
                    cats = r.face_blendshapes[0]
                    if bs_names is None:
                        bs_names = [c.category_name for c in cats]
                    BS.append([c.score for c in cats])
                    M = np.array(r.facial_transformation_matrixes[0])
                    p_, y_, r_ = rot_to_euler(M)
                    HEAD.append([p_, y_, r_, M[0, 3], M[1, 3], M[2, 3]])
                    ok = 1
                else:
                    BS.append([np.nan] * 52); HEAD.append([np.nan] * 6); ok = 0
                if q.pose_landmarks:
                    lm = q.pose_landmarks[0]
                    ls, rs, nose = lm[11], lm[12], lm[0]
                    # torso roll (shoulder line), lateral shift, shoulder height rel nose (shrug), width (lean fwd proxy)
                    dx, dy = (ls.x - rs.x) * W, (ls.y - rs.y) * H
                    roll = math.atan2(dy, dx)
                    midx, midy = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
                    width = math.hypot(dx, dy) / W
                    TORSO.append([roll, midx, midy - nose.y, width, (ls.visibility + rs.visibility) / 2])
                else:
                    TORSO.append([np.nan] * 5)
                OK.append(ok)
        _TS += int(len(frames) * 1000 / 30) + 1000
        np.savez_compressed(out_npz, bs=np.array(BS, np.float32), head=np.array(HEAD, np.float32),
                            torso=np.array(TORSO, np.float32), ok=np.array(OK, np.int8),
                            bs_names=np.array(bs_names or []))
        return name, 'ok'
    except Exception as e:
        return name, 'err ' + str(e)[:200]


if __name__ == '__main__':
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, '*.flv')))
    lim = int(os.environ.get('LIMIT', '0'))
    if lim:
        files = files[:lim]
    n = 0
    with Pool(nw) as pool:
        for name, st in pool.imap_unordered(process, [(f, out_dir) for f in files], chunksize=4):
            n += 1
            if st.startswith('err') or n % 200 == 0:
                print(n, name, st, flush=True)
    print('DONE', n)
