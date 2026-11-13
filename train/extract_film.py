"""Animated film -> per-frame face blendshapes / head pose (numbers only) for animation-style statistics.
Usage: python extract_film.py <video> <out.npz> [fps]
Runs the face landmarker per frame (IMAGE mode, so shot cuts do not confuse tracking) and records a
shot-cut signal so statistics are computed inside continuous shots only.
"""
import os, sys, subprocess, math
import numpy as np

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')


def rot_to_euler(m):
    R = np.asarray(m)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    return math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])


def main():
    src, out = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', src], capture_output=True, text=True).stdout.strip().split(',')
    W, H = int(probe[0]), int(probe[1])
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'face_landmarker.task')),
                                       running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.4,
                                       output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    fl = vision.FaceLandmarker.create_from_options(opt)
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps={fps}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    BS, HEAD, BOX, CUT = [], [], [], []
    names = None; prev = None; i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        small = fr[::16, ::16].astype(np.int16)
        CUT.append(float(np.abs(small - prev).mean()) if prev is not None else 0.0); prev = small
        r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)))
        if r.face_blendshapes:
            cats = r.face_blendshapes[0]
            if names is None: names = [c.category_name for c in cats]
            BS.append([c.score for c in cats])
            M = np.array(r.facial_transformation_matrixes[0]); HEAD.append(list(rot_to_euler(M)) + [M[0, 3], M[1, 3], M[2, 3]])
            lm = r.face_landmarks[0]; xs = [q.x for q in lm]; ys = [q.y for q in lm]
            BOX.append([min(xs), min(ys), max(xs), max(ys)])
        else:
            BS.append([np.nan] * 52); HEAD.append([np.nan] * 6); BOX.append([np.nan] * 4)
        i += 1
        if i % 2000 == 0: print(i, 'frames, detected', int(np.isfinite(np.array(BS)[:, 0]).sum()), flush=True)
    np.savez_compressed(out, bs=np.array(BS, np.float32), head=np.array(HEAD, np.float32), box=np.array(BOX, np.float32),
                        cut=np.array(CUT, np.float32), bs_names=np.array(names or []), fps=fps)
    print('DONE', i, 'detected', int(np.isfinite(np.array(BS)[:, 0]).sum()))


if __name__ == '__main__':
    main()
