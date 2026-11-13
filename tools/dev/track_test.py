import sys, os, glob, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
MODELS = '/root/sm/models'
opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'face_landmarker.task')), running_mode=vision.RunningMode.IMAGE, num_faces=1,
                                   min_face_detection_confidence=float(sys.argv[2]) if len(sys.argv) > 2 else 0.4, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
fl = vision.FaceLandmarker.create_from_options(opt)
files = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))[::10]
n = 0
for f in files:
    r = fl.detect(mp.Image.create_from_file(f))
    if r.face_blendshapes:
        n += 1
        if n <= 3:
            d = {c.category_name: round(c.score, 2) for c in r.face_blendshapes[0] if c.score > 0.2}
            print(os.path.basename(f), d)
print('detected', n, 'of', len(files))
