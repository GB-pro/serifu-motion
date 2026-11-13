#!/bin/bash
# Run the face tracker over rendered avatar takes (frames -> video -> per-frame blendshapes / head pose at 24 fps).
# track_takes.sh <workDir> [kinds="linear auto"]
W=$1; KINDS=${2:-"linear auto"}
mkdir -p $W/tracked
for k in $KINDS; do for d in $W/frames/*.$k; do
  n=$(basename $d); out=$W/tracked/$n.npz
  [ -f $out ] && continue
  ffmpeg -v error -y -framerate 30 -i $d/%05d.jpg -c:v libx264 -crf 14 -pix_fmt yuv420p $W/tracked/$n.mp4 || continue
  ~/venv/bin/python -I ~/sm/train/extract_film.py $W/tracked/$n.mp4 $out 24 2>/dev/null | tail -1
done; done
