#!/bin/bash
# strip.sh <track.f32> <char> <out.png> [step] [T] [start]
S=/tmp/claude-0/-home-user/27debe6e-12cc-5211-be32-f25b302423b0/scratchpad
step=${4:-3}; start=${6:-0}
T=${5:-$(python3 -c "import os,sys; print(os.path.getsize(sys.argv[1])//4//40)" "$1")}
idx=$(python3 -c "import sys; print(list(range(int(sys.argv[3]), int(sys.argv[1]), int(sys.argv[2]))))" $T $step $start)
echo "{\"mode\":\"strip\",\"char\":\"$2\",\"track\":\"$1\",\"idx\":$idx,\"cols\":${COLS:-10},\"cw\":${CW:-192},\"ch\":${CH:-230},\"view\":{\"y\":0.85,\"ly\":0.55,\"z\":11.5}}" > $S/strip_spec.json
node ~/sm/sheet.mjs $S/strip_spec.json "$3"
