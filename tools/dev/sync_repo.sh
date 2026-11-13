#!/bin/bash
# Mirror the working tree into the repository clone (no build output, no caches, no downloaded model files outside app/).
# Usage: tools/dev/sync_repo.sh <repoDir>
set -e
SRC=$(cd "$(dirname "$0")/../.." && pwd); DST=${1:?repo dir}
cd "$DST"; rm -rf app train tools models
(cd "$SRC" && tar cf - --exclude=node_modules --exclude=__pycache__ app train tools models/anim_stats.json \
   render.mjs uitest.mjs sheet.mjs shot.mjs strip.sh package.json package-lock.json) | tar xf -
printf 'node_modules/\n__pycache__/\ndata/\n*.npz\n*.f32\nmodels/*.task\nmodels/emo_*.json\n' > .gitignore
git add -A; git status --short | wc -l
