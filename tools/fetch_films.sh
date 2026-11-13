#!/bin/bash
# Fetch freely licensed animated films (Blender open movies) for animation-style statistics.
# Only numbers are extracted from them; the videos are not kept in the repository or the app.
UA="film-stats-fetch/1.0 (generic research client)"
OUT=${1:-$HOME/data/films}
fetch() { # name url
  local d="$OUT/$1" f="$OUT/$1/video.${2##*.}"
  mkdir -p "$d"
  [ -s "$d/ok" ] && { echo "have $1"; return; }
  for try in 1 2 3 4 5 6; do
    code=$(curl -sS -L -A "$UA" --retry 2 --max-time 1800 -o "$f" -w "%{http_code}" "$2")
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$code" = "200" ] && [ "$size" -gt 1000000 ]; then echo "$2" > "$d/ok"; echo "ok $1 $size"; return; fi
    echo "retry $1 code=$code size=$size (try $try)"; rm -f "$f"; sleep 620
  done
  echo "FAILED $1"
}
W=https://upload.wikimedia.org/wikipedia/commons/transcoded
fetch elephants_dream "https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov"
fetch sprite_fright "$W/7/76/Sprite_Fright_-_Blender_Open_Movie-full_movie.webm/Sprite_Fright_-_Blender_Open_Movie-full_movie.webm.720p.vp9.webm"
sleep 20
fetch spring "$W/a/a5/Spring_-_Blender_Open_Movie.webm/Spring_-_Blender_Open_Movie.webm.720p.vp9.webm"
sleep 20
fetch glass_half "$W/0/02/Glass_Half_-_Blender_Open_Movie-full_movie.webm/Glass_Half_-_Blender_Open_Movie-full_movie.webm.720p.vp9.webm"
sleep 20
fetch coffee_run "$W/3/3f/Coffee_Run_-_Blender_Open_Movie-full_movie.webm/Coffee_Run_-_Blender_Open_Movie-full_movie.webm.720p.vp9.webm"
sleep 20
fetch cosmos_laundromat "$W/3/36/Cosmos_Laundromat_-_First_Cycle_-_Official_Blender_Foundation_release.webm/Cosmos_Laundromat_-_First_Cycle_-_Official_Blender_Foundation_release.webm.720p.vp9.webm"
echo ALLDONE
