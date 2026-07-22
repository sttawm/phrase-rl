#!/bin/bash
# Checkpoint <-> git archive tool. GitHub caps files at 100MB, so each
# checkpoint dir is stored as a gzip tar split into 90MB parts under
# results/checkpoints/archive/<name>.tgz.part-* plus a sha256 manifest.
#
#   ckpt_archive.sh pack <ckpt_dir> [<ckpt_dir> ...]   # dir -> parts (+manifest)
#   ckpt_archive.sh unpack <name> [<dest_parent>]      # parts -> dir (default results/checkpoints/)
#
# Standing discipline (CLAUDE.md): every named checkpoint an experiment
# depends on gets packed and committed the moment it is produced/pulled.
set -euo pipefail
ARC=results/checkpoints/archive
mode=$1; shift

if [ "$mode" = pack ]; then
  mkdir -p $ARC
  for d in "$@"; do
    d=${d%/}
    name=$(basename "$d")
    parent=$(dirname "$d")
    tar czf "$ARC/$name.tgz" -C "$parent" "$name"
    sha=$(shasum -a 256 "$ARC/$name.tgz" 2>/dev/null || sha256sum "$ARC/$name.tgz")
    sz=$(du -h "$ARC/$name.tgz" | cut -f1)
    split -b 90m "$ARC/$name.tgz" "$ARC/$name.tgz.part-"
    rm "$ARC/$name.tgz"
    echo "{\"name\": \"$name\", \"tgz_sha256\": \"${sha%% *}\", \"tgz_size\": \"$sz\", \"src\": \"$d\", \"parts\": $(ls $ARC/$name.tgz.part-* | wc -l | tr -d ' ')}" > "$ARC/$name.manifest.json"
    echo "packed $name ($sz)"
  done
elif [ "$mode" = unpack ]; then
  name=$1
  dest=${2:-results/checkpoints}
  mkdir -p "$dest"
  cat $ARC/$name.tgz.part-* > /tmp/_ckpt_$name.tgz
  want=$(python3 -c "import json;print(json.load(open('$ARC/$name.manifest.json'))['tgz_sha256'])")
  got=$(shasum -a 256 /tmp/_ckpt_$name.tgz 2>/dev/null || sha256sum /tmp/_ckpt_$name.tgz); got=${got%% *}
  [ "$want" = "$got" ] || { echo "SHA MISMATCH for $name"; exit 1; }
  tar xzf /tmp/_ckpt_$name.tgz -C "$dest"
  rm /tmp/_ckpt_$name.tgz
  echo "unpacked $name -> $dest/$name"
else
  echo "usage: $0 pack|unpack ..."; exit 1
fi
