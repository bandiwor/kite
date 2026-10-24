#!/bin/sh
# usage: tools/bench.sh [runs]   — best-of-N wall time for every benchmark (needs a release build)
BIN="$(dirname "$0")/../target/release/kite"
RUNS=${1:-3}
for f in "$(dirname "$0")"/../examples/bench/*.kite; do
  best=999999
  for i in $(seq "$RUNS"); do
    s=$(date +%s%N); "$BIN" "$f" >/dev/null; e=$(date +%s%N)
    t=$(( (e - s) / 1000000 ))
    [ "$t" -lt "$best" ] && best=$t
  done
  printf "%-12s %6d ms\n" "$(basename "$f" .kite)" "$best"
done
