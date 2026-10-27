#!/usr/bin/env bash
# DeepSeek-V4.1-Flash dual-variant ring: ONE engine serving BOTH names.
#   DeepSeek-v4.1-Flash              -> stock weights, unmodified, bit-identical output
#   DeepSeek-v4.1-Flash-Abliterated  -> stock weights + rank-1 abliteration hook on layers 10-35
# Only the STOCK weights folder is mounted; the ablit shard copy is not used here.
# Rollback = ~/dsv41-knapcio-ring-up.sh (stock) or ~/dsv41-knapcio-ablit-up.sh (ablit copy).
set -euo pipefail
export DSV41_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash-1017
export DSV41_NFS_VOLUME=dsv41-1017-weights
export DSV41_SERVED_MODEL_NAME=DeepSeek-v4.1-Flash
export DSV41_IMAGE=dsv41-knapcio:dual-aff52fa
export DSV41_DUAL_VARIANT=1
export DSV41_ABLIT_DIR=$HOME/models/ablit-dual
# PREFILL_SP=comm: transport only (reduce-scatter/all-gather instead of full-row all-reduce),
# nothing row-sharded. The adapter's copy of _forward_layers_hc_pre_from_prev now slices
# forward_batch.ablit_mask alongside R (shard slice at p.lo:p.hi, tail slice at the late-layer
# boundary), and restores it afterwards, so the hook stays row-aligned in every mode.
# 'comm' keeps the mask alignment trivial; '1' is the same code path plus row sharding.
export DSV41_PREFILL_SP=${DSV41_PREFILL_SP:-1}
NODES="10.10.1.102 10.10.1.95 10.10.1.124 10.10.1.117"
WAIT_S=${DUAL_UP_WAIT_S:-480}
fail(){ echo "DUAL-UP FAILED: $*"; exit 1; }

echo "== prechecks (no engine touched)"
[ -f "$DSV41_MODEL_DIR/config.json" ] || fail "stock weights missing: $DSV41_MODEL_DIR"
[ -f "$DSV41_ABLIT_DIR/ablit_direction.safetensors" ] || fail "artifact missing: $DSV41_ABLIT_DIR"
[ -f "$DSV41_ABLIT_DIR/ablit_direction.json" ] || fail "artifact manifest missing"
python3 - "$DSV41_ABLIT_DIR/ablit_direction.json" <<'PYEOF' || fail "artifact json invalid"
import json, sys
d = json.load(open(sys.argv[1]))
order = d.get("lambda_layer_order") or (d.get("hook") or {}).get("lambda_layer_order")
assert order == list(range(10, 36)), "expected lambda_layer_order 10..35, got %r" % (order,)
assert len(d.get("lambdas", [])) == 26 or (d.get("hook") or {}).get("lambda_shape") == [26]
print("  ok: 26 lambdas, layers 10-35, r_shape %s" % ((d.get("hook") or {}).get("r_shape"),))
PYEOF

# Image + artifact must be identical on all four ranks. A fanout can still be streaming when the
# switch is triggered, so wait for convergence instead of racing it (and instead of failing the
# switch, which would make the dashboard reload the previous model for 20 minutes).
deadline=$(( $(date +%s) + WAIT_S ))
while :; do
  ids=""; missing=""
  for n in $NODES; do
    ssh -o BatchMode=yes "$n" "test -f $DSV41_ABLIT_DIR/ablit_direction.safetensors" </dev/null \
      || missing="$missing $n(artifact)"
    id=$(ssh -o BatchMode=yes "$n" "docker image inspect -f '{{.Id}}' $DSV41_IMAGE 2>/dev/null" </dev/null || true)
    [ -n "$id" ] || missing="$missing $n(image)"
    ids="$ids ${id:-none}"
  done
  unique=$(printf '%s\n' $ids | sort -u | wc -l)
  if [ -z "$missing" ] && [ "$unique" -eq 1 ]; then break; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    fail "nodes not converged after ${WAIT_S}s: ids:$ids missing:${missing:-none} — a stale node would run the old adapter and fall back to the Transformers loader"
  fi
  echo "  waiting: ids:$ids missing:${missing:-none} (fanout in progress?)"
  sleep 20
done
echo "  all 4 nodes: artifact + identical image $(printf '%s' $ids | cut -c8-19)"
[ "${1:-}" = "--check" ] && { echo "DUAL-UP CHECK OK (engine not touched)"; exit 0; }
exec "$HOME/dsv41-knapcio-ring-up.sh" "$@"
