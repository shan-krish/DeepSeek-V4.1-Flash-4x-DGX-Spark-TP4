#!/bin/bash
# Bring the DeepSeek-V4.1 4-Spark ring up after any node reboot / mesh doubt.
# A neighbour reboot can leave the RoCEnante mesh (markers / hairpin rules) stale, and a
# stale path makes the engine fail at "RoCE queue-pair connect ... Connection timed out".
# Steps: stop engine -> fabric GID check/fix -> re-apply dsv41-mesh on all nodes -> verify rules/markers/watchdog ->
# RDMA test of all 4 opposite-node paths -> serve. Aborts before serving if any check fails.
# usage: ~/dsv41-ring-up.sh [--no-serve]
set -uo pipefail
NODES="10.10.1.102 10.10.1.95 10.10.1.124 10.10.1.117"
cd ~/dsv41-knapcio || exit 1
fail() { echo "RING-UP FAILED: $*"; echo "Fallback without the mesh: cp .env.tp4.prod-ring-dual .env.tp4 && ./start-tp4.sh serve"; exit 1; }
echo "== stop engine"; ./start-tp4.sh stop >/dev/null 2>&1
# The GPU shares host memory, and SGLang sizes the KV pool from what is free at start: the page cache
# alone cost 14% of the pool (3.29M vs 3.80M tokens), so drop it and compact memory first.
echo "== free host memory (drop page cache, compact)"
for n in $NODES; do
  ssh -o BatchMode=yes $n 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"; echo "$(hostname) available $(free -g | awk "/Mem/{print \$7}") GiB"' </dev/null | sed 's/^/  /'
done
echo "== fabric GID check (IPv4 RoCE v2 must be at index 3; a cable swap can move it)"
for n in $NODES; do
  out=$(ssh -o BatchMode=yes $n "~/fabric-gidfix.sh" </dev/null 2>&1) || fail "GID fix on $n: $(echo "$out" | grep -v ': ok$' | tr '\n' ' ')"
  echo "$out" | grep -v ': ok$' | sed 's/^/  /'
done
echo "== re-apply mesh + checks"
for n in $NODES; do
  out=$(ssh -o BatchMode=yes $n "sudo systemctl restart dsv41-mesh && r=0; for d in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do r=\$((r+\$(tc -s filter show dev \$d ingress 2>/dev/null | grep -c in_hw))); done; echo \"\$(hostname) rules=\$r markers=\$(systemctl list-units --no-legend --state=active dsv41-mesh-marker@* | wc -l) watchdog=\$(test -e /dev/watchdog0 && echo ok || echo MISSING)\"" </dev/null 2>&1)
  echo "  $out"
  [[ "$out" == *"rules=2 markers=2 watchdog=ok"* ]] || fail "mesh/watchdog check on $n"
done
echo "== opposite-node RDMA paths"
while read c cd s sd; do
  ssh -o BatchMode=yes $s "timeout 15 ib_write_lat -d $sd -x 3 --flow_label=16383 -s 61440 -n 300 -F" </dev/null >/dev/null 2>&1 & sleep 2
  r=$(ssh -o BatchMode=yes $c "timeout 12 stdbuf -oL ib_write_lat -d $cd -x 3 --flow_label=16383 -s 61440 -n 300 -F $s" </dev/null 2>&1 | awk "/^ *61440/{print \$5}"); wait
  echo "  $c $cd -> $s $sd: ${r:-FAILED} us"
  [ -n "$r" ] || fail "path $c $cd -> $s $sd"
done <<LIST
10.10.1.102 rocep1s0f0 10.10.1.124 rocep1s0f1
10.10.1.102 roceP2p1s0f1 10.10.1.124 roceP2p1s0f0
10.10.1.95 rocep1s0f0 10.10.1.117 rocep1s0f1
10.10.1.95 roceP2p1s0f1 10.10.1.117 roceP2p1s0f0
LIST
[ "${1:-}" = "--no-serve" ] && { echo "RING-UP OK (not serving, --no-serve)"; exit 0; }
echo "== serve"
: > ~/dsv41-knapcio-serve.log
./start-tp4.sh serve 2>&1 | tee -a ~/dsv41-knapcio-serve.log
grep -E "RoCEnante ready|API is up|\[x\]" ~/dsv41-knapcio-serve.log
# The engine binds and finishes warm-up a few seconds after "API is up" is printed; a single
# probe raced it and reported a healthy engine as failed. Retry before declaring failure.
for _i in $(seq 1 36); do curl -sf -m 5 http://127.0.0.1:8888/health >/dev/null && break; sleep 5; done
curl -sf -m 10 http://127.0.0.1:8888/health >/dev/null || fail "engine not healthy after serve (see ~/dsv41-knapcio-serve.log)"
echo "RING-UP DONE"
