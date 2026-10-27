# Dual-variant serving: stock + abliterated from one engine and one weight set

One TP4 ring process serves both `DeepSeek-v4.1-Flash` and `DeepSeek-v4.1-Flash-Abliterated`.
The abliterated variant is **not** a second copy of the weights: it is a rank-1 projection applied
to the attention output of layers 10–35 at request time.

```
y  <-  y - lambda_l * mask * (r . y) * r        r shared, lambda_l per layer, mask per token
```

`mask` is 0 for stock requests (bitwise no-op) and 1 for abliterated ones, taken per token from the
batch, so a single forward pass can carry both variants.

## Why this is exact enough

The published overlay (`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated@de412d2`) replaces only
`attn.wo_b` in layers 10–35. Measured against it:

| quantity | value |
|---|---|
| shared direction `r`, cos across layers | 1.000 (one direction, sign-normalised) |
| per-layer `lambda` | 3.295–3.402 (the card's 3.5 is the rounded author value) |
| rank-1 energy of `ΔW` | 0.866–0.891 |
| `‖A − (I − λ·rrᵀ)W‖ / ‖A‖` | **1.96%** — inside the artifact's own FP8 quantisation noise |

Artifact shipped to each rank: `ablit_direction.safetensors` (20,768 B: `r` float32[5120] + 26 λ).

## What the patch adds

`patches/sglang-dual-variant-abliteration.patch` (13 files):

| area | change |
|---|---|
| `srt/layers/ablit_dual.py` (new) | artifact loader, masked rank-1 `apply()`, row-count guard, `enabled()` gate |
| `srt/models/deepseek_v4.py` | enable per layer 10–35, apply after `wo_b` (before the fused all-reduce, which is exact for a linear map), slice the mask at the layer-20 bounded-replay tail |
| `srt/managers/*` | `ablit_variant` flag on `GenerateReqInput` → tokenizer → `Req` |
| `srt/managers/schedule_batch.py` | `Req.set_ablit_variant()` appends `"\x00ablit"` to `extra_key`, so radix / FULL / SWA / compressed-KV pages and L3 prefetch never mix variants |
| `srt/model_executor/forward_batch_info.py` | `ablit_mask` built per token (extend: `repeat_interleave(extend_seq_lens)`; decode/verify: `repeat_interleave(decode_num_tokens_per_req())`) |
| `srt/model_executor/*buffers*`, `cuda_graph_buffer_registry.py`, `decode_cuda_graph_runner.py` | replay-safe static buffer + graph slot, zero padding, clear-when-absent |
| `srt/entrypoints/*` | read `X-Ablit-Variant: 1`; advertise both names in `/v1/models` |

`adapter/prefill_sp.py` (this repo's own adapter) additionally:

- re-pins `v4.MQALayer.forward` and `v4.DeepseekV4Model._forward_layers_hc_pre_from_prev` to the
  patched sources (`256a61f4fffa21ea`, `9d54c8f1df58f972`) — **any further edit to either function
  must be re-pinned**, or the adapter refuses at import and SGLang falls back to the Transformers
  loader (which dies later in the multimodal processor);
- slices `forward_batch.ablit_mask` alongside the rows it tails at the late-layer boundary, and
  restores it afterwards. No shard slice is needed: the attention/MoE wrappers gather the row shard
  back to full rows before `wo_b`, so the hook always sees `p.rows` rows.

## Configuration

| flag | meaning |
|---|---|
| `DSV41_DUAL_VARIANT=1` | installs the hook; with `0` every code path above is inert (no allocation, no cache-key change, `/v1/models` unchanged) |
| `DSV41_ABLIT_DIR` | directory holding `ablit_direction.{safetensors,json}` (mounted at `/ablit`) |
| `X-Ablit-Variant: 1` | per-request variant selector; the proxy sets it when the client asks for the abliterated name |

`start.sh` mounts the artifact read-only into head and workers, forwards the two variables, and
allows `DSV41_PREFILL_SP` to override `EXTRA_CONTAINER_ENV`. `scripts/dual-up.sh` launches the ring
with the stock weights folder, the stock worker volume, and the stock served name; it refuses to
start unless all four ranks carry the same image and artifact.

## Measured (4× DGX Spark, TP4, aff52fa tree)

Ladder, cold prefill tok/s:

| run | 20k | 40k | 80k | 160k | 240k |
|---|---|---|---|---|---|
| single-variant stock (reference) | 5718 | 4649* | 5890 | 5863 | 5804 |
| **dual, stock name** | 5467 | 5816 | 5825 | 5755 | 5669 |
| **dual, abliterated name** | 5526 | 5682 | 5864 | 5573 | 5664 |

\*outlier point (its neighbours are 5718/5890).

Decode (prose / code / counting, 1200 tok): 58.2 / 113.7 / 152.4 stock, 57.9 / 107.8 / 154.2 ablit,
against 58.8 / 113.3 / 156.7 single-variant. KV pool 3,728,896 tokens.

`DSV41_PREFILL_SP=1` is required: with `0` or `comm` prefill falls to ~4.6–4.8k (−19%). `comm` alone
recovers nothing because all-reduce *is* reduce-scatter + all-gather; the win is stage-1 row sharding
(plus the stage-2 MXFP8 attention-input gathers).

## Quality

- qeval (75 tasks), stock vs abliterated in the same process: **0 broke, 0 fixed**, p=1.000.
- `treadon/abliteration-eval` (333 prompts), abliterated name:
  harmful **0/200**, over-refusal **0/83**, multilingual **0/50** refused — identical under
  `SP=0`, `comm` and `SP=1`. LLM judge over all 333 answers: 0 REFUSAL, 0 PARTIAL, 333 COMPLY.
- Stock control (same prompts, same process): 57/60 sampled harmful prompts still refused.
- Needle at 124k: PASS on both names. Interleaved mixed batches: 12/12 identical to solo runs.

## Rollback

Select `DeepSeek-V4.1-Flash` or `DeepSeek-V4.1-Flash Abliterated` on `/inference` (runs the
single-variant scripts on the previous image), or `DSV41_DUAL_VARIANT=0` with the same image.
No weight file is ever written by this feature.

## Known limits

- The hook is one GEMV + rank-1 update per layer per forward; measured cost is inside run-to-run
  noise (dual stock vs dual abliterated at 240k: 5669 vs 5664).
- `DSV41_PREFILL_SP=1` requires the mask slices in `adapter/prefill_sp.py`; dropping them makes the
  hook see 128 tail rows against a full-batch mask and fails loudly with a row-count error.
- The measurements above were taken on the `aff52fa` tree (same SGLang REF and the same files as
  current `main`); the v2.6 image has not been benchmarked with this patch yet.