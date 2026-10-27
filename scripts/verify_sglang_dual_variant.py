#!/usr/bin/env python3
"""Fail the image build if the dual-variant (stock + abliterated) patch did not land.

The hook is inert unless DSV41_DUAL_VARIANT=1, so a silently dropped patch would only show up as
an abliterated request being served by stock weights. These are textual anchors: this script runs
on the staged tree before the image is built and must not import sglang.

usage: verify_sglang_dual_variant.py <python-dir>
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
CHECKS = {
    "sglang/srt/layers/ablit_dual.py": [
        "class AblitDirection",
        "def apply(",
        "def apply_hook(",
        "LAYER_LO, LAYER_HI = 10, 35",
        "mask has",  # row-mismatch guard
    ],
    "sglang/srt/models/deepseek_v4.py": [
        "from sglang.srt.layers.ablit_dual import (",
        "self._ablit_on = bool(_ablit_enabled())",
        "o = _ablit_apply_hook(",
        "forward_batch.ablit_mask = tail.rows(forward_batch.ablit_mask)",
    ],
    "sglang/srt/managers/schedule_batch.py": [
        "def set_ablit_variant(self)",
        '"\\x00ablit"',
    ],
    "sglang/srt/managers/io_struct.py": ["ablit_variant"],
    "sglang/srt/managers/tokenizer_manager.py": ["ablit_variant=obj.ablit_variant"],
    "sglang/srt/managers/scheduler.py": ["ablit_variant=recv_req.ablit_variant"],
    "sglang/srt/entrypoints/openai/serving_chat.py": ['headers.get("x-ablit-variant")'],
    "sglang/srt/entrypoints/http_server.py": ["DeepSeek-v4.1-Flash-Abliterated"],
    "sglang/srt/model_executor/forward_batch_info.py": [
        "ablit_mask: Optional[torch.Tensor] = None",
        "def _init_ablit_mask(",
        "decode_num_tokens_per_req()",
    ],
    "sglang/srt/model_executor/runner_utils/buffers.py": ["ablit_mask: torch.Tensor"],
    "sglang/srt/model_executor/input_buffers.py": ['"ablit_mask"'],
    "sglang/srt/model_executor/cuda_graph_buffer_registry.py": ['GraphSlot(\n            "ablit_mask"'],
    "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py": [
        "ablit_mask=buffers.ablit_mask[:num_tokens]",
        "ablit_mask=_slot(\"ablit_mask\")",
    ],
}

bad = []
for rel, needles in CHECKS.items():
    path = ROOT / rel
    if not path.is_file():
        bad.append(f"{rel}: missing")
        continue
    text = path.read_text()
    for needle in needles:
        if needle not in text:
            bad.append(f"{rel}: missing anchor {needle!r}")

if bad:
    print("dual-variant patch incomplete:", file=sys.stderr)
    for b in bad:
        print("  " + b, file=sys.stderr)
    sys.exit(1)
print(f"dual-variant patch OK ({len(CHECKS)} files)")