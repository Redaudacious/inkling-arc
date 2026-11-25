# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
"""Per-visit cache owners of Inkling2 (XPU port).

Every (layer, loop pass) visit owns two caches, both registered as attention
layers so vLLM's KV cache manager allocates, prefix-caches and offloads them:

* `Inkling2KVCache`: the visit's K/V (`SlidingWindowSpec` for the local
  layers, `FullAttentionSpec` for the global ones); metadata from
  `FlashAttentionBackend`'s builder.
* `Inkling2ConvState`: the short-conv inputs of the visit's four conv streams
  (K, V, attention output, MLP output) as a per-sequence state of the last
  `sconv_kernel_size - 1` tokens (a `MambaSpec`, so vLLM's hybrid allocator,
  prefix caching in align mode and the KV offload treat it like any recurrent
  state), the streams packed per KV head (`ref.ConvLayout`); metadata from the
  short-conv builder plus per-token request indices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.models.inkling.amd.ops.sconv import sconv_seq_metadata
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    ShortConvAttentionBackend,
    ShortConvAttentionMetadata,
    ShortConvAttentionMetadataBuilder,
)
from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointBuilder
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from .ops.ref import ConvLayout


def _register(prefix: str, layer: nn.Module) -> None:
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = layer


def layer_metadata(prefix: str):
    """This step's metadata of cache owner `prefix` (None in a dummy run)."""
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        return None
    return md.get(prefix)


class Inkling2KVCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, num_heads: int, num_kv_heads: int, head_dim: int,
                 sliding_window: int | None) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads  # read by the metadata builder
        self.head_size = head_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        vllm_config = get_current_vllm_config()
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.dtype = kv_cache_dtype_str_to_dtype(self.kv_cache_dtype, vllm_config.model_config)
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        dtype = self.dtype
        if os.environ.get("VLLM_INKLING2_KV") == "k8v4" and self.sliding_window is None:
            # K8/V4 slots for the full-length (global) visits only: uint8, 2*D bytes per
            # (token, KV head) = K int8 (D) + V int4 (D/2, fp16 scale per 16) + K f32 scale;
            # written by triton_sconv, read by triton_attention. Twice the tokens per page at
            # the bf16 page size, so the pages match the bf16 window layers and the drafter.
            # The 40 window visits stay bf16: their KV is bounded (1,024 tokens) and int4 V
            # error compounds over the looped core.
            dtype = torch.uint8
            block_size = int(os.environ.get("VLLM_INKLING2_KV_BLOCK", str(2 * block_size)))
        if self.sliding_window is not None:
            return SlidingWindowSpec(block_size=block_size, num_kv_heads=self.num_kv_heads,
                                     head_size=self.head_dim, dtype=dtype,
                                     sliding_window=self.sliding_window)
        return FullAttentionSpec(block_size=block_size, num_kv_heads=self.num_kv_heads,
                                 head_size=self.head_dim, dtype=dtype)

    def view(self) -> torch.Tensor:
        """The visit's cache as [num_blocks, H, block_size, 2 * D]."""
        return self.kv_cache


@dataclass
class Inkling2ConvMetadata(ShortConvAttentionMetadata):
    """Short-conv metadata plus what the Inkling2 conv kernel reads, all in
    persistent buffers (the kernel is captured in graphs): per token its request
    and first row, and the group's block table, from which the kernel takes each
    request's state slot (column 0, or in align mode the block of its last
    token, which the runner has already seeded with the previous state)."""

    seq_idx: torch.Tensor | None = None
    query_start: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    align: bool = False
    state_block_size: int = 1
    # prefix caching (align mode): per request the prompt checkpoint's offset into this step
    # (0 = none) and the block it is written to (-1 = none)
    ckpt_off: torch.Tensor | None = None
    ckpt_si: torch.Tensor | None = None


class Inkling2ConvMetadataBuilder(ShortConvAttentionMetadataBuilder):
    metadata_cls = Inkling2ConvMetadata
    needs_causal_conv1d_metadata = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        n_tok = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buf = torch.zeros(n_tok, dtype=torch.int32, device=device)
        self.query_start_buf = torch.zeros(n_tok, dtype=torch.int32, device=device)
        self.align = vllm_config.cache_config.mamba_cache_mode == "align"
        # Prefix caching keeps a short-conv state only at each prompt's checkpoint block, which
        # the kernel writes (Inkling2ConvState.get_kv_cache_spec asks for one in align mode).
        self.ckpt = (MambaPrefillCheckpointBuilder(vllm_config, kv_cache_spec)
                     if self.align and getattr(kv_cache_spec, "num_prefill_checkpoint_blocks", 0) > 0
                     else None)
        n_req = vllm_config.scheduler_config.max_num_seqs
        self.ck_off_buf = torch.zeros(n_req, dtype=torch.int32, device=device)
        self.ck_si_buf = torch.full((n_req,), -1, dtype=torch.int32, device=device)
        self._ck_dirty = False  # buffers hold a checkpoint from an earlier step (reset needed)
        self.decode_len = 1 + spec_tokens(vllm_config)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False, **kw):
        m = common_attn_metadata
        md = super().build(common_prefix_len, m, fast_build, **kw)
        n_act = int(m.query_start_loc_cpu[-1])
        n_pad = m.slot_mapping.shape[0]
        sconv_seq_metadata(m.query_start_loc, m.num_reqs, n_act, self.seq_idx_buf,
                           self.query_start_buf, n_pad)
        ck_kw = {}
        if self.ckpt is not None:
            # 23 Nov night (stack profile: these two fills ran for every conv group on every decode step, outside
            # the decode graph): reset only when an earlier step left a checkpoint in the buffers.
            qlens = m.query_start_loc_cpu.diff().tolist()
            rows = [i for i, q in enumerate(qlens[:m.num_reqs]) if q > self.decode_len]
            ck = None
            if rows and getattr(m, "seq_lens_cpu_upper_bound", None) is not None:
                ck = self.ckpt.build(m, rows)
            if self._ck_dirty or ck is not None:
                self.ck_off_buf.zero_()
                self.ck_si_buf.fill_(-1)
                self._ck_dirty = False
            if ck is not None:
                self.ck_off_buf[ck.request_rows] = ck.checkpoint_offsets
                self.ck_si_buf[ck.request_rows] = ck.state_indices.to(torch.int32)
                self._ck_dirty = True
            ck_kw = dict(ckpt_off=self.ck_off_buf, ckpt_si=self.ck_si_buf)
        return replace(md, **ck_kw, seq_idx=self.seq_idx_buf[:n_pad],
                       query_start=self.query_start_buf[:n_pad],
                       query_start_loc=m.query_start_loc, block_table=m.block_table_tensor,
                       align=self.align, state_block_size=self.kv_cache_spec.block_size)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        return replace(super().update_block_table(metadata, blk_table, slot_mapping),
                       block_table=blk_table)


def spec_tokens(vllm_config) -> int:
    sc = getattr(vllm_config, "speculative_config", None)
    return 0 if sc is None else int(sc.num_speculative_tokens or 0)


class Inkling2ConvBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING2_CONV_STATE"

    @staticmethod
    def get_builder_cls() -> type[Inkling2ConvMetadataBuilder]:
        return Inkling2ConvMetadataBuilder


class Inkling2ConvState(nn.Module, MambaBase):
    def __init__(self, prefix: str, layout: ConvLayout, kernel_size: int) -> None:
        super().__init__()
        self.prefix = prefix
        self.layout = layout
        self.kernel_size = kernel_size
        vllm_config = get_current_vllm_config()
        self.spec = spec_tokens(vllm_config)
        self.dtype = vllm_config.model_config.dtype
        assert self.dtype == torch.bfloat16, "Inkling2 conv state is bf16"
        self.kv_cache = (torch.tensor([]),)
        _register(prefix, self)

    def forward(self): ...

    @staticmethod
    def state_shape(layout: ConvLayout, kernel_size: int, spec: int = 0) -> tuple[int, int]:
        """(kernel_size - 1 + spec, H * W): the last inputs, oldest first (with speculative
        decoding, vLLM's rolling conv state of spec extra rows)."""
        return (kernel_size - 1 + spec, layout.num_heads * layout.width)

    def get_state_shape(self):
        return (self.state_shape(self.layout, self.kernel_size, self.spec),)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (self.dtype,)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Prefix caching (align mode): one prefill checkpoint block per request, written by the
        # conv kernel at the scheduler's checkpoint position (any token position, like Mamba2).
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, MambaSpec) and vllm_config.cache_config.mamba_cache_mode == "align":
            return replace(spec, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=1)
        return spec

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        """The attention / MLP stream states hold the full hidden size on every rank;
        the K/V state holds this rank's KV heads."""
        return self.layout.k_off < 0

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2ConvBackend

    @property
    def state(self) -> torch.Tensor:
        """[num_slots, kernel_size - 1, H * W]."""
        return self.kv_cache[0]
