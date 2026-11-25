# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head.

Dual drafters (VLLM_INKLING2_LOOKUP_CHAIN=1, Claude 23 Nov 2026; same idea as Strata's
`--spec 4 --spec-min-p 0.5 --lookup-chain 4` with the default `--suffix-draft`): DSpark is
the first drafter, the suffix lookup the second (no weights, all on the GPU, after the draft
graph). The lookup finds the latest earlier place in the conversation where the current text
already occurred and proposes the tokens that followed it.
- Lead (--suffix-draft): when the history's last VLLM_INKLING2_LOOKUP_LEAD+ tokens repeat an
  earlier stretch and DSpark's first draft agrees with the lookup, the lookup's continuation
  fills the window (copied code, edits, quoted text).
- Chain (--lookup-chain): otherwise DSpark keeps its drafts while it is sure (the first always,
  each further one while it was at least VLLM_INKLING2_LOOKUP_MIN_P sure of the one before) and
  the lookup continues after them (a match of VLLM_INKLING2_LOOKUP_MIN+ tokens, drafts included).
The verify window stays 1 + 7 rows. Lookup drafts are one-hot proposals for the probabilistic
rejection sampler (q = 1 on the token), so the output distribution is unchanged."""

import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

LOOKUP_CHAIN = os.environ.get("VLLM_INKLING2_LOOKUP_CHAIN", "0") == "1"
_LOOKUP_MIN_P = float(os.environ.get("VLLM_INKLING2_LOOKUP_MIN_P", "0.5"))
_LOOKUP_MIN = int(os.environ.get("VLLM_INKLING2_LOOKUP_MIN", "3"))
_LOOKUP_MAX = int(os.environ.get("VLLM_INKLING2_LOOKUP_MAX", "8"))
# The lookup leads (takes the whole window, Strata --suffix-draft) on a repeat this long.
_LOOKUP_LEAD = int(os.environ.get("VLLM_INKLING2_LOOKUP_LEAD", str(_LOOKUP_MAX)))
_LOOKUP_LOG_EVERY = int(os.environ.get("VLLM_INKLING2_LOOKUP_LOG", "1000"))
_SCAN_BLOCK = 256
_VOCAB_BLOCK = 4096


@triton.jit
def _draft_conf_kernel(
    logits_ptr, s_req, s_col,  # [max_num_reqs, K, V] raw draft logits (before temperature)
    idx_ptr,  # [B] batch -> request state
    temp_ptr,  # [max_num_reqs]
    tok_ptr, s_tok,  # [B, K] drafts
    conf_ptr,  # [B, K] out: the draft's probability under the drafter
    V, K: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)
    r = tl.load(idx_ptr + b).to(tl.int64)
    t = tl.load(temp_ptr + r).to(tl.float32)
    inv = tl.where(t > 0.0, 1.0 / t, 1.0)
    base = logits_ptr + r * s_req + k * s_col
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32) * inv
        nm = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - nm) + tl.sum(tl.exp(x - nm), axis=0)
        m = nm
    tok = tl.maximum(tl.load(tok_ptr + b * s_tok + k), 0)
    xt = tl.load(base + tok).to(tl.float32) * inv
    tl.store(conf_ptr + b * K + k, tl.exp(xt - m) / s)


@triton.jit
def _num_kept(conf_ptr, b, min_p, K: tl.constexpr, K_PO2: tl.constexpr):
    # Strata's rule: the first draft always, each further one while the drafter was at
    # least min_p sure of the one before.
    kk = tl.arange(0, K_PO2)
    c = tl.load(conf_ptr + b * K + kk, mask=kk < K - 1, other=0.0)
    low = (c < min_p) | (kk >= K - 1)
    first_low = tl.min(tl.where(low, kk, K_PO2), axis=0)
    return tl.minimum(first_low + 1, K)


@triton.jit
def _chain_scan_kernel(
    tok_ids_ptr, s_ids,  # [max_num_reqs, L] int32 history (prompt + accepted output)
    idx_ptr, total_len_ptr, num_sampled_ptr,
    drafts_ptr, s_drafts,  # [B, K] int64
    conf_ptr, min_p,
    scr_lead_ptr, scr_chain_ptr, s_scratch,  # [B, n_blocks] int64 out
    L,
    K: tl.constexpr, K_PO2: tl.constexpr,
    MIN_N: tl.constexpr, MAX_N: tl.constexpr, MAX_N_PO2: tl.constexpr, BLOCK_L: tl.constexpr,
):
    # Two suffixes in one pass over the history: the lead suffix (the history's own last
    # tokens) and the chain suffix (history + DSpark's kept drafts).
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1
    r = tl.load(idx_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + b)
    kept = _num_kept(conf_ptr, b, min_p, K, K_PO2).to(tl.int64)
    eligible = (ns > 0) & (seq_len > MIN_N)
    chain_ok = eligible & (kept < K)
    off = b * s_scratch + blk
    if not (eligible & (blk * BLOCK_L <= seq_len - MIN_N - 1)):
        tl.store(scr_lead_ptr + off, tl.zeros((), tl.int64))
        tl.store(scr_chain_ptr + off, tl.zeros((), tl.int64))
        return
    row = r * s_ids
    si = tl.arange(0, MAX_N_PO2).to(tl.int64)
    hpos = seq_len - MAX_N + si
    lead_suf = tl.load(tok_ids_ptr + row + hpos, mask=(si < MAX_N) & (hpos >= 0),
                       other=-1).to(tl.int32)
    vpos = seq_len + kept - MAX_N + si
    in_rng = (si < MAX_N) & (vpos >= 0)
    from_hist = vpos < seq_len
    th = tl.load(tok_ids_ptr + row + vpos, mask=in_rng & from_hist, other=-1).to(tl.int32)
    td = tl.load(drafts_ptr + b * s_drafts + (vpos - seq_len), mask=in_rng & (~from_hist),
                 other=-1).to(tl.int32)
    chain_suf = tl.where(from_hist, th, td)
    pos = blk * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int64)
    best_lead = tl.zeros([BLOCK_L], dtype=tl.int64)
    best_chain = tl.zeros([BLOCK_L], dtype=tl.int64)
    for n in tl.static_range(MIN_N, MAX_N + 1):
        # a match must end before the last history token: at least one continuation token
        base = (pos >= 0) & (pos <= seq_len - n - 1)
        m_lead = base
        m_chain = base & chain_ok
        for j in tl.static_range(0, n):
            tok = tl.load(tok_ids_ptr + row + pos + j, mask=base, other=0).to(tl.int32)
            sel = si == (MAX_N - n) + j
            m_lead = m_lead & (tok == tl.sum(tl.where(sel, lead_suf, 0)))
            m_chain = m_chain & (tok == tl.sum(tl.where(sel, chain_suf, 0)))
        best_lead = tl.where(m_lead, n * Lp1 + pos + 1, best_lead)  # longest, then latest
        best_chain = tl.where(m_chain, n * Lp1 + pos + 1, best_chain)
    tl.store(scr_lead_ptr + off, tl.max(best_lead, axis=0))
    tl.store(scr_chain_ptr + off, tl.max(best_chain, axis=0))


@triton.jit
def _chain_finalize_kernel(
    tok_ids_ptr, s_ids, idx_ptr, total_len_ptr, num_sampled_ptr,
    drafts_ptr, s_drafts, conf_ptr, min_p,
    scr_lead_ptr, scr_chain_ptr, s_scratch,
    fill_ptr,  # [B, K] int32 out: lookup token written at that slot, else -1
    prev_from_ptr, prev_n_ptr,  # [max_num_reqs] int32: last step's lookup slots, per request
    stats_ptr,  # [7] int64, see _lookup_chain
    L, N_BLOCKS, LEAD_MIN,
    K: tl.constexpr, K_PO2: tl.constexpr, N_BLOCKS_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1
    r = tl.load(idx_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + b).to(tl.int64)
    live = ns > 0
    # The verdict on last step's lookup drafts: ns - 1 drafts were accepted in a row.
    p_from = tl.load(prev_from_ptr + r).to(tl.int64)
    p_n = tl.load(prev_n_ptr + r).to(tl.int64)
    graded = live & (p_n > 0)
    acc = tl.minimum(tl.maximum(ns - 1 - p_from, 0), p_n)
    tl.atomic_add(stats_ptr + 5, tl.where(graded, acc, 0))
    tl.atomic_add(stats_ptr + 6, tl.where(graded, p_n, 0))
    kept = _num_kept(conf_ptr, b, min_p, K, K_PO2).to(tl.int64)
    nb = tl.arange(0, N_BLOCKS_PO2).to(tl.int64)
    s_lead = tl.max(tl.load(scr_lead_ptr + b * s_scratch + nb, mask=nb < N_BLOCKS, other=0),
                    axis=0)
    s_chain = tl.max(tl.load(scr_chain_ptr + b * s_scratch + nb, mask=nb < N_BLOCKS, other=0),
                     axis=0)
    n_lead = tl.where(s_lead > 0, (s_lead - 1) // Lp1, 0)
    start_lead = tl.where(s_lead > 0, (s_lead - 1) - n_lead * Lp1 + n_lead, 0)
    n_chain = tl.where(s_chain > 0, (s_chain - 1) // Lp1, 0)
    start_chain = tl.where(s_chain > 0, (s_chain - 1) - n_chain * Lp1 + n_chain, 0)
    # Lead (Strata --suffix-draft): the text repeats an earlier stretch of LEAD_MIN+ tokens and
    # DSpark's own first guess agrees with the lookup: the lookup's continuation takes the window.
    first_lk = tl.load(tok_ids_ptr + r * s_ids + start_lead, mask=s_lead > 0, other=-1).to(tl.int64)
    first_dr = tl.load(drafts_ptr + b * s_drafts)
    lead = live & (n_lead >= LEAD_MIN) & (first_lk == first_dr)
    # Chain (Strata --lookup-chain): after DSpark's sure drafts.
    chain = live & (~lead) & (s_chain > 0) & (kept < K)
    lo = tl.where(lead, 1, kept)
    start = tl.where(lead, start_lead + 1, start_chain)
    m = tl.where(lead, tl.minimum(seq_len - start_lead, K) - 1,
                 tl.where(chain, tl.minimum(seq_len - start_chain, K - kept), 0))
    m = tl.maximum(m, 0)
    kk = tl.arange(0, K_PO2).to(tl.int64)
    slot = kk - lo
    ok = (kk >= lo) & (slot < m) & (kk < K)
    g = tl.load(tok_ids_ptr + r * s_ids + start + slot, mask=ok, other=0).to(tl.int64)
    tl.store(drafts_ptr + b * s_drafts + kk, g, mask=ok)
    tl.store(fill_ptr + b * K + kk, tl.where(ok, g, -1).to(tl.int32), mask=kk < K)
    tl.store(prev_from_ptr + r, tl.where(live, lo, 0).to(tl.int32))
    tl.store(prev_n_ptr + r, tl.where(live, m, 0).to(tl.int32))
    tl.atomic_add(stats_ptr + 0, live.to(tl.int64))
    tl.atomic_add(stats_ptr + 1, (lead & (m > 0)).to(tl.int64))
    tl.atomic_add(stats_ptr + 2, (chain & (m > 0)).to(tl.int64))
    tl.atomic_add(stats_ptr + 3, m)
    tl.atomic_add(stats_ptr + 4, tl.where(live, kept, 0))


@triton.jit
def _one_hot_kernel(
    logits_ptr, s_req, s_col, idx_ptr, fill_ptr, V, K: tl.constexpr, BLOCK: tl.constexpr,
):
    # A lookup draft is a deterministic proposal: q = 1 on its token (raw logit 0, the
    # rest -inf; dividing by the temperature keeps it one-hot).
    b = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)
    tok = tl.load(fill_ptr + b * K + k)
    if tok < 0:
        return
    r = tl.load(idx_ptr + b).to(tl.int64)
    offs = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    val = tl.where(offs == tok, 0.0, float("-inf"))
    tl.store(logits_ptr + r * s_req + k * s_col + offs,
             val.to(logits_ptr.dtype.element_ty), mask=offs < V)


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        # Second drafter (suffix lookup); req_states is set by the model runner.
        self.req_states = None
        self._lookup = LOOKUP_CHAIN
        if self._lookup:
            k = self.num_speculative_steps
            self._max_model_len = vllm_config.model_config.max_model_len
            self._n_blocks = triton.cdiv(self._max_model_len, _SCAN_BLOCK)
            self._conf = torch.ones(self.max_num_reqs, k, dtype=torch.float32, device=device)
            self._scr_lead = torch.zeros(
                self.max_num_reqs, self._n_blocks, dtype=torch.int64, device=device
            )
            self._scr_chain = torch.zeros_like(self._scr_lead)
            self._fill = torch.full((self.max_num_reqs, k), -1, dtype=torch.int32,
                                    device=device)
            # Last step's lookup slots per request, graded by this step's verdict.
            self._prev_from = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
            self._prev_n = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
            # rows, lead rows, chain rows, lookup tokens, DSpark kept drafts,
            # lookup tokens accepted, lookup tokens verified
            self._stats = torch.zeros(7, dtype=torch.int64, device=device)
            self._calls = 0
            logger.info(
                "Inkling2 dual drafters: DSpark + suffix lookup (lead on a %d+ token repeat "
                "that agrees with DSpark's first draft, else chained after DSpark's drafts "
                "sure to %.2f; matches of %d-%d tokens)",
                _LOOKUP_LEAD, _LOOKUP_MIN_P, _LOOKUP_MIN, _LOOKUP_MAX)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        target = target_model.get_language_model()
        model.attach_target(target)  # folds the Markov head from the bf16 LM head
        # Then draft from the target's int8 head (half the bytes per draft step) and free the
        # 0.9 GB bf16 copy the draft no longer needs (more KV cache). VLLM_INKLING2_DRAFT_HEAD=bf16
        # keeps the reference path.
        hq = getattr(target, "head_q", None)
        if hq is not None and os.environ.get("VLLM_INKLING2_DRAFT_HEAD", "int8") == "int8":
            model.head_q, model.head_s = hq, target.head_s
            w = target.lm_head.weight
            w.data = torch.empty(0, dtype=w.dtype, device=w.device)
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
        return model

    def prepare_context_anchor(self, input_batch, num_rejected: torch.Tensor) -> None:
        """The Markov head's recurrent input: each request's feature row at its last
        accepted target position."""
        num_reqs = input_batch.num_reqs
        query_start_loc = input_batch.query_start_loc
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n_spec):
            state, bias = self.model.markov_step(state, prev, context[i])
            if self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, *args, **kwargs):
        drafts = super().propose(
            input_batch, attn_metadata, slot_mappings, last_hidden_states,
            aux_hidden_states, num_sampled, num_rejected, last_sampled,
            next_prefill_tokens, temperature, seeds, *args, **kwargs)
        if (self._lookup and not kwargs.get("dummy_run", False)
                and self.draft_logits is not None and self.req_states is not None
                and input_batch.num_reqs > 0):
            self._lookup_chain(input_batch, num_sampled, temperature)
        return drafts

    @torch.inference_mode()
    def _lookup_chain(self, input_batch, num_sampled: torch.Tensor,
                      temperature: torch.Tensor) -> None:
        """Second drafter: the suffix lookup chained after DSpark's sure drafts."""
        num_reqs = input_batch.num_reqs
        k = self.num_speculative_steps
        k_po2 = triton.next_power_of_2(k)
        idx = input_batch.idx_mapping
        drafts = self.draft_tokens
        logits = self.draft_logits
        vocab = logits.shape[-1]
        ids = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        _draft_conf_kernel[(num_reqs, k - 1)](
            logits, logits.stride(0), logits.stride(1), idx, temperature,
            drafts, drafts.stride(0), self._conf, vocab, K=k, BLOCK=_VOCAB_BLOCK,
            num_warps=8)
        _chain_scan_kernel[(num_reqs, self._n_blocks)](
            ids, ids.stride(0), idx, total_len, num_sampled, drafts, drafts.stride(0),
            self._conf, _LOOKUP_MIN_P, self._scr_lead, self._scr_chain,
            self._scr_lead.stride(0), self._max_model_len, K=k, K_PO2=k_po2,
            MIN_N=_LOOKUP_MIN, MAX_N=_LOOKUP_MAX,
            MAX_N_PO2=triton.next_power_of_2(_LOOKUP_MAX), BLOCK_L=_SCAN_BLOCK,
            num_warps=4)
        _chain_finalize_kernel[(num_reqs,)](
            ids, ids.stride(0), idx, total_len, num_sampled, drafts, drafts.stride(0),
            self._conf, _LOOKUP_MIN_P, self._scr_lead, self._scr_chain,
            self._scr_lead.stride(0), self._fill, self._prev_from, self._prev_n,
            self._stats, self._max_model_len, self._n_blocks, _LOOKUP_LEAD,
            K=k, K_PO2=k_po2, N_BLOCKS_PO2=triton.next_power_of_2(self._n_blocks),
            num_warps=4)
        _one_hot_kernel[(num_reqs, k, triton.cdiv(vocab, _VOCAB_BLOCK))](
            logits, logits.stride(0), logits.stride(1), idx, self._fill, vocab,
            K=k, BLOCK=_VOCAB_BLOCK, num_warps=4)
        self._calls += 1
        if _LOOKUP_LOG_EVERY and self._calls % _LOOKUP_LOG_EVERY == 0:
            rows, lead, chain, toks, kept, acc, graded = self._stats.tolist()
            self._stats.zero_()
            logger.info(
                "Inkling2 dual drafters, last %d steps: %d drafting rows; DSpark kept %.2f "
                "drafts/row; lookup led %d rows, chained %d rows, %d lookup tokens; "
                "lookup tokens accepted %d of %d (%.1f%%)", _LOOKUP_LOG_EVERY, rows,
                kept / max(rows, 1), lead, chain, toks, acc, graded,
                100.0 * acc / max(graded, 1))
