# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed by inkling-arc (Nov 2026) from F3zz1k/vllm inkling2-b70 @ edd722c for the Intel Arc Pro B70; see README.
import os
import sys
from abc import ABC, abstractmethod
from bisect import bisect_left

import tokenizers.decoders
from tokenizers import Tokenizer
from transformers import TokenizersBackend

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"

# inkling-arc (23 Nov 2026): stop strings never end generation inside reasoning or a tool call.
# VLLM_STOP_GATE="arm=<tok>,...;disarm=<tok>,...": a stop string is matched only in text that starts after an arm token
# and ends at the next disarm token. The state at the start of generation is the prompt's last gate token; a prompt with
# no gate token (a raw completion) is not gated. Unset = upstream behaviour. Found with lm-eval AIME 2025: Inkling wrote
# "Question:" inside its reasoning, the task's stop list ended generation after 19 tokens, the answer was empty.
_STOP_GATE_SPEC = os.environ.get("VLLM_STOP_GATE", "")
_stop_gate_cache: dict[int, tuple[frozenset[int], frozenset[int]] | None] = {}


def _stop_gate_ids(tokenizer: TokenizerLike) -> tuple[frozenset[int], frozenset[int]] | None:
    if not _STOP_GATE_SPEC:
        return None
    key = id(tokenizer)
    if key not in _stop_gate_cache:
        parts = dict(p.split("=", 1) for p in _STOP_GATE_SPEC.split(";") if "=" in p)
        vocab = tokenizer.get_vocab()

        def ids(name: str) -> frozenset[int]:
            toks = [t.strip() for t in parts.get(name, "").split(",") if t.strip()]
            missing = [t for t in toks if t not in vocab]
            if missing:
                logger.warning("VLLM_STOP_GATE: not in the vocabulary: %s", missing)
            return frozenset(vocab[t] for t in toks if t in vocab)

        arm, disarm = ids("arm"), ids("disarm")
        _stop_gate_cache[key] = (arm, disarm) if arm else None
        if arm:
            logger.info("Stop strings gated: armed by %s, disarmed by %s", sorted(arm), sorted(disarm))
    return _stop_gate_cache[key]


def uses_fast_detokenizer(tokenizer: TokenizerLike) -> bool:
    """Whether the engine detokenizes with `FastIncrementalDetokenizer`."""
    return isinstance(tokenizer, TokenizersBackend)


class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if uses_fast_detokenizer(tokenizer):
            # Fast tokenizer => use tokenizers library DecodeStream.
            d: BaseIncrementalDetokenizer = FastIncrementalDetokenizer(tokenizer, request)
        else:
            # Fall back to slow python-based incremental detokenization.
            d = SlowIncrementalDetokenizer(tokenizer, request)
        d._init_stop_gate(tokenizer, request)
        return d


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Stop gate (VLLM_STOP_GATE): None = not gated. While armed, stop strings are searched from _gate_lo; a block
        # that closes inside a step leaves its end in _gate_hi for that step's check.
        self._gate: tuple[frozenset[int], frozenset[int]] | None = None
        self._gate_armed = True
        self._gate_lo = 0
        self._gate_hi: int | None = None
        self._max_stop_len = max((len(s) for s in self.stop), default=0)

        # Generation data
        self.output_text = ""

    def _init_stop_gate(self, tokenizer: TokenizerLike, request: EngineCoreRequest) -> None:
        if not self.stop:
            return
        gate = _stop_gate_ids(tokenizer)
        prompt = request.prompt_token_ids
        if gate is None or not prompt:
            return
        arm, disarm = gate
        for tid in reversed(prompt):
            if tid in arm or tid in disarm:
                self._gate = gate
                self._gate_armed = tid in arm
                return

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        num_prev_tokens = self.num_output_tokens()
        first_new_token_index = len(self.token_ids)
        # Output text offsets at the boundaries of the new tokens.
        token_offsets = [len(self.output_text)] if self.stop else None
        gate = self._gate
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            prev_len = len(self.output_text)
            self.output_text += self.decode_next(new_token_id)
            if token_offsets is not None:
                token_offsets.append(len(self.output_text))
            if gate is not None:
                if new_token_id in gate[0]:
                    self._gate_armed, self._gate_lo, self._gate_hi = True, len(self.output_text), None
                elif new_token_id in gate[1] and self._gate_armed:
                    self._gate_armed, self._gate_hi = False, prev_len

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)
            if token_offsets is not None:
                token_offsets.append(len(self.output_text))

        # 2) Evaluate stop strings.
        stop_string = None
        if token_offsets is not None and self.num_output_tokens() > self.min_tokens:
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            stop_check_offset = token_offsets[max(self.min_tokens - num_prev_tokens, 0)]
            if gate is None:
                stop = check_stop_strings(
                    output_text=self.output_text,
                    new_char_count=len(self.output_text) - stop_check_offset,
                    stop=self.stop,
                    include_in_output=self.include_stop_str_in_output,
                )
            else:
                # Search only the new text of the current (or just closed) answer block: [start, hi), with a window
                # that reaches back at most one stop string's length, never before the block start.
                hi = len(self.output_text) if self._gate_armed else self._gate_hi
                stop = None
                if hi is not None:
                    start = max(stop_check_offset, self._gate_lo)
                    if hi > start:
                        w0 = max(self._gate_lo, start - self._max_stop_len + 1)
                        stop = check_stop_strings(
                            output_text=self.output_text[w0:hi],
                            new_char_count=hi - start,
                            stop=self.stop,
                            include_in_output=self.include_stop_str_in_output,
                        )
                    if stop is not None:
                        s, t = stop
                        t = hi if t == -1 else t + w0
                        stop = (s, -1 if t >= len(self.output_text) else t)
            self._gate_hi = None
            if stop is not None:
                stop_string, truncate_to = stop
                stop_end = len(self.output_text) if truncate_to == -1 else truncate_to
                if not self.include_stop_str_in_output:
                    stop_end += len(stop_string)
                # Drop tokens generated after the one completing the stop string
                # (possible when multiple tokens are appended in one step).
                num_kept = bisect_left(token_offsets, stop_end, lo=1)
                del self.token_ids[first_new_token_index + num_kept :]
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""
        # We return the full output text if the sequence is finished.
        # Nothing is held back while the stop gate is closed (reasoning streams without delay).
        buffer_length = 0 if finished or (self._gate is not None and not self._gate_armed) else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizersBackend, request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> str | None:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.

    When several stop strings match within the newly generated text (for
    example when speculative decoding appends multiple tokens in a single
    step), the stop string that completes earliest in the text is selected,
    so the result matches appending one token at a time. Ties are broken by
    stop-list order.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        # Prefer the stop string that completes earliest in the text.
        end = stop_index + stop_string_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    if include_in_output:
        # Truncate to end of stop string.
        if best_end >= len(output_text):
            # No truncation required.
            return best_stop_str, -1
        return best_stop_str, best_end

    # Truncate the output text to the beginning of the stop string.
    return best_stop_str, best_stop_index
