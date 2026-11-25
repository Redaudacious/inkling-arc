# inkling-arc

Thinking Machines' Inkling 2 Mini (26.4B MoE, 3.4B active) served with vLLM-XPU on one Intel Arc Pro B70 (32 GB).

Written with Anthropic's Claude models. Tested on one Arc Pro B70 only.

## Goal

Run Inkling 2 Mini on a single 32 GB Arc card without re-quantizing its experts, fast enough for agent work:
4 chats × 131,072 tokens, or 1 chat × 524,288 tokens.

## Numbers

One Arc Pro B70, `serve multi`, 23–24 Nov 2026.

| What | Result |
|---|---|
| Output, 1 chat | 85 tok/s (median, 600-token replies) |
| Output, 4 chats at once | 179 tok/s total |
| Prompt reading, 32K tokens | 7.79 s (4,137 tok/s) |
| Prompt reading, 96K tokens | 27.5 s (3,515 tok/s) |
| Output at ~95K context, 1 chat | 59–91 tok/s |
| KV cache pool | 329,808 tokens |
| Drafter: first draft token accepted | 72% |
| Drafter: accepted tokens per step | 2.46 of 7 |
| New request while another chat reads a 32K prompt | first token after 1.75 s |
| Restart until ready (kernel caches kept) | ~120 s |
| `serve single`: needle at 495,084 tokens | found, 301 s end to end |

Output speed moves about ±8% with how many drafted tokens are accepted, which depends on the text. The drafter figures
are the live server's counters (27,621 drafts, 24 Nov).

## Quality checks

| Check | Result | Checkpoint |
|---|---|---|
| 20-case eval (`tools/inkeval`): instructions, maths, reasoning, tool calls, Rust code compiled and run, needles at 31,629 and 107,579 tokens, an image | 20/20 | heretic, 24 Nov |
| Same eval | 20/20 in most runs, 19/20 in some: one coding case can use up its 2,000-token budget reasoning at temperature 0 | original, 23–24 Nov |
| IFEval, prompt strict / instruction loose (lm-eval 0.4.11) | 94.6% / 97.1% | original |
| AIME 2025, one sample per problem | 63.3% (19/30) | original |
| MRCR v2 8-needle, 64K–120K, 20 items, official grader | 0.766 (15/20 exact) | original |
| Needle at 495,084 tokens (`serve single`) | found | original |

"Original" is `thinkingmachines/Inkling-2-Mini`; "heretic" is
[arcitech-psp/Inkling-2-Mini-Heretic-NVFP4-XPU](https://huggingface.co/arcitech-psp/Inkling-2-Mini-Heretic-NVFP4-XPU).
Not run here: LongBench v2 and GraphWalks (several hours each on one card).

## Install

```sh
git clone https://github.com/arcitech-psp/inkling-arc
cd inkling-arc
./inkling-arc install
```

`install` runs doctor, build, fetch, pack, `serve multi` and `test quick`. The server speaks the OpenAI API at
`http://127.0.0.1:8000/v1`, model `inkling-2-mini`, with tool calls, images and `reasoning_effort`.

DavidAU's heretic fine-tune instead of the original weights:

```sh
INKLING_ARC_MODEL=arcitech-psp/Inkling-2-Mini-Heretic-NVFP4-XPU INKLING_ARC_DATA=./data-heretic ./inkling-arc install
```

Needs Linux, an Arc Pro B70 (PCI 8086:e223) with a working render node, podman, and about 70 GB of disk (images 25 GB,
checkpoint 20 GB, packs 15 GB). The first build compiles the base image on the CPU: 1–2 hours.

| Command | Does |
|---|---|
| `doctor` | checks GPU, container engine, disk, images, files |
| `build [--skip-base]` | base image, builder image, kernels, Rust tools, serving image |
| `fetch` | checkpoint from Hugging Face (`INKLING_ARC_MODEL`, default `thinkingmachines/Inkling-2-Mini`) |
| `pack` | expert packs on the GPU: the same codes and scales in XMX tile order (76 s measured) |
| `serve multi` / `serve single` | 4 × 131,072 tokens / 1 × 524,288 tokens |
| `test quick` / `test full` | kernel tests against reference, stop-word test, 20-case eval / plus a needle at ~491K tokens (needs `serve single`) |
| `bench` | output speed with 1 and 4 chats |
| `stop` | stops the server |

## What is changed from stock vLLM-XPU

The base is F3zz1k's Inkling 2 port of vLLM on Lumnus's B70 kit. The changes are in `engine/overlay` and `kernels/`.

- **Experts:** the shipped NVFP4 and MXFP8 codes and scales are kept unchanged, only re-ordered into XMX (DPAS) tiles.
  The kernel decodes them to fp16 in registers and accumulates in fp32. Checked per layer against an f64 reference:
  relative error ~6e-8. The B70 has no FP4/FP8 maths, and nothing runs in FP8.
- **Dense projections and LM head:** int8 weights, one scale per row (W8A8). This is the one lossy weight step.
- **KV cache:** K int8 + V int4 (K8/V4) on the 8 global-attention visits. The local layers (1,024-token window) stay bf16.
- **Global attention:** an ESIMD XMX kernel for decode, draft verification and prefill. At prefill, keys more than
  1,024 positions back go through oneDNN's fused attention and are merged in. Within 0.2% of the reference.
- **Drafting:** the model's own DSpark drafter (7 tokens) plus a suffix lookup on the GPU that continues copied text.
- **Prefix caching** works with the short-conv state: an agent turn that extends the previous prompt starts in under a second.
- **Stop words** are matched only in answer text, never inside reasoning or a tool call.
- **`serve multi`:** while other chats run, a long prompt reads at most 2,048 tokens per step.

## Tests

`test quick` runs the attention kernels against reference (decode, verify, prefill, mixed batches, shuffled pages),
the far-key prefill against reference, the stop-word test and the 20-case eval.

Results at release (24 Nov 2026):

- 20-case eval: 20/20, heretic checkpoint, on the running server.
- Every Rust tool builds from a fresh clone.
- Kernel tests: passed on the from-scratch image build (23 Nov) and after each kernel change since (expert test
  relative error ~6e-8 on 24 Nov). They were not re-run on this exact commit: they need the card free, and it was serving.

## Limits

- Tested on one Arc Pro B70 only. Other Intel GPUs can be tried with `INKLING_ARC_ANY_INTEL=1`; untested.
- Decode is close to the card's memory bandwidth: the expert kernels read at 82–89% of 608 GB/s. More speed needs fewer
  bytes per token or more accepted drafts.
- The 4 seats share one 329,808-token pool. vLLM estimates 2.52 requests at the full 131,072 tokens at the same time.
- The short-conv state is saved once per prompt, at its last 64-token boundary. A request that shares a long context
  but differs before that point reads the context again (regenerate, edited message). Agent turns are not affected.
- Inkling cannot turn reasoning fully off. `reasoning_effort` sets how much; the chat template ignores `enable_thinking`.
- Output at temperature 0 is not bit-identical from run to run.

## DSH and Atelier

`integrations/dsh-provider.yml` is a provider block for DSH-based harnesses such as Atelier: `reasoning_effort` medium,
temperature 1.0, 131,072-token window, images on, long cache retention.

## Credits and licences

inkling-arc is Apache-2.0 (`LICENSE`). It builds on:

- Inkling 2 Mini, Thinking Machines (Apache-2.0)
- vLLM (Apache-2.0) and F3zz1k's Inkling 2 XPU port, `F3zz1k/vllm` branch `inkling2-b70` (Apache-2.0)
- Lumnus `b70-flash-next` kit (Apache-2.0; it includes MIT material by wu1ff)
- vLLM PR #69186 by mgaruccio (fp32 LM head on XPU)
- SergiioB `intel-arc-pro-b70-inference-cookbook` (MIT): the ESIMD attention kernel ours is ported from
- Danmoreng `intel-b70-inkling2-mini-vllm` (MIT): the hybrid prefill idea and oneDNN Graph attention with statistics
- oneDNN (Apache-2.0)
- Heretic checkpoint: DavidAU's fine-tune (Apache-2.0), from yanauto's heretic model

Details and the MIT notices are in `NOTICE` and `licenses/`.
