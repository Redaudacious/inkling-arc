# inkling-arc

Inkling 2 Mini (Thinking Machines, 26.4B MoE, 3.4B active) needed to run well on the Intel Arc Pro B70 (32 GB).
This is what was worked on in vLLM-XPU to get there. Written with Anthropic's Claude models.

## Numbers

One Arc Pro B70, 23–24 Nov 2026.

| | |
|---|---|
| Output, 1 chat | 85 tok/s |
| Output, 4 chats | 179 tok/s total |
| Prompt reading | 4,137 tok/s at 32K, 3,515 tok/s at 96K |
| Context | 4 × 131,072 or 1 × 524,288 tokens |
| KV cache pool | 329,808 tokens |
| Drafter | 72% first token accepted, 2.46 tokens per step |
| 20-case eval (instructions, maths, reasoning, tool calls, code, image, needles at 31.6K / 107.6K) | 20/20 |
| IFEval, prompt strict / instruction loose | 94.6% / 97.1% |
| AIME 2025 | 63.3% |
| MRCR v2 8-needle, 64K–120K | 0.766 |
| Needle at 495,084 tokens | found |

## What was worked on

- Experts: the shipped NVFP4/MXFP8 codes and scales kept as they are, re-ordered for XMX; decoded in registers, fp32 accumulate. No FP8 maths.
- Dense projections and LM head: int8 weights, one scale per row.
- KV cache: K int8 + V int4 on the global-attention layers.
- Global attention: ESIMD XMX kernel for decode, draft verify and prefill; oneDNN for keys beyond the 1,024 window.
- Drafting: the model's DSpark drafter (7 tokens) plus a suffix lookup on the GPU.
- Prefix caching with the short-conv state.
- Stop words matched only in answer text.

## Install

```sh
git clone https://github.com/arcitech-psp/inkling-arc
cd inkling-arc
./inkling-arc install
```

Linux, Arc Pro B70, podman, about 70 GB of disk. The first build takes 1–2 hours. Serves the OpenAI API at
`http://127.0.0.1:8000/v1`.

| Command | |
|---|---|
| `serve multi` / `serve single` | 4 × 131,072 / 1 × 524,288 tokens |
| `test quick` | kernel tests, stop words, 20-case eval |
| `bench` | output speed, 1 and 4 chats |
| `doctor`, `build`, `fetch`, `pack`, `stop` | setup steps |

Heretic fine-tune: `INKLING_ARC_MODEL=arcitech-psp/Inkling-2-Mini-Heretic-NVFP4-XPU ./inkling-arc install`

## Limits

- Tested on one Arc Pro B70 only.
- Decode is near the card's memory bandwidth (82–89% of 608 GB/s).
- The 4 chats share one KV pool.
- Reasoning cannot be turned fully off; `reasoning_effort` sets how much.

## Licence

Apache-2.0. Builds on Inkling 2 Mini (Thinking Machines, Apache-2.0), vLLM (Apache-2.0), F3zz1k's Inkling 2 XPU port,
the Lumnus B70 kit (with MIT material by wu1ff), mgaruccio's vLLM PR #69186, SergiioB's ESIMD attention kernel (MIT),
Danmoreng's hybrid prefill (MIT) and oneDNN (Apache-2.0). Heretic weights: DavidAU (Apache-2.0), from yanauto. Notices in
`NOTICE` and `licenses/`.
