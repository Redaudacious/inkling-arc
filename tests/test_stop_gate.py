# Stop-gate test for the patched vllm/v1/engine/detokenizer.py (CPU only, real Inkling tokenizer, no model).
# Cases: stop word inside reasoning is ignored and found in the answer; per-token vs 8-token steps; a stop word split
# across the reasoning/answer boundary never matches; tool-call JSON is not cut; an ungated raw prompt behaves as upstream.
import os, sys, types
os.environ["VLLM_STOP_GATE"] = ("arm=<|content_text|>;disarm=<|content_thinking|>,<|content_invoke_tool_json|>,"
    "<|content_invoke_tool_text|>,<|content_xml|>,<|content_image|>,<|content_tool_error|>,<|content_model_end_sampling|>,"
    "<|end_message|>,<|message_model|>,<|message_user|>,<|message_system|>,<|message_tool|>")
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.engine.detokenizer import IncrementalDetokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
V = tok.get_vocab()
def ids(text): return tok.encode(text, add_special_tokens=False)
def sp(name): return [V[name]]

PROMPT = sp("<|message_user|>") + sp("<|content_text|>") + ids("Find x.") + sp("<|end_message|>") + sp("<|message_model|>")

def run(prompt, out_ids, stop, step):
    req = types.SimpleNamespace(request_id="t", prompt_token_ids=prompt,
                                sampling_params=SamplingParams(stop=stop, skip_special_tokens=True, max_tokens=4096))
    d = IncrementalDetokenizer.from_new_request(tok, req)
    hit = None
    for i in range(0, len(out_ids), step):
        hit = d.update(out_ids[i:i + step], False)
        if hit:
            break
    return hit, d.output_text, d

fails = 0
def check(name, cond, info):
    global fails
    print(("PASS " if cond else "FAIL ") + name + "  " + repr(info)[:160])
    fails += 0 if cond else 1

think = sp("<|content_thinking|>") + ids("We need answer math. Question: what base? Let's think.") + sp("<|end_message|>")
answer = sp("<|message_model|>") + sp("<|content_text|>") + ids("The answer is 70. Question: next one") + sp("<|end_message|>")
for step in (1, 3, 8):
    hit, text, _ = run(PROMPT, think + answer, ["Question:"], step)
    check(f"reasoning ignored, answer stopped (step {step})",
          hit == "Question:" and text.endswith("The answer is 70. ") and "We need answer math. Question: what base?" in text,
          (hit, text[-40:]))

# a stop word split across the boundary: reasoning ends "...Quest", answer starts "ion: ..."
split_think = sp("<|content_thinking|>") + ids("thinking Quest") + sp("<|end_message|>")
split_ans = sp("<|message_model|>") + sp("<|content_text|>") + ids("ion: fine and done") + sp("<|end_message|>")
for step in (1, 8):
    hit, text, _ = run(PROMPT, split_think + split_ans, ["Question:"], step)
    check(f"no match across the boundary (step {step})", hit is None, (hit, text))

# tool-call JSON with a blank line inside is not cut by a "\n\n" stop
tool = sp("<|message_model|>") + ids("get_weather") + sp("<|content_invoke_tool_json|>") + ids('{"name":"get_weather",\n\n"args":{"city":"SF"}}') + sp("<|end_message|>")
hit, text, _ = run(PROMPT, think + tool, ["\n\n"], 8)
check("tool JSON not cut", hit is None and text.endswith('"SF"}}'), (hit, text[-30:]))

# answer block closing inside the same step as the stop word is still caught
hit, text, _ = run(PROMPT, think + sp("<|message_model|>") + sp("<|content_text|>") + ids("ok Question:") + sp("<|end_message|>"), ["Question:"], 64)
check("stop found in a block that closes in the same step", hit == "Question:" and text.endswith("ok "), (hit, text[-20:]))

# raw completion (no gate token in the prompt) keeps upstream behaviour
hit, text, d = run(ids("Count: one two Question: three"), ids(" four Question: five"), ["Question:"], 1)
check("ungated raw prompt stops as upstream", hit == "Question:" and d._gate is None, (hit, text))

# include_stop_str_in_output keeps the stop word
req = types.SimpleNamespace(request_id="t2", prompt_token_ids=PROMPT,
                            sampling_params=SamplingParams(stop=["Question:"], include_stop_str_in_output=True, skip_special_tokens=True))
d = IncrementalDetokenizer.from_new_request(tok, req)
out = think + answer
for i in range(0, len(out), 8):
    hit = d.update(out[i:i + 8], False)
    if hit: break
check("include_stop_str_in_output", hit == "Question:" and d.output_text.endswith("70. Question:"), d.output_text[-24:])

print("RESULT", "ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
