# ASCEND_RT_VISIBLE_DEVICES=0 python examples/sample.py --model='/path/models/Qwen3-8B'
# ASCEND_RT_VISIBLE_DEVICES=0,1 python examples/sample.py --model='/path/models/Qwen3-8B'

from xllm import ArgumentParser, LLM, RequestParams

# Create an LLM.
parser = ArgumentParser()
llm = LLM(**vars(parser.parse_args()))

# selector must be a stable single special token in the target tokenizer.
selector = "masked"

prompts = [
    f"candidate_a={selector}, candidate_b={selector}",
    f"user_feature={selector}",
]

# RequestParams can still carry generic sampling knobs.
# sample() will enforce:
# - max_tokens=1
# - n=1
# - best_of=1
# - logprobs=True
request_params_list = []
for _ in prompts:
    request_params = RequestParams()
    request_params.temperature = 0.0
    request_params.top_p = 1.0
    request_params_list.append(request_params)

outputs = llm.sample(
    prompts,
    selector=selector,
    request_params=request_params_list,
    logprobs=5,
    wait_schedule_done=True,
)

# One RequestOutput per input prompt.
# output.outputs is expanded by selector hits in that prompt.
for i, output in enumerate(outputs):
    print(f"[prompt-{i}] {output.prompt!r}")
    for sample_output in output.outputs:
        print(
            f"  sample_id={sample_output.index}, "
            f"token={sample_output.text!r}, "
            f"token_ids={sample_output.token_ids}, "
            f"finish_reason={sample_output.finish_reason}"
        )

llm.finish()
