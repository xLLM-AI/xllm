# python examples/generate_beam_search.py --model='/path/models/Qwen2-7B-Instruct' --devices='npu:0'
# python generate_beam_search.py --model='/path/models/Qwen2-7B-Instruct' --devices='npu:0,npu:1'

from xllm import ArgumentParser, BeamSearchParams, LLM

# Create an LLM.
parser = ArgumentParser()
llm = LLM(**vars(parser.parse_args()))

beam_search_params = BeamSearchParams(
    beam_width=2,
    # Top-k candidate count for beam expansion. Defaults to beam_width.
    top_logprobs=4,
    max_tokens=20,
)

outputs = llm.beam_search(
    [
     {"prompt": "Hello, my name is "},
     {"prompt": "The president of the United States is "},
     {"prompt": "The capital of France is "},
     {"prompt": "The future of AI is "}
    ],
    params=beam_search_params,
)

for output in outputs:
    generated_text = output.sequences[0].text
    print(f"Generated text: {generated_text!r}")

llm.finish()
