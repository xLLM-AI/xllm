# ASCEND_RT_VISIBLE_DEVICES=0 python examples/generate.py --model='/path/models/Qwen2-7B-Instruct'
# ASCEND_RT_VISIBLE_DEVICES=0,1 python generate.py --model='/path/models/Qwen2-7B-Instruct'

from xllm import ArgumentParser, LLM, SamplingParams

# Create an LLM.
parser = ArgumentParser()
llm = LLM(**vars(parser.parse_args()))

# Create sampling params.
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=10,
)

# Generate texts from the prompts. The output is a list of RequestOutput
# objects that contain the prompt, generated text, and other information.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params=sampling_params)

# Print the outputs.
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

llm.finish()
