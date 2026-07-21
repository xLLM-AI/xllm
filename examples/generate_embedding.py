# ASCEND_RT_VISIBLE_DEVICES=0 python examples/generate_embedding.py --model='/path/models/Qwen3-8B' --runner pooling
# ASCEND_RT_VISIBLE_DEVICES=0,1 python generate_embedding.py --model='/path/models/Qwen3-8B'

from xllm import ArgumentParser, LLM, PoolingParams

# Create an embedding LLM.
parser = ArgumentParser()
args = parser.parse_args()
llm = LLM(**vars(args))

# Create pooling params.
pooling_params = PoolingParams()

inputs = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.embed(inputs, pooling_params=pooling_params)

# Print the outputs.
for i, output in enumerate(outputs):
    input_str = output.prompt
    generated_embedding = output.outputs.embedding
    print(f"Input: {input_str!r}, Generated embedding: {generated_embedding!r}")

llm.finish()

