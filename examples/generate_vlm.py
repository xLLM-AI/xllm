# python generate_vlm.py --model /path/to/Qwen2.5-VL-7B-Instruct/ --max_seqs_per_batch 4 --devices='npu:0' --enable_shm

from xllm import ArgumentParser, SamplingParams
from xllm import LLM
# from xllm import VLM
import base64
import os

def encode_image_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"not found image: {file_path}")
    with open(file_path, "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
    return result

parser = ArgumentParser()
args = parser.parse_args()
# vlm = VLM(**vars(args))
vlm = LLM(**vars(args))


image_1 = "./images/3.jpg"
image_2 = "./images/4.jpg"

# image_base64_1 = encode_image_from_file(image_1)
# image_base64_2 = encode_image_from_file(image_2)

# image_1 = f"data:image/jpeg;base64,{image_base64_1}"
# image_2 = f"data:image/jpeg;base64,{image_base64_2}"

requests = [
    {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "请描述这张图片。<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {
            "image": image_1,
        },
    },
    {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "请对比这两张图片的主要区别。<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {
            "image": [image_1, image_2],
        },
    },
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

outputs = vlm.generate(
    requests,
    sampling_params=sampling_params
)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

vlm.finish()
