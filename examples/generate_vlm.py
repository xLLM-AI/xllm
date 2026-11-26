# ASCEND_RT_VISIBLE_DEVICES=0 python generate_vlm.py --model /path/to/Qwen2.5-VL-7B-Instruct/ --max_seqs_per_batch 4 --enable_shm

# from xllm import VLM

from xllm import LLM, ArgumentParser, SamplingParams

parser = ArgumentParser()
args = parser.parse_args()
# vlm = VLM(**vars(args))
vlm = LLM(**vars(args))


image_1 = "./images/3.jpg"
image_2 = "./images/4.jpg"

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

outputs = vlm.generate(requests, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

vlm.finish()
