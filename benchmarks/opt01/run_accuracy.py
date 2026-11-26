# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Run complete EvalScope splits with no sampling limit and isolated outputs."""

import argparse
from pathlib import Path

from evalscope import run_task
from evalscope.config import TaskConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "mm_bench"), required=True)
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--api-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    work = Path("accuracy") / args.variant / args.dataset
    if args.smoke:
        work = work.with_name(work.name + "_smoke")
    if work.exists():
        raise FileExistsError(f"Use a fresh variant to avoid mixed results: {work}")
    generation = {"temperature": 0, "top_p": 1, "max_tokens": 2048, "seed": 42}
    if args.model == "Qwen3-8B":
        generation["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    config = TaskConfig(
        model=args.model,
        api_url=args.api_url,
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=[args.dataset],
        dataset_args={"mm_bench": {"subset_list": ["cn", "en"]}} if args.dataset == "mm_bench" else {},
        limit=2 if args.smoke else None,
        eval_batch_size=32,
        generation_config=generation,
        timeout=900,
        seed=42,
        judge_strategy="rule",
        ignore_errors=False,
        work_dir=str(work),
        no_timestamp=True,
    )
    run_task(task_cfg=config)


if __name__ == "__main__":
    main()
