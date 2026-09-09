# xLLM Coding Agent Instructions

## Directory Structure

```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

## Code Style Guide

* Before editing, creating, refactoring, or reviewing any file under `xllm/`, you **MUST** read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).
* The file above is a **required instruction file**, not an optional reference. Do not skip reading it.
* After editing any C++ file, run clang-format 20.1.6 with the repo `.clang-format` on every touched file before finishing. Do not leave formatting for the user to remind you.
* Apply the rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) to **both code generation and code review**.
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* If [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) specifies a rule, that rule takes precedence over the Google C++/Python Style Guide.
* Use the Google C++/Python Style Guide only for cases not specified in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).

## Review Instructions

* For code review tasks, you **MUST** first read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Then read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and apply it during the review.
* Review code changes for quality, security, performance, correctness, and maintainability following the project-specific standards.
* Review code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Use the review workflow, checklist, severity rules, and output format defined in [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Apply the Google C++/Python Style Guide only when the project-specific style guide does not define the rule.
* Focus the review on the requested diff or changed files. Do not comment on unrelated code.