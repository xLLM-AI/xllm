# xLLM Coding Agent Instructions

## Code Style Guide

* Before editing, creating, refactoring, or reviewing any file under `xllm/`, you **MUST** read the shared rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and the guide for each language you touch: [C++](.agents/skills/code-review/references/custom-code-style-cpp.md) or [Python](.agents/skills/code-review/references/custom-code-style-python.md).
* The files above are **required instruction files**, not optional references. Do not skip reading them.
* After editing any C++ file, run clang-format 20.1.6 with the repo `.clang-format` on every touched file before finishing. Do not leave formatting for the user to remind you.
* Apply the rules in the shared guide and the matching language guide to **both code generation and code review**.
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* If the shared guide or the matching language guide specifies a rule, that rule takes precedence over the Google C++/Python Style Guide.
* Use the Google C++/Python Style Guide only for cases not specified in the shared guide or the matching language guide.

### Implementation Principles

* Check unknown inputs once at the boundary responsible for them, then pass typed values inward. Do not repeat validation already guaranteed by a schema, constructor, or internal type.
* Before adding complexity for a hypothetical or highly unlikely race or security issue, explain the concrete failure, its likelihood, and the added complexity, and get the user's agreement. Do not broaden the task for theoretical robustness.
* Keep one-off logic at its call site. Extract a helper when it is reused, isolates a complex boundary, or names a distinct concept that makes the caller clearer.
* Avoid unnecessary `try/catch` or `try/except`; follow the shared code style guide for required exception boundaries.
* Keep the main function focused on normal execution. Do not wrap simple expressions in layers of abstraction.
* Comment on subtle constraints or surprising behavior rather than restating obvious code.

## Testing and Verification

* Test real implementations, avoid copying their logic into tests, and use mocks sparingly.
* Do not create tests for reversible, low-impact changes, or tests that simply restate the implementation. Any tests you add must exercise meaningful behavior.
* Run the checks appropriate to the change. After they pass, expand or repeat them only for subsequent changes, failures, or unresolved concerns.

## Working Style

* Avoid unsolicited warnings, disclaimers, approval workflows, and safety checklists motivated only by hypothetical risks.
* Treat user requests to perform work as instructions to act, and carry the authorized task through completion. Treat new messages received while working as guidance for the active task unless the user clearly cancels it or requests an incompatible objective.

## Review Instructions

* For code review tasks, you **MUST** first read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Then read the shared rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and the matching language guide(s) ([C++](.agents/skills/code-review/references/custom-code-style-cpp.md), [Python](.agents/skills/code-review/references/custom-code-style-python.md)) and apply them during the review.
* Review code changes for quality, security, performance, correctness, and maintainability following the project-specific standards.
* Review code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Use the review workflow, checklist, severity rules, and output format defined in [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Apply the Google C++/Python Style Guide only when the project-specific style guide does not define the rule.
* Focus the review on the requested diff or changed files. Do not comment on unrelated code.
