## Description

<!-- What does this PR do? Briefly describe the changes and why they are needed. -->

## Related Issues

<!-- Link related issues here. Use "Fixes #123" when this PR closes an issue. -->

## Change Type

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor
- [ ] Documentation
- [ ] Test
- [ ] Build or CI

## Pull Request Checklist

Thank you for contributing to xLLM. Before requesting review, please make sure the following items are complete.

### PR Title and Commit Messages

- [ ] The PR title and each commit message follow the xLLM commit format: `<type>: <subject>`.

> Allowed types: feat, bugfix, docs, test, refactor, chore, style, revert, perf, model, build, release.
> The subject should use clear English, start with a verb, include at least 4 words, and end with `.`.

### Pre-commit Checks

- [ ] I have installed `pre-commit` by running `pip install pre-commit` or an equivalent command.
- [ ] I have installed the hooks with `pre-commit install`.
- [ ] I have run `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

### Self Review

- [ ] I have self-reviewed the code according to `.agents/skills/code-review/references/custom-code-style.md`, especially code written or assisted by AI.
- [ ] I have rebased this PR onto the latest `main` branch.

### Build and Test Coverage

- [ ] Tests have been added or updated as needed.
- [ ] CUDA: `python setup.py build test` has passed on a CUDA machine.
- [ ] NPU: `python setup.py build test` has passed on an NPU machine.
- [ ] MLU: `python setup.py build test` has passed on an MLU machine.

## Reviewer Notes

<!-- Optional: anything reviewers should focus on, known risks, follow-ups, or validation gaps. -->
