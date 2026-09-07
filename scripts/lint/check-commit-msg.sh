#!/usr/bin/env bash
# Enforce commit message subject format.
#
# Format: <type>: <description>.
#   - type must be lowercase and one of the whitelist
#   - ": " (colon + single space) follows the type
#   - description has >= 4 whitespace-separated tokens
#   - subject ends with a literal "."
#
# Uppercase identifiers (class names / file names / acronyms) are allowed
# inside tokens. Only the first line (subject) is checked; body is free-form.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <commit-msg-file>" >&2
  exit 2
fi

msg_file="$1"
# Strip a trailing CR so CRLF-formatted commit-msg files (Windows / some GUIs)
# don't fail the "ends with ." check.
subject="$(head -n1 "$msg_file" | tr -d '\r')"

# Skip git-generated messages that authors don't hand-write.
case "$subject" in
  "Merge "*|"Revert "*|"fixup! "*|"squash! "*|"amend! "*|"Initial commit")
    exit 0
    ;;
esac

regex='^(feat|bugfix|docs|test|refactor|chore|style|revert|perf|model|build|skills): (\S+ ){3,}\S+\.$'

if [[ "$subject" =~ $regex ]]; then
  exit 0
fi

cat >&2 <<'EOF'
------------------------------------------------------------
Commit message subject rejected.

Format: <type>: <description>.
  - type in {feat|bugfix|docs|test|refactor|chore|style|revert|perf|model|build|skills}
  - ": " (colon + single space) after type
  - description has >= 4 whitespace-separated tokens
  - subject ends with a literal "."

Good:
  feat: enable int8 sfa main kv cache for glm.
  refactor: extract LLMRequestFactory from LLMMaster in distributed runtime.

Bad:
  Feat: xxx yyy zzz www.        (type must be lowercase)
  feat:enable xxx yyy zzz.      (missing space after colon)
  feat: enable xxx yyy zzz      (missing trailing period)
  chore: update CODEOWNERS.     (fewer than 4 tokens)
  feat: foo  bar baz qux.       (tokens must be separated by a single space; no double space or tab)

Fix:
  git commit --amend              # edit last commit
  git rebase -i <base>            # edit multiple commits
------------------------------------------------------------
EOF

echo "Rejected subject: $subject" >&2
exit 1

