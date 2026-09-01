#!/usr/bin/env bash
# Posts OCR review comments as PR review comments; falls back to one issue
# comment when every post fails. code-review.yml runs the base branch's copy
# of this script, see #2004.
set -u

REPO="${REPO:?REPO is required}"
PR_NUMBER="${PR_NUMBER:?PR_NUMBER is required}"
WORK_DIR="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"

set +e
HEAD_SHA=$(git rev-parse HEAD)
POSTED=0
FAILED=0
while IFS= read -r comment; do
  PATH_VAL=$(echo "$comment" | jq -r '.path')
  START_LINE=$(echo "$comment" | jq -r '.start_line')
  END_LINE=$(echo "$comment" | jq -r '.end_line')
  CONTENT=$(echo "$comment" | jq -r '.content')
  SUGGESTION=$(echo "$comment" | jq -r '.suggestion_code // empty')
  if [ -n "$SUGGESTION" ]; then
    # shellcheck disable=SC2016
    BODY=$(printf '%s\n\n```suggestion\n%s\n```' "$CONTENT" "$SUGGESTION")
  else
    BODY="$CONTENT"
  fi
  if gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments" \
    -f body="$BODY" \
    -f path="$PATH_VAL" \
    -F start_line="$START_LINE" \
    -F line="$END_LINE" \
    -f commit_id="$HEAD_SHA" \
    -H "Accept: application/vnd.github+json" 2>/dev/null; then
    POSTED=$((POSTED + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done < <(jq -c '.comments[]' "$WORK_DIR/ocr-output.json")
if [ "$FAILED" -gt 0 ] && [ "$POSTED" -eq 0 ]; then
  # The fallback used $REVIEW_COMMENT_COUNT, which nothing ever defined, so
  # the summary read "found  issue(s)". Count what was attempted.
  TOTAL=$((POSTED + FAILED))
  # shellcheck disable=SC2016
  SUMMARY=$(
    jq -r \
      '.comments[] | "**\(.path):\(.start_line)-\(.end_line)**\n\(.content)\n"' \
      "$WORK_DIR/ocr-output.json"
  )
  BODY=$(
    printf '## OpenCodeReview found %s issue(s)\n\n%s' \
      "$TOTAL" \
      "$SUMMARY"
  )
  gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" \
    -f body="$BODY" \
    2>/dev/null
fi
exit 0
