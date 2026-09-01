#!/usr/bin/env bash
# Runs OpenCodeReview and exports exit_code + comment count as step outputs.
# code-review.yml runs the base branch's copy of this script, see #2004.
set -u

BASE_REF="${BASE_REF:?BASE_REF is required}"
WORK_DIR="${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"

set +e
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# OCR reads its config from ~/.opencodereview/config.json, which carries the
# meta-router proxy and its token. Never override OCR_LLM_TOKEN from a
# missing API_KEY: that failed every PR.
/usr/local/bin/ocr review --from "origin/$BASE_REF" --to HEAD --format json \
  --audience agent --repo "$WORK_DIR" \
  > "$WORK_DIR/ocr-output.json" 2>"$WORK_DIR/ocr-stderr.log"
EXIT_CODE=$?
echo "exit_code=$EXIT_CODE" >> "$GITHUB_OUTPUT"
if [ $EXIT_CODE -ne 0 ]; then
  echo "::error::OpenCodeReview failed with exit code $EXIT_CODE"
  echo "=== STDERR ==="
  cat "$WORK_DIR/ocr-stderr.log"
  echo "=== STDOUT ==="
  cat "$WORK_DIR/ocr-output.json"
  exit 1
fi

COMMENTS=$(jq '.summary.comments // 0' "$WORK_DIR/ocr-output.json" 2>/dev/null)
if [ -z "$COMMENTS" ] || [ "$COMMENTS" = "null" ]; then
  echo "::error::OpenCodeReview produced invalid JSON output"
  echo "=== STDOUT ==="
  cat "$WORK_DIR/ocr-output.json"
  echo "=== STDERR ==="
  cat "$WORK_DIR/ocr-stderr.log"
  exit 1
fi
echo "comments=$COMMENTS" >> "$GITHUB_OUTPUT"
echo "OCR found $COMMENTS comment(s) in this run"
