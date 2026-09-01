#!/usr/bin/env bash
# Minimizes this bot's earlier review comments so only the current run shows.
# minimizeComment takes a GraphQL node id; the REST id silently matched nothing (#2002).
set -uo pipefail

REPO="${REPO:?REPO is required}"
PR_NUMBER="${PR_NUMBER:?PR_NUMBER is required}"
BOT_LOGIN="${BOT_LOGIN:-github-actions[bot]}"
REPLY_BODY="Resolved, superseded by a newer review run."

# A failed listing must not read as "nothing to resolve": that is the same
# silent success this script exists to remove.
comments=$(gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments" --paginate \
  --jq ".[] | select(.user.login == \"${BOT_LOGIN}\") | \"\(.id) \(.node_id)\"") || {
  echo "::error::could not list existing review comments" >&2
  exit 1
}

if [ -z "$comments" ]; then
  echo "No previous comments to resolve."
  exit 0
fi

total=0
minimized=0
while read -r rest_id node_id; do
  [ -n "$rest_id" ] || continue
  total=$((total + 1))
  gh api "repos/${REPO}/pulls/comments/${rest_id}/replies" \
    --method POST -f body="$REPLY_BODY" >/dev/null 2>&1 || true
  if gh api graphql -f query="
    mutation {
      minimizeComment(input: {subjectId: \"${node_id}\", classifier: OUTDATED}) {
        minimizedComment { isMinimized minimizedReason }
      }
    }" >/dev/null 2>&1; then
    minimized=$((minimized + 1))
  else
    echo "::warning::could not minimize comment ${rest_id} (node ${node_id})"
  fi
done <<<"$comments"

# One failure is a flake. None succeeding is the #2002 signature: the mutation
# takes an id the API never resolves, so it reports success while doing nothing.
if [ "$minimized" -eq 0 ]; then
  echo "::error::no comment could be minimized (${total} tried)" >&2
  exit 1
fi

echo "Resolved ${minimized} previous comment(s)."
