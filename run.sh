#!/bin/sh
# Start the bench with a credential the transcript never sees.
#
# The key lives in the macOS Keychain, not in a file, not in your shell
# history, and not in anything an agent reads. This script substitutes it into
# the environment of one process and nothing else.
#
#   store it once (prompts, echoes nothing):
#     security add-generic-password -s anthropic-api-key -a "$USER" -w
#
#   then just:
#     ./run.sh [port]
#
# Prefer `ant auth login` if you have the CLI — that stores a short-lived
# OAuth token instead of a static key, and this script is unnecessary: the
# SDK finds the profile on its own.
set -e
PORT="${1:-8770}"

if [ -z "$ANTHROPIC_API_KEY" ]; then
  if KEY=$(security find-generic-password -s anthropic-api-key -w 2>/dev/null); then
    ANTHROPIC_API_KEY="$KEY"
    export ANTHROPIC_API_KEY
    unset KEY
    echo "  credential: macOS Keychain (anthropic-api-key)"
  else
    echo "  credential: none — the LLM stage will skip and the keyword"
    echo "              scaffold will carry every run. Store one with:"
    echo "                security add-generic-password -s anthropic-api-key -a \"\$USER\" -w"
  fi
else
  echo "  credential: ANTHROPIC_API_KEY from the environment"
fi

exec python3 server.py "$PORT"
