#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${ROOT_DIR}/prompt.txt"

if [[ "$#" -gt 0 ]]; then
  PROMPT="$*"
elif [[ -f "$PROMPT_FILE" ]]; then
  PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
else
  PROMPT="pick up the green block"
fi

cd "$ROOT_DIR"

if [[ ! -f devel/setup.bash ]]; then
  echo "Missing devel/setup.bash. Run catkin_make first." >&2
  exit 1
fi

source devel/setup.bash
mkdir -p logs/ros

echo "Starting Sagittarius VLM agent"
echo "Prompt: ${PROMPT}"

exec roslaunch sagittarius_vlm_agent vlm_agent.launch \
  prompt:="${PROMPT}"
