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

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="${ROOT_DIR}/logs/launch"
ROS_LOG_DIR_RUN="${ROOT_DIR}/logs/ros/${RUN_STAMP}"
LAUNCH_LOG="${LAUNCH_LOG_DIR}/${RUN_STAMP}_vlm_agent.log"

mkdir -p "$LAUNCH_LOG_DIR" "$ROS_LOG_DIR_RUN"
ln -sfn "$LAUNCH_LOG" "${ROOT_DIR}/logs/latest_launch.log"

exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "Starting Sagittarius VLM agent"
echo "Prompt: ${PROMPT}"
echo "Launch log: ${LAUNCH_LOG}"
echo "ROS log dir: ${ROS_LOG_DIR_RUN}"

exec stdbuf -oL -eL roslaunch sagittarius_vlm_agent vlm_agent.launch \
  prompt:="${PROMPT}" \
  ros_log_dir:="${ROS_LOG_DIR_RUN}"
