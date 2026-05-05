# Sagittarius VLM Agent

Function-calling VLM agent for prompt-driven Sagittarius tabletop tasks.

The agent does not ask the VLM to generate Python code. Each step asks the model
for one JSON tool call, executes that tool locally, records the result in memory,
and repeats until `finish_task`.

## MVP Tools

- `capture_image`
- `detect_object`
- `detect_objects`
- `pick_object`
- `place_object`
- `open_gripper`
- `close_gripper`
- `verify_grasp`
- `finish_task`

`execute_motion` is implemented but disabled by default. Set
`agent/enable_motion_tools: true` to expose it to the planner.

## Execution Modes

`agent.execution_mode` defaults to `auto`.

- `auto`: simple pick / pick-and-place prompts use a deterministic function
  sequence; other prompts fall back to the agent loop.
- `template`: only deterministic simple templates are allowed.
- `agent`: always ask the VLM planner for the next tool call.

## Example

```bash
source devel/setup.bash
export OPENAI_API_KEY=...
roslaunch sagittarius_vlm_agent vlm_agent.launch prompt:="pick up the green block and place it on the red block"
```

The launch file starts MoveIt, the existing `sgr_ctrl` action server, USB camera,
and the new `vlm_agent_executor`.

## Notes

- Object placement uses calibrated pixel-to-XY conversion and configured fixed Z
  heights. Without a depth camera, placing an object "on" another object is a
  fixed-height approximation.
- Agent artifacts are saved under `logs/vlm_agent/<timestamp>` by default,
  including `memory.json`, captured images, annotated images, and `run.log`.
- ROS node logs are directed to `logs/ros` by the launch file.
