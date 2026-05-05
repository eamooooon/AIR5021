# sagittarius_vlm_task_router

Unified prompt-driven task entry for the final project.

Current supported tasks:

- `clean_desk`
- `wave_hand`
- `nod`
- `draw_circle`
- `spin_wrist`

Launch:

```bash
roslaunch sagittarius_vlm_task_router vlm_task.launch \
  prompt:="wave your hand twice" \
  api_base:="https://api.openai.com/v1" \
  openai_api_key:="YOUR_KEY" \
  model:="gpt-5.4"
```
