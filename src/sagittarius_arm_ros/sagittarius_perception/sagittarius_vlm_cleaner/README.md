# sagittarius_vlm_cleaner

First task package for the final project proposal.

Implemented workflow:

- move once to an observation pose
- call a VLM to detect all visible tabletop objects in a single frame
- build a fixed grasp queue from that one-shot result
- continuously pick and place each object into configured place slots
- do not return to the observation pose between grasps

Launch:

```bash
roslaunch sagittarius_vlm_cleaner vlm_clean_desk.launch \
  prompt:="clean the desk by picking all visible objects" \
  api_base:="https://api.openai.com/v1" \
  openai_api_key:="YOUR_KEY" \
  model:="gpt-5.4"
```
