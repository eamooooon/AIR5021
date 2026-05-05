# Sagittarius VLM Agent Function Calling 方案

## 1. 目标

在 `src/sagittarius_arm_ros/sagittarius_perception` 下新增一套完整的 VLM Agent 功能，让用户可以通过自然语言 prompt 控制机械臂完成任务。

目标形式不是让 VLM 生成 Python 代码，而是让 VLM 每一步输出结构化 function call，例如：

```json
{
  "tool": "detect_object",
  "arguments": {
    "query": "green block"
  }
}
```

程序收到后，在本地 tool registry 中调用对应 Python 函数。这样 VLM 只负责决策和填参数，真实机械臂控制仍由本地代码完成。

## 2. 当前代码状态

当前 `sagittarius_perception` 下已有三个主要功能包：

```text
sagittarius_object_color_detector
sagittarius_vlm_cleaner
sagittarius_vlm_task_router
```

它们不依赖 `manipulate-anything`。当前真实机械臂流程使用的是：

```text
USB RGB 相机
VLM 输出 bbox / grasp_pixel
线性回归 pixel -> robot XY
固定 Z 高度
ROS action /sgr532/sgr_ctrl
MoveIt 执行机械臂动作
```

已有代码不能直接等价于 function calling agent，但很多底层能力可以复用。

## 3. 旧代码能复用什么

### 3.1 机械臂控制

复用：

```text
sagittarius_object_color_detector/nodes/sgr_ctrl.py
sagittarius_object_color_detector/action/SGRCtrl.action
```

作用：

```text
接收 XYZ/RPY 目标位姿
执行普通移动、抓取、放置、回 home/sleep
内部调用 MoveIt
内部完成夹爪开闭和抓取 payload 验证
```

这部分不建议重写。新的 agent tools 只需要继续向 `/sgr532/sgr_ctrl` 发送 `SGRCtrlGoal`。

### 3.2 相机图像获取

可复用现有 executor 中的逻辑：

```text
CvBridge
订阅 /usb_cam/image_raw
wait_for_frame()
```

来源：

```text
sagittarius_object_color_detector/nodes/vlm_grasp_executor.py
sagittarius_vlm_cleaner/nodes/clean_desk_executor.py
```

建议抽成新 tool：

```text
capture_image()
```

### 3.3 VLM 单物体检测

可复用：

```text
sagittarius_object_color_detector/nodes/vlm_grasp_planner.py
sagittarius_object_color_detector/nodes/vlm_grasp_common.py
```

它现在可以根据 prompt + image 返回：

```json
{
  "target": "green block",
  "bbox": [x_min, y_min, x_max, y_max],
  "grasp_pixel": [x, y],
  "gripper_width": 0.02,
  "yaw_deg": 0.0,
  "confidence": 0.9
}
```

建议包装成新 tool：

```text
detect_object(query)
```

### 3.4 VLM 多物体检测

可复用：

```text
sagittarius_vlm_cleaner/nodes/clean_desk_planner.py
sagittarius_vlm_cleaner/nodes/clean_desk_common.py
```

它可以根据 prompt + image 返回多个物体：

```json
{
  "objects": [
    {
      "label": "green block",
      "bbox": [x_min, y_min, x_max, y_max],
      "grasp_pixel": [x, y],
      "confidence": 0.9
    }
  ],
  "summary": "..."
}
```

建议包装成新 tool：

```text
detect_objects(query)
```

### 3.5 像素到机器人坐标转换

复用：

```text
sagittarius_object_color_detector/config/vision_config.yaml
```

当前转换公式：

```python
robot_x = k1 * pixel_y + b1
robot_y = k2 * pixel_x + b2
```

建议包装成新 tool 或工具函数：

```text
pixel_to_robot_xy(pixel_x, pixel_y)
```

注意：当前没有深度相机，Z 高度仍然来自配置。

### 3.6 运动任务

可复用：

```text
sagittarius_vlm_task_router/nodes/motion_task_executor.py
```

已有运动任务：

```text
wave_hand
nod
draw_circle
spin_wrist
```

建议包装成：

```text
execute_motion(task_type, parameters)
```

## 4. 新增 ROS 包结构

建议新增包：

```text
src/sagittarius_arm_ros/sagittarius_perception/sagittarius_vlm_agent/
```

目录结构：

```text
sagittarius_vlm_agent/
  CMakeLists.txt
  package.xml
  README.md

  launch/
    vlm_agent.launch

  config/
    agent.yaml

  nodes/
    vlm_agent_executor.py
    agent_planner.py
    agent_common.py
    memory.py
    tool_registry.py
    vision_tools.py
    robot_tools.py
    motion_tools.py
```

## 5. 模块职责

### 5.1 `vlm_agent_executor.py`

主入口，负责 agent loop。

伪代码：

```python
while not rospy.is_shutdown():
    tool_call = planner.next_tool_call(prompt, memory)
    result = tool_registry.execute(tool_call)
    memory.record(tool_call, result)

    if tool_call.tool == "finish_task":
        break
```

它不直接写具体抓取逻辑，只负责调度。

### 5.2 `agent_planner.py`

负责调用 VLM，让 VLM 输出下一步 function call。

输入：

```text
用户 prompt
当前 memory
可用工具列表
最近图像/检测结果
```

输出：

```json
{
  "tool": "pick_object",
  "arguments": {
    "object_id": "green_block_1"
  },
  "rationale": "The green block has been detected and should be picked next."
}
```

### 5.3 `tool_registry.py`

保存 tool 名称到 Python 函数的映射。

示例：

```python
TOOLS = {
    "capture_image": capture_image,
    "detect_object": detect_object,
    "detect_objects": detect_objects,
    "pick_object": pick_object,
    "place_object": place_object,
    "open_gripper": open_gripper,
    "close_gripper": close_gripper,
    "verify_grasp": verify_grasp,
    "execute_motion": execute_motion,
    "finish_task": finish_task,
}
```

### 5.4 `memory.py`

保存 agent 中间状态。

建议内容：

```json
{
  "prompt": "pick the green block and place it on the red block",
  "step": 3,
  "latest_image_path": ".../step_003.jpg",
  "objects": {
    "green_block_1": {
      "label": "green block",
      "bbox": [100, 80, 160, 140],
      "grasp_pixel": [130, 110],
      "robot_xy": [0.23, 0.04],
      "confidence": 0.91
    }
  },
  "history": [
    {
      "tool": "capture_image",
      "success": true
    }
  ]
}
```

### 5.5 `vision_tools.py`

负责图像相关工具：

```text
capture_image()
detect_object(query)
detect_objects(query)
save_annotated_image()
```

可以复用现有 VLM planner。

### 5.6 `robot_tools.py`

负责真实机械臂工具：

```text
pixel_to_robot_xy()
move_to_pose()
pick_object()
place_object()
open_gripper()
close_gripper()
verify_grasp()
return_home()
```

其中 `pick_object()` 和 `place_object()` 内部可以继续调用 `/sgr532/sgr_ctrl`。

### 5.7 `motion_tools.py`

封装已有运动任务：

```text
wave_hand
nod
draw_circle
spin_wrist
```

内部复用 `MotionTaskExecutor`。

## 6. 第一版工具列表

MVP 阶段建议只支持这些工具：

```text
capture_image
detect_object
pick_object
place_object
verify_grasp
finish_task
```

工具 schema：

```json
[
  {
    "name": "capture_image",
    "description": "Capture the latest RGB frame from the USB camera.",
    "parameters": {}
  },
  {
    "name": "detect_object",
    "description": "Detect one object in the current image.",
    "parameters": {
      "query": "string"
    }
  },
  {
    "name": "pick_object",
    "description": "Pick a previously detected object.",
    "parameters": {
      "object_id": "string"
    }
  },
  {
    "name": "place_object",
    "description": "Place the held object at a fixed slot or near a detected target object.",
    "parameters": {
      "target_object_id": "string",
      "slot_index": "integer"
    }
  },
  {
    "name": "verify_grasp",
    "description": "Verify whether the gripper is holding an object using servo payload.",
    "parameters": {}
  },
  {
    "name": "finish_task",
    "description": "Finish the current task.",
    "parameters": {
      "reason": "string"
    }
  }
]
```

## 7. 第一版支持任务

优先支持：

```text
pick up the green block
pick up the blue block
pick up the green block and place it on the red block
clean the green and blue objects
```

不要第一版就支持任意复杂任务。先把函数调用闭环跑通。

## 8. 示例流程：抓绿色方块放到红色方块上

用户输入：

```text
pick up the green block and place it on the red block
```

Agent 执行：

```text
1. capture_image()
2. detect_object(query="green block")
3. detect_object(query="red block")
4. pick_object(object_id="green_block_1")
5. verify_grasp()
6. place_object(target_object_id="red_block_1")
7. finish_task(reason="green block placed on red block")
```

其中：

```text
detect_object
  -> VLM 返回 grasp_pixel
  -> pixel_to_robot_xy 得到 robot_xy
  -> 写入 memory

pick_object
  -> 从 memory 读取 green_block_1.robot_xy
  -> 固定 pick_z
  -> 调 /sgr532/sgr_ctrl 抓取

place_object
  -> 从 memory 读取 red_block_1.robot_xy
  -> 固定 place_z
  -> 调 /sgr532/sgr_ctrl 放置
```

## 9. Agent Planner Prompt 建议

系统 prompt 应该明确告诉模型：

```text
You are a robot agent controller.
You cannot directly control the robot.
You can only call one tool at a time from the provided tool list.
Return exactly one JSON object.
Do not output Python code.
Do not invent tools.
Use memory to refer to detected objects.
If required information is missing, call capture_image or detect_object first.
```

输出格式：

```json
{
  "tool": "detect_object",
  "arguments": {
    "query": "green block"
  },
  "rationale": "Need to find the target object before picking."
}
```

## 10. 配置文件设计

`config/agent.yaml`：

```yaml
agent:
  model: "gpt-5.4"
  api_base: "https://api.openai.com/v1"
  request_timeout: 30.0
  max_steps: 10
  save_results: true
  results_dir: "/home/robotics/Team1/results"

robot:
  robot_name: "sgr532"
  arm_name: "sgr532"
  image_topic: "/usb_cam/image_raw"
  pick_z: 0.02
  place_z: 0.05
  pre_grasp_offset_z: 0.05
  lift_offset_z: 0.08
  default_roll: 0.0
  default_pitch: 1.57
  default_yaw: 0.0
  open_gripper_width: 0.0
  grasp_close_width: -0.021
  min_gripper_width: -0.021
  max_gripper_width: 0.0
  grasp_payload_threshold: 24

place_slots:
  - [0.00, 0.20]
  - [0.00, 0.16]
  - [0.00, 0.12]
```

## 11. Launch 文件设计

`launch/vlm_agent.launch`：

```xml
<launch>
  <arg name="robot_name" default="sgr532"/>
  <arg name="robot_model" default="$(arg robot_name)"/>
  <arg name="video_dev" default="/dev/usb_cam"/>
  <arg name="prompt" default="pick up the green block"/>
  <arg name="model" default="gpt-5.4"/>
  <arg name="api_base" default="https://api.openai.com/v1"/>
  <arg name="openai_api_key" default="$(optenv OPENAI_API_KEY '')"/>
  <arg name="vision_config" default="$(find sagittarius_object_color_detector)/config/vision_config.yaml"/>
  <arg name="agent_config" default="$(find sagittarius_vlm_agent)/config/agent.yaml"/>

  <include file="$(find sagittarius_moveit)/launch/demo_true.launch">
    <arg name="robot_name" value="$(arg robot_name)"/>
    <arg name="robot_model" value="$(arg robot_model)"/>
  </include>

  <node pkg="sagittarius_object_color_detector" type="sgr_ctrl.py" name="sgr_ctrl_node" output="screen" ns="$(arg robot_name)"/>

  <include file="$(find sagittarius_object_color_detector)/launch/usb_cam.launch">
    <arg name="video_dev" value="$(arg video_dev)"/>
  </include>

  <rosparam command="load" file="$(arg agent_config)" ns="vlm_agent_executor"/>

  <node pkg="sagittarius_vlm_agent" type="vlm_agent_executor.py" name="vlm_agent_executor" output="screen">
    <param name="robot_name" value="$(arg robot_name)"/>
    <param name="arm_name" value="$(arg robot_name)"/>
    <param name="prompt" value="$(arg prompt)"/>
    <param name="model" value="$(arg model)"/>
    <param name="api_base" value="$(arg api_base)"/>
    <param name="openai_api_key" value="$(arg openai_api_key)"/>
    <param name="vision_config" value="$(arg vision_config)"/>
  </node>
</launch>
```

## 12. 结果保存

为了 presentation/report，建议每次运行保存：

```text
results/<timestamp>/
  prompt.txt
  step_000_image.jpg
  step_001_detect_green_block.jpg
  step_002_detect_red_block.jpg
  memory.json
  tool_calls.json
  final_summary.json
```

`tool_calls.json` 示例：

```json
[
  {
    "step": 1,
    "tool_call": {
      "tool": "detect_object",
      "arguments": {
        "query": "green block"
      }
    },
    "result": {
      "success": true,
      "object_id": "green_block_1"
    }
  }
]
```

## 13. 实现步骤

### Step 1：新建 ROS package

```bash
cd src/sagittarius_arm_ros/sagittarius_perception
catkin_create_pkg sagittarius_vlm_agent rospy std_msgs sensor_msgs cv_bridge actionlib sagittarius_object_color_detector sdk_sagittarius_arm
```

然后添加：

```text
launch/
config/
nodes/
```

### Step 2：复制并整理基础工具代码

从旧 executor 中抽取：

```text
wait_for_frame
pixel_to_robot_xy
send_pose_goal
set_gripper_width
verify_grasp
```

放入：

```text
robot_tools.py
vision_tools.py
```

### Step 3：实现 `AgentMemory`

先支持：

```text
add_object
get_object
record_tool_call
save_json
```

### Step 4：实现 `ToolRegistry`

注册 MVP tools：

```text
capture_image
detect_object
pick_object
place_object
verify_grasp
finish_task
```

### Step 5：实现 `AgentPlanner`

调用 Chat Completions API，让模型输出 JSON tool call。

第一版可以不用 OpenAI 原生 tools 参数，直接让模型输出 JSON。这样兼容当前 `api.aibold.art/v1`。

### Step 6：实现 `vlm_agent_executor.py`

主循环：

```text
for step in range(max_steps):
    ask planner for next tool call
    validate tool name
    execute tool
    save memory
    if finish_task: break
```

### Step 7：先跑最小任务

先测试：

```bash
roslaunch sagittarius_vlm_agent vlm_agent.launch \
  prompt:="pick up the green block" \
  api_base:="https://api.aibold.art/v1" \
  openai_api_key:="..." \
  model:="gpt-5.4"
```

再测试：

```bash
roslaunch sagittarius_vlm_agent vlm_agent.launch \
  prompt:="pick up the green block and place it on the red block" \
  api_base:="https://api.aibold.art/v1" \
  openai_api_key:="..." \
  model:="gpt-5.4"
```

## 14. 注意事项

### 14.1 旧代码不能原样复用

旧代码大多是完整 executor，例如：

```text
CleanDeskExecutor.execute()
VlmGraspExecutor.execute()
MotionTaskExecutor.execute_motion()
```

它们是“一次执行完整流程”，不是“单步工具函数”。

因此需要拆分或重新封装成小工具。

### 14.2 不要让 VLM 生成动作代码

禁止这种方式：

```text
VLM 生成 Python 函数
本地 exec 执行
```

Agent 必须只能调用已有 tools。

### 14.3 当前没有深度估计

当前定位方式是：

```text
RGB image pixel -> robot XY
Z fixed by config
```

因此适合桌面平面物体。复杂堆叠、多高度物体需要额外深度模块。

### 14.4 放置到目标物体上是简化实现

第一版：

```text
place_x = target.robot_x
place_y = target.robot_y
place_z = fixed place_z
```

如果目标有高度，后续需要增加：

```text
target_height
depth camera
manual z offset
```

### 14.5 Agent 需要最大步数限制

必须配置：

```yaml
max_steps: 10
```

避免模型重复调用工具导致机械臂无限执行。

## 15. 最终效果

完成后，系统会从：

```text
prompt -> 固定 executor
```

升级为：

```text
prompt -> agent planner -> function call -> tool execution -> memory -> next function call
```

也就是更标准的 function calling agent 架构。

