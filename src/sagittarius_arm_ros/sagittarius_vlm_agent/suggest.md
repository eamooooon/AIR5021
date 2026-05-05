# Agent Function Calling 方案可行性分析与改进建议

## 1. 总体判断

`AGENT_FUNCTION_CALLING_DESIGN.md` 中提出的方案整体可行，而且方向正确。它把当前项目从：

```text
prompt -> 固定 executor
```

提升为：

```text
prompt -> agent planner -> tool call -> execution result -> memory -> next tool call
```

这比当前的一次性 VLM 调用更有深度，也更接近真实机器人 agent 架构。

综合判断：

```text
技术可行性：高
工程复杂度：中高
课程创新价值：高
真实机械臂稳定性风险：中高
```

方案中最重要的优点是：**不让 VLM 生成 Python 代码，也不让 VLM 直接控制机械臂**。VLM 只输出结构化 function call，真实机械臂控制仍由本地确定性工具函数完成。这一点对真实机器人系统非常关键。

---

## 2. 当前方案能解决的问题

当前项目的主要问题是 VLM 使用较浅：

```text
一次 prompt / image -> 一次 JSON -> 执行固定流程
```

新的 function calling agent 方案可以升级为：

```text
多轮决策
中间状态记忆
工具调用
执行反馈
失败重试
结果记录
```

例如，原来的清桌任务大致是：

```text
拍一张图
VLM 返回所有物体
机械臂按顺序抓取
```

新方案可以变成：

```text
capture_image
detect_object("green block")
pick_object("green_block_1")
verify_grasp
place_object(slot_0)
capture_image
detect_object("blue block")
pick_object("blue_block_1")
verify_grasp
place_object(slot_1)
finish_task
```

这样项目不再只是“调用 VLM 识别物体”，而是一个 **VLM-guided tool-using robotic agent**。

---

## 3. 主要风险

### 3.1 Agent 每一步都问 VLM，可能不稳定

原方案中的主循环是：

```python
while not rospy.is_shutdown():
    tool_call = planner.next_tool_call(prompt, memory)
    result = tool_registry.execute(tool_call)
    memory.record(tool_call, result)
```

这个设计理论上很灵活，但真实运行时可能出现：

```text
VLM 重复调用 detect_object
VLM 忘记已经抓过某个物体
VLM 提前 finish_task
VLM 选择不存在的 object_id
VLM 在没有图像或检测结果时要求 pick_object
```

因此不建议让 VLM 完全自由决定每一步。

建议增加一个显式的 agent state machine：

```text
INIT
  -> NEED_IMAGE
  -> NEED_DETECTION
  -> READY_TO_PICK
  -> VERIFY_GRASP
  -> READY_TO_PLACE
  -> VERIFY_PLACE
  -> DONE / FAILED
```

VLM 可以选择目标和参数，但状态机决定当前允许哪些工具。

例如当前状态是 `NEED_DETECTION`，只允许：

```text
detect_object
detect_objects
finish_task
```

当前状态是 `READY_TO_PICK`，才允许：

```text
pick_object
capture_image
```

这样可以显著减少模型乱调工具。

### 3.2 Tool 粒度还需要调整

原方案 MVP tools：

```text
capture_image
detect_object
pick_object
place_object
verify_grasp
finish_task
```

这套工具可以跑通第一版，但建议进一步区分：

```text
认知工具：不会动机械臂
执行工具：会动机械臂
```

认知工具：

```text
capture_image
detect_object
detect_objects
verify_scene
pixel_to_robot_xy
```

执行工具：

```text
move_to_observe_pose
pick_object
place_object
open_gripper
close_gripper
return_home
execute_motion
```

每个执行工具都需要明确的前置条件。例如 `pick_object(object_id)` 的前置条件应该是：

```text
object_id exists in memory
object has robot_xy
robot is not holding anything
confidence >= threshold
```

如果不满足，工具不应执行真实机械臂动作，而是返回：

```json
{
  "success": false,
  "error": "object_id_not_found",
  "recoverable": true,
  "suggested_next_tools": ["detect_object"]
}
```

### 3.3 Memory 需要更机器人化

原方案中的 memory 设计方向正确，但还需要增加机器人状态和任务状态。

建议结构：

```json
{
  "robot_state": {
    "holding_object_id": null,
    "last_pose": "observe",
    "gripper_state": "open"
  },
  "scene_state": {
    "image_id": "step_003",
    "objects": {}
  },
  "task_state": {
    "completed_objects": [],
    "failed_objects": [],
    "retry_count": {}
  }
}
```

尤其要有：

```text
holding_object_id
completed_objects
failed_objects
retry_count
```

否则 VLM 容易让机械臂重复抓已经处理过的物体。

### 3.4 `place_object(target_object_id)` 比设计中更复杂

原方案中设想：

```text
place_x = target.robot_x
place_y = target.robot_y
place_z = fixed place_z
```

这对“放到红色方块上”是一个简化实现，但真实风险较高：

```text
红色方块有高度，但系统没有深度
place_z 固定可能太低，撞到目标物
place_z 太高，物体可能放不上去
目标 bbox 中心不一定是可放置中心
```

第一版建议不要宣传成精准 “place on the red block”，而是先实现：

```text
place near the red block
```

如果必须实现 “on top of red block”，建议增加配置：

```yaml
target_object_height_z: 0.025
place_on_target_z_offset: 0.030
```

然后使用：

```text
place_z = table_z + target_object_height_z + release_offset
```

没有深度相机时，这个高度只能人工配置。

---

## 4. 建议补强的关键机制

### 4.1 Tool precondition/effect 机制

每个 tool 不应只是一个 Python 函数，还应声明：

```json
{
  "name": "pick_object",
  "preconditions": [
    "object_exists",
    "object_has_robot_xy",
    "gripper_empty"
  ],
  "effects_on_success": [
    "holding_object"
  ]
}
```

例如：

```python
ToolSpec(
    name="pick_object",
    preconditions=["object_exists", "robot_not_holding"],
    effects=["holding_object"],
)
```

这样系统会从普通 function calling 升级为更正规的：

```text
symbolic planning + robotic tool execution
```

这可以作为报告中的重要创新点。

### 4.2 增加 visual verification

原方案 MVP 只有：

```text
verify_grasp
```

它依赖夹爪 payload，只能判断夹爪是否受力，不能判断：

```text
是不是抓了正确物体
物体是否真的离开原位置
是否成功放置到目标区域
任务是否完成
```

建议新增：

```text
verify_object_removed(object_id)
verify_object_placed(object_id, target)
verify_task_complete(query)
```

实现方式可以复用 VLM：

```text
before image + after image + verification question -> Yes/No JSON
```

示例输出：

```json
{
  "question": "Is the green block no longer at its original tabletop location?",
  "answer": true,
  "confidence": 0.86,
  "rationale": "The green block is absent from the previous location."
}
```

这和 Manipulate-Anything 的 sub-task verification 思路类似，能明显提升项目深度。

### 4.3 Planner 输出增加置信度和风险

原方案 planner 输出：

```json
{
  "tool": "detect_object",
  "arguments": {},
  "rationale": "..."
}
```

建议升级为：

```json
{
  "tool": "pick_object",
  "arguments": {
    "object_id": "green_block_1"
  },
  "confidence": 0.82,
  "risk": "low",
  "rationale": "The green block has been detected and localized."
}
```

如果：

```text
confidence < 0.6
risk != low
```

系统不直接执行物理动作，而是重新观察、重新检测，或请求人工确认。

### 4.4 执行前增加 safety gate

任何会移动机械臂的 tool 执行前，都应该检查：

```text
目标坐标是否在 workspace 内
z 是否高于最小安全高度
是否超过最大步数
是否连续失败太多次
object confidence 是否足够
当前是否已经 holding object
```

示例：

```python
def validate_pose(x, y, z):
    return (
        0.08 <= x <= 0.35 and
        -0.18 <= y <= 0.22 and
        0.01 <= z <= 0.25
    )
```

如果不通过，直接拒绝执行：

```json
{
  "success": false,
  "error": "pose_out_of_workspace"
}
```

这既提升安全性，也能作为报告里的工程深度。

---

## 5. 推荐架构：Plan + Tool Calling 混合模式

原方案是纯 agent loop：每一步都问 VLM 下一步。更稳妥的是混合架构：

```text
第一阶段：VLM 生成高层多步骤 plan
第二阶段：本地 executor 按 plan 执行
第三阶段：每步失败时再问 VLM 重新规划
```

推荐整体流程：

```text
User Prompt
  -> MultiStepPlanner 生成 plan
  -> PlanExecutor 执行 step
  -> ToolRegistry 调工具
  -> Verifier 检查结果
  -> FailureRecoveryPlanner 必要时重规划
```

这样比每一步自由 function calling 更适合真实机械臂，因为：

```text
整体任务结构更稳定
VLM 只在必要时介入
执行器更容易调试
失败恢复逻辑更清晰
```

---

## 6. 建议的 ROS 包结构

可以在原方案基础上增加三个关键模块：

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
    vlm_agent_executor.py        # 主循环
    agent_planner.py             # 生成 high-level plan 或 next tool call
    agent_common.py              # dataclass/schema
    memory.py                    # 状态和日志
    tool_registry.py             # 工具注册和校验
    tool_specs.py                # 工具 schema/precondition/effect
    vision_tools.py              # capture/detect/verify image
    robot_tools.py               # pick/place/gripper/move
    motion_tools.py              # wave/nod/circle/spin
    safety.py                    # workspace/safety validation
    recovery.py                  # retry/replan 策略
```

相比原方案，新增重点是：

```text
tool_specs.py
safety.py
recovery.py
```

这三个模块会让系统更像可靠机器人 agent，而不是简单的 VLM 调函数循环。

---

## 7. MVP 范围建议

原方案第一版支持任务：

```text
pick up the green block
pick up the blue block
pick up the green block and place it on the red block
clean the green and blue objects
```

建议 MVP 收敛为两个任务：

```text
1. pick up the green block
2. clean the green and blue objects
```

原因：

```text
pick up green block：验证 agent function calling 最小链路
clean green and blue objects：展示多步骤、闭环、replanning
```

“放到红色方块上”涉及目标高度、碰撞和放置稳定性，建议作为第二阶段。

---

## 8. 推荐实现路线

### Step 1：最小 function calling 链路

实现：

```text
capture_image
detect_object
pick_object
verify_grasp
finish_task
```

目标任务：

```text
pick up the green block
```

### Step 2：加入 memory 和结构化日志

保存：

```text
memory.json
tool_calls.json
step_000_image.jpg
step_001_detect_green_block.jpg
final_summary.json
```

### Step 3：加入 visual verification

新增：

```text
verify_object_removed
verify_task_complete
```

### Step 4：实现闭环清桌

目标任务：

```text
clean the green and blue objects
```

执行策略：

```text
observe -> detect next object -> pick -> verify -> place -> verify -> reobserve
```

### Step 5：加入失败恢复

支持：

```text
重试当前目标
重新检测当前目标
跳过低置信度目标
达到最大失败次数后停止
```

### Step 6：做 baseline 对比实验

对比：

```text
旧版 one-shot clean_desk
新版 function-calling closed-loop agent
```

评价指标：

```text
抓取成功率
清理完成率
平均尝试次数
失败恢复次数
总执行时间
VLM 调用次数
```

---

## 9. 报告中的创新点表述

如果按该方案推进，可以把创新点表述为：

```text
1. We redesign the VLM interface from one-shot JSON prediction to a function-calling robotic agent.

2. The VLM can only select from verified local robot tools, while all physical actions are guarded by preconditions and workspace safety checks.

3. We introduce an explicit memory module that stores detected objects, robot holding state, tool history, and retry status.

4. We add closed-loop visual and gripper-based verification after physical actions, enabling re-observation and recovery from failed grasps.

5. We evaluate the agent against the previous one-shot VLM baseline on tabletop cleaning tasks.
```

这比当前项目的：

```text
We use VLM to detect objects and guide robot motion
```

更有深度，也更容易回应“创新点不足”的评价。

---

## 10. 最终结论

`AGENT_FUNCTION_CALLING_DESIGN.md` 的方案值得推进。它的核心优点是：

```text
不让 VLM 直接控制机械臂
不让 VLM 生成代码
复用现有 ROS/MoveIt/VLM 抓取模块
引入 tool calling、memory、闭环执行
```

但为了让它真正适合真实机械臂运行，建议进一步补强：

```text
状态机约束
tool precondition/effect
visual verification
safety gate
failure recovery
结构化日志
baseline 对比实验
```

推荐的务实落地路线是：

```text
第一步：实现 capture_image + detect_object + pick_object + verify_grasp + finish_task

第二步：加入 memory 和 tool_calls.json 保存

第三步：加入 verify_object_removed 的 VLM 图像验证

第四步：实现 clean green and blue objects 的闭环重规划

第五步：和旧版 one-shot clean_desk 做实验对比
```

这样既能跑出真实机械臂效果，也能在课程汇报中讲出清晰的技术深度。
