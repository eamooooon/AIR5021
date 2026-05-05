#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import traceback

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from agent_planner import AgentPlanner
from memory import AgentMemory
from motion_tools import MotionTools
from robot_tools import RobotTools
from tool_registry import ToolRegistry
from vision_tools import VisionTools


def get_param(name, default=None):
    if rospy.has_param("~" + name):
        return rospy.get_param("~" + name)
    return default


def get_agent_param(name, default=None):
    if rospy.has_param("~" + name):
        value = rospy.get_param("~" + name)
        if value not in ("", None):
            return value
    nested = "~agent/" + name
    if rospy.has_param(nested):
        value = rospy.get_param(nested)
        if value not in ("", None):
            return value
    return default


def get_robot_param(name, default=None):
    if rospy.has_param("~" + name):
        return rospy.get_param("~" + name)
    nested = "~robot/" + name
    if rospy.has_param(nested):
        return rospy.get_param(nested)
    return default


def mask_secret(value):
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return "%s...%s" % (value[:4], value[-4:])


class VlmAgentExecutor:
    def __init__(self):
        self.prompt = get_param("prompt", "pick up the green block")
        self.api_key = (
            get_agent_param("openai_api_key", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.api_base = get_agent_param("api_base", "https://api.openai.com/v1")
        self.model = get_agent_param("model", "gpt-5.4")
        self.request_timeout = float(get_agent_param("request_timeout", 30.0))
        self.max_steps = int(get_agent_param("max_steps", 10))
        self.execution_mode = str(get_agent_param("execution_mode", "auto")).strip().lower()
        self.save_results = bool(get_agent_param("save_results", True))
        self.results_dir = str(get_agent_param("results_dir", "")).strip()
        self.min_confidence = float(get_agent_param("min_confidence", 0.35))
        self.debug_image = bool(get_agent_param("debug_image", True))
        self.vision_config = get_param("vision_config", "")
        self.hand_eye_config = get_param("hand_eye_config", "")
        self.enable_motion_tools = bool(get_agent_param("enable_motion_tools", False))

        self.robot_config = {
            "robot_name": get_robot_param("robot_name", "sgr532"),
            "arm_name": get_robot_param("arm_name", get_robot_param("robot_name", "sgr532")),
            "action_wait_timeout": get_robot_param("action_wait_timeout", 30.0),
            "move_group_wait_timeout": get_robot_param("move_group_wait_timeout", 15.0),
            "servo_info_wait_timeout": get_robot_param("servo_info_wait_timeout", 15.0),
            "default_pick_roll": get_robot_param("default_pick_roll", 0.0),
            "default_pick_pitch": get_robot_param("default_pick_pitch", 1.57),
            "default_pick_yaw": get_robot_param("default_pick_yaw", 0.0),
            "observe_x": get_robot_param("observe_x", 0.20),
            "observe_y": get_robot_param("observe_y", 0.00),
            "observe_z": get_robot_param("observe_z", 0.15),
            "pick_z": get_robot_param("pick_z", 0.02),
            "pre_grasp_offset_z": get_robot_param("pre_grasp_offset_z", 0.05),
            "lift_offset_z": get_robot_param("lift_offset_z", 0.08),
            "place_z": get_robot_param("place_z", 0.05),
            "place_roll": get_robot_param("place_roll", 0.0),
            "place_pitch": get_robot_param("place_pitch", 1.57),
            "place_yaw": get_robot_param("place_yaw", 1.57),
            "open_gripper_width": get_robot_param("open_gripper_width", 0.0),
            "grasp_close_width": get_robot_param("grasp_close_width", -0.021),
            "min_gripper_width": get_robot_param("min_gripper_width", -0.021),
            "max_gripper_width": get_robot_param("max_gripper_width", 0.0),
            "grasp_payload_threshold": get_robot_param("grasp_payload_threshold", 24),
            "place_slots": get_robot_param("place_slots", [[0.0, 0.2]]),
            "hand_eye_config": self.hand_eye_config,
        }
        self.memory = AgentMemory(self.prompt, self.results_dir, self.save_results)

        if not self.api_key:
            self.memory.append_log("ERROR", "Missing OpenAI API key")
            raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or ~openai_api_key.")
        if not self.vision_config:
            self.memory.append_log("ERROR", "Missing ~vision_config")
            raise RuntimeError("Missing ~vision_config")
        rospy.loginfo(
            "VLM agent API config: api_base=%s model=%s api_key=%s",
            self.api_base,
            self.model,
            mask_secret(self.api_key),
        )
        self.memory.append_log(
            "INFO",
            "VLM agent initialized",
            {
                "prompt": self.prompt,
                "api_base": self.api_base,
                "model": self.model,
                "execution_mode": self.execution_mode,
                "api_key": mask_secret(self.api_key),
                "results_dir": self.memory.get_results_dir() if self.save_results else "",
            },
        )

        self.robot_tools = RobotTools(self.memory, self.vision_config, self.robot_config)
        self.vision_tools = VisionTools(
            self.memory,
            self.api_base,
            self.api_key,
            self.model,
            self.request_timeout,
            get_robot_param("image_topic", "/usb_cam/image_raw"),
            self.min_confidence,
            self.debug_image,
        )
        self.motion_tools = None
        if self.enable_motion_tools:
            self.motion_tools = MotionTools(self.robot_config["robot_name"], self.robot_config["arm_name"], self.prompt)
        self.registry = ToolRegistry(self.vision_tools, self.robot_tools, self.motion_tools)
        self.planner = AgentPlanner(self.api_base, self.api_key, self.model, self.request_timeout, rospy_module=rospy)

    def parse_template_task(self):
        prompt = self.prompt.strip().lower()
        place_match = re.search(
            r"(?:pick up|pick|grab)\s+(?:the\s+)?(.+?)\s+(?:and\s+)?(?:place|put)\s+(?:it\s+)?(?:on|onto|near|at)\s+(?:the\s+)?(.+)",
            prompt,
        )
        if place_match:
            return {
                "type": "pick_and_place",
                "object_query": place_match.group(1).strip(),
                "target_query": place_match.group(2).strip(),
            }

        pick_match = re.search(r"(?:pick up|pick|grab)\s+(?:the\s+)?(.+)", prompt)
        if pick_match:
            return {
                "type": "pick",
                "object_query": pick_match.group(1).strip(),
            }
        return None

    def execute_and_record(self, tool, arguments=None, rationale=""):
        if arguments is None:
            arguments = {}
        self.memory.next_step()
        tool_call = {
            "tool": tool,
            "arguments": arguments,
            "rationale": rationale,
        }
        rospy.loginfo("Template tool=%s args=%s", tool, arguments)
        self.memory.append_log("INFO", "Template selected tool", tool_call)
        try:
            result = self.registry.execute(tool_call)
        except Exception as exc:
            result = {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            rospy.logerr("Tool %s failed: %s", tool, exc)
            self.memory.append_log("ERROR", "Tool execution failed", result)
            self.memory.record(tool_call, result)
            self.memory.save_snapshot()
            raise

        self.memory.append_log("INFO", "Tool execution finished", result)
        self.memory.record(tool_call, result)
        snapshot_path = self.memory.save_snapshot()
        if snapshot_path:
            rospy.loginfo("Saved agent memory to %s", snapshot_path)
            self.memory.append_log("INFO", "Saved agent memory", {"path": snapshot_path})
        return result

    def execute_template(self, task):
        rospy.loginfo("Using template execution: %s", task)
        self.memory.append_log("INFO", "Using template execution", task)
        self.execute_and_record("capture_image", {}, "Capture an image before visual detection.")
        self.execute_and_record(
            "detect_objects",
            {"query": self.prompt},
            "Generate local block proposals before selecting the requested object.",
        )
        obj_result = self.execute_and_record(
            "select_object",
            {"query": task["object_query"]},
            "Select the object requested by the user from local proposals.",
        )
        object_id = obj_result["object_id"]

        target_id = ""
        if task["type"] == "pick_and_place":
            target_result = self.execute_and_record(
                "select_object",
                {"query": task["target_query"]},
                "Select the placement target requested by the user from local proposals.",
            )
            target_id = target_result["object_id"]

        self.execute_and_record(
            "pick_object",
            {"object_id": object_id},
            "Pick the detected object.",
        )

        if task["type"] == "pick_and_place":
            self.execute_and_record(
                "place_object",
                {"target_object_id": target_id},
                "Place the held object at the detected target.",
            )
            reason = "Picked %s and placed it at %s." % (object_id, target_id)
        else:
            reason = "Picked %s." % object_id

        self.execute_and_record("finish_task", {"reason": reason}, "Template task completed.")

    def execute(self):
        rospy.loginfo("Starting VLM agent task: %s", self.prompt)
        self.memory.append_log("INFO", "Starting VLM agent task", {"prompt": self.prompt})
        self.robot_tools.move_to_observe()
        self.memory.append_log("INFO", "Moved robot to observe pose")

        template_task = self.parse_template_task()
        if self.execution_mode == "template":
            if template_task is None:
                raise RuntimeError("execution_mode=template but prompt did not match a supported template")
            self.execute_template(template_task)
            return
        if self.execution_mode == "auto" and template_task is not None:
            self.execute_template(template_task)
            return

        for _ in range(self.max_steps):
            self.memory.next_step()
            rospy.loginfo("Agent step %d/%d", self.memory.step, self.max_steps)
            self.memory.append_log("INFO", "Agent step started", {"max_steps": self.max_steps})
            tool_call = self.planner.next_tool_call(self.prompt, self.memory.snapshot(), self.registry.schemas())
            rospy.loginfo("Planner selected tool=%s args=%s rationale=%s", tool_call.get("tool"), tool_call.get("arguments"), tool_call.get("rationale", ""))
            self.memory.append_log("INFO", "Planner selected tool", tool_call)

            try:
                result = self.registry.execute(tool_call)
            except Exception as exc:
                result = {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                rospy.logerr("Tool %s failed: %s", tool_call.get("tool"), exc)
                self.memory.append_log("ERROR", "Tool execution failed", result)
            else:
                self.memory.append_log("INFO", "Tool execution finished", result)

            self.memory.record(tool_call, result)
            snapshot_path = self.memory.save_snapshot()
            if snapshot_path:
                rospy.loginfo("Saved agent memory to %s", snapshot_path)
                self.memory.append_log("INFO", "Saved agent memory", {"path": snapshot_path})

            if tool_call.get("tool") == "finish_task" and result.get("success"):
                rospy.loginfo("VLM agent finished: %s", result.get("reason", ""))
                self.memory.append_log("INFO", "VLM agent finished", {"reason": result.get("reason", "")})
                return

        self.memory.append_log("ERROR", "Agent reached max_steps without finish_task", {"max_steps": self.max_steps})
        raise RuntimeError("Agent reached max_steps=%d without finish_task" % self.max_steps)


def main():
    rospy.init_node("vlm_agent_executor", anonymous=False)
    executor = VlmAgentExecutor()
    executor.execute()


if __name__ == "__main__":
    main()
