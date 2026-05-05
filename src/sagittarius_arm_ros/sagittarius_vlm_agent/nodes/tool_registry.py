#!/usr/bin/env python3
# -*- coding: utf-8 -*-


TOOL_SCHEMAS = [
    {
        "name": "capture_image",
        "description": "Capture the latest RGB frame from the USB camera.",
        "parameters": {},
    },
    {
        "name": "detect_object",
        "description": "Compatibility helper: generate local block proposals, select one object by query, and store it in memory.",
        "parameters": {"query": "string"},
    },
    {
        "name": "detect_objects",
        "description": "Generate local color-independent block proposals in the current image and store them as obj_1, obj_2, ... in memory.",
        "parameters": {"query": "string"},
    },
    {
        "name": "select_object",
        "description": "Select one object_id from existing local proposals using the user's query. The VLM must return only an existing proposal ID.",
        "parameters": {"query": "string"},
    },
    {
        "name": "pick_object",
        "description": "Pick a previously detected object using its memory object_id.",
        "parameters": {"object_id": "string"},
    },
    {
        "name": "place_object",
        "description": "Place the held object at a fixed slot or near a detected target object.",
        "parameters": {"target_object_id": "string optional", "slot_index": "integer optional"},
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper.",
        "parameters": {},
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper.",
        "parameters": {},
    },
    {
        "name": "verify_grasp",
        "description": "Verify whether the gripper is holding an object using servo payload.",
        "parameters": {},
    },
    {
        "name": "execute_motion",
        "description": "Execute a known gesture/motion task.",
        "parameters": {"task_type": "wave_hand|nod|draw_circle|spin_wrist", "parameters": "object optional"},
    },
    {
        "name": "finish_task",
        "description": "Finish the current task.",
        "parameters": {"reason": "string"},
    },
]


class ToolRegistry:
    def __init__(self, vision_tools, robot_tools, motion_tools=None):
        self.vision_tools = vision_tools
        self.robot_tools = robot_tools
        self.motion_tools = motion_tools

    def schemas(self):
        if self.motion_tools is None:
            return [schema for schema in TOOL_SCHEMAS if schema["name"] != "execute_motion"]
        return TOOL_SCHEMAS

    def execute(self, tool_call):
        tool = tool_call.get("tool", "")
        arguments = tool_call.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            raise RuntimeError("Tool arguments must be an object")

        if tool == "capture_image":
            return self.vision_tools.capture_image()
        if tool == "detect_object":
            return self.vision_tools.detect_object(arguments.get("query", ""), self.robot_tools)
        if tool == "detect_objects":
            return self.vision_tools.detect_objects(arguments.get("query", ""), self.robot_tools)
        if tool == "select_object":
            return self.vision_tools.select_object(arguments.get("query", ""))
        if tool == "pick_object":
            return self.robot_tools.pick_object(arguments.get("object_id", ""))
        if tool == "place_object":
            return self.robot_tools.place_object(
                target_object_id=arguments.get("target_object_id", "") or "",
                slot_index=arguments.get("slot_index"),
            )
        if tool == "open_gripper":
            return self.robot_tools.open_gripper()
        if tool == "close_gripper":
            return self.robot_tools.close_gripper()
        if tool == "verify_grasp":
            return self.robot_tools.verify_grasp()
        if tool == "execute_motion":
            if self.motion_tools is None:
                raise RuntimeError("execute_motion is not enabled")
            return self.motion_tools.execute_motion(arguments.get("task_type", ""), arguments.get("parameters", {}) or {})
        if tool == "finish_task":
            return {"success": True, "finished": True, "reason": arguments.get("reason", "")}
        raise RuntimeError("Unknown tool: %s" % tool)
