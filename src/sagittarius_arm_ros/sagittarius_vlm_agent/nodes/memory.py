#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
from datetime import datetime

from agent_common import object_key


class AgentMemory:
    def __init__(self, prompt, results_dir="", save_results=True):
        self.prompt = prompt
        self.step = 0
        self.latest_image_path = ""
        self.objects = {}
        self.history = []
        self.held_object_id = ""
        self.save_results = save_results
        self.results_dir = results_dir
        self.run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.active_results_dir = None

    def get_results_dir(self):
        if self.active_results_dir is not None:
            return self.active_results_dir
        base_dir = self.results_dir
        if not base_dir:
            base_dir = os.path.join(os.getenv("ROS_HOME", os.path.expanduser("~/.ros")), "vlm_agent_results")
        run_dir = os.path.join(base_dir, "vlm_agent", self.run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        self.active_results_dir = run_dir
        return run_dir

    def next_step(self):
        self.step += 1
        return self.step

    def make_object_id(self, label):
        base = object_key(label)
        index = 1
        object_id = "%s_%d" % (base, index)
        while object_id in self.objects:
            index += 1
            object_id = "%s_%d" % (base, index)
        return object_id

    def add_object(self, label, bbox, grasp_pixel, robot_xy, confidence, gripper_width=0.02, yaw_deg=0.0, rationale="", object_id=None, extra=None):
        if object_id is None:
            object_id = self.make_object_id(label)
        self.objects[object_id] = {
            "id": object_id,
            "label": label,
            "bbox": bbox,
            "grasp_pixel": grasp_pixel,
            "robot_xy": robot_xy,
            "confidence": confidence,
            "gripper_width": gripper_width,
            "yaw_deg": yaw_deg,
            "rationale": rationale,
        }
        if extra:
            self.objects[object_id].update(extra)
        return object_id

    def get_object(self, object_id):
        if object_id not in self.objects:
            raise RuntimeError("Unknown object_id=%s" % object_id)
        return self.objects[object_id]

    def record(self, tool_call, result):
        self.history.append(
            {
                "step": self.step,
                "tool": tool_call.get("tool", ""),
                "arguments": tool_call.get("arguments", {}),
                "success": bool(result.get("success", False)),
                "result": result,
            }
        )

    def snapshot(self):
        return {
            "prompt": self.prompt,
            "step": self.step,
            "latest_image_path": self.latest_image_path,
            "held_object_id": self.held_object_id,
            "objects": self.objects,
            "history": self.history,
            "run_log_path": self.get_run_log_path() if self.save_results else "",
        }

    def get_run_log_path(self):
        return os.path.join(self.get_results_dir(), "run.log")

    def append_log(self, level, message, payload=None):
        if not self.save_results:
            return ""
        path = self.get_run_log_path()
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "step": self.step,
            "message": message,
        }
        if payload is not None:
            record["payload"] = payload
        with open(path, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def save_snapshot(self):
        if not self.save_results:
            return ""
        path = os.path.join(self.get_results_dir(), "memory.json")
        with open(path, "w") as handle:
            json.dump(self.snapshot(), handle, indent=2, ensure_ascii=False)
        return path
