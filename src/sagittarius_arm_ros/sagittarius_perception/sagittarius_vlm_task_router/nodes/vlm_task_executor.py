#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

CLEANER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "sagittarius_vlm_cleaner", "nodes")
CLEANER_DIR = os.path.abspath(CLEANER_DIR)
if os.path.isdir(CLEANER_DIR) and CLEANER_DIR not in sys.path:
    sys.path.insert(0, CLEANER_DIR)

from clean_desk_executor import CleanDeskExecutor
from motion_task_executor import MotionTaskExecutor
from task_router_planner import PromptTaskRouter


class UnifiedTaskExecutor:
    def __init__(self):
        self.robot_name = rospy.get_param("~robot_name", "sgr532")
        self.arm_name = rospy.get_param("~arm_name", self.robot_name)
        self.prompt = rospy.get_param("~prompt", "clean the desk")
        self.api_key = rospy.get_param("~openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        self.api_base = rospy.get_param("~api_base", "https://api.openai.com/v1")
        self.model = rospy.get_param("~model", "gpt-5.4")
        self.request_timeout = float(self.get_config_param("request_timeout", 20.0))
        self.use_llm_routing = bool(self.get_config_param("use_llm_routing", True))
        self.supported_tasks = list(
            self.get_config_param("supported_tasks", ["clean_desk", "wave_hand", "nod", "draw_circle", "spin_wrist"])
        )

        self.router = PromptTaskRouter(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.request_timeout,
            use_llm_routing=self.use_llm_routing,
            rospy_module=rospy,
        )

    def get_config_param(self, name, default):
        private_name = "~" + name
        legacy_name = "~task_router/" + name
        if rospy.has_param(private_name):
            return rospy.get_param(private_name)
        if rospy.has_param(legacy_name):
            return rospy.get_param(legacy_name)
        return default

    def execute(self):
        rospy.loginfo("Unified task prompt: %s", self.prompt)
        intent = self.router.route(self.prompt)
        rospy.loginfo(
            "Task routing result: task_type=%s rationale=%s parameters=%s supported_tasks=%s",
            intent.task_type,
            intent.rationale,
            intent.parameters,
            self.supported_tasks,
        )

        if intent.task_type == "clean_desk":
            executor = CleanDeskExecutor()
            executor.execute()
            return

        if intent.task_type in ["wave_hand", "nod", "draw_circle", "spin_wrist"]:
            executor = MotionTaskExecutor(robot_name=self.robot_name, arm_name=self.arm_name, rospy_module=rospy)
            executor.execute_motion(self.prompt, intent.task_type, intent.parameters)
            return

        raise RuntimeError("Unsupported or unknown task_type=%s for prompt=%s" % (intent.task_type, self.prompt))


def main():
    rospy.init_node("vlm_task_executor", anonymous=False)
    executor = UnifiedTaskExecutor()
    try:
        executor.execute()
    except Exception as exc:
        rospy.logerr("Unified task execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
