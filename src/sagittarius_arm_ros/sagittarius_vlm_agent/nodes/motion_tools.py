#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAGITTARIUS_ARM_ROS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PERCEPTION_DIR = os.path.join(SAGITTARIUS_ARM_ROS_DIR, "sagittarius_perception")
TASK_ROUTER_NODES = os.path.join(PERCEPTION_DIR, "sagittarius_vlm_task_router", "nodes")
if TASK_ROUTER_NODES not in sys.path:
    sys.path.insert(0, TASK_ROUTER_NODES)

from motion_task_executor import MotionTaskExecutor


class MotionTools:
    def __init__(self, robot_name, arm_name, prompt):
        self.prompt = prompt
        self.executor = MotionTaskExecutor(robot_name, arm_name)

    def execute_motion(self, task_type, parameters=None):
        if parameters is None:
            parameters = {}
        self.executor.execute_motion(self.prompt, task_type, parameters)
        return {"success": True, "task_type": task_type, "parameters": parameters}
