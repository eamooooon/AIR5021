#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class MotionTools:
    def __init__(self, robot_name, arm_name, prompt):
        self.robot_name = robot_name
        self.arm_name = arm_name
        self.prompt = prompt

    def execute_motion(self, task_type, parameters=None):
        raise RuntimeError(
            "execute_motion is not implemented in standalone sagittarius_vlm_agent. "
            "Use pick/place tools or implement local motion sequences here."
        )
