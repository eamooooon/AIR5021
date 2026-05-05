#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import time

import actionlib
import moveit_commander
import rospy
import tf.transformations as transformations
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose

from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal


class MotionTaskExecutor:
    def __init__(self, robot_name, arm_name, rospy_module=rospy):
        self.rospy = rospy_module
        self.robot_name = robot_name
        self.arm_name = arm_name
        self.default_roll = float(self.get_config_param("default_roll", 0.0))
        self.default_pitch = float(self.get_config_param("default_pitch", 1.57))
        self.default_yaw = float(self.get_config_param("default_yaw", 0.0))
        self.home_x = float(self.get_config_param("home_x", 0.18))
        self.home_y = float(self.get_config_param("home_y", 0.00))
        self.home_z = float(self.get_config_param("home_z", 0.16))
        self.wave_x = float(self.get_config_param("wave_x", 0.18))
        self.wave_y = float(self.get_config_param("wave_y", 0.00))
        self.wave_z = float(self.get_config_param("wave_z", 0.18))
        self.nod_x = float(self.get_config_param("nod_x", 0.20))
        self.nod_y = float(self.get_config_param("nod_y", 0.00))
        self.nod_z = float(self.get_config_param("nod_z", 0.16))
        self.circle_x = float(self.get_config_param("circle_center_x", 0.18))
        self.circle_y = float(self.get_config_param("circle_center_y", 0.00))
        self.circle_z = float(self.get_config_param("circle_center_z", 0.16))
        self.default_circle_radius = float(self.get_config_param("default_circle_radius", 0.04))
        self.default_circle_points = int(self.get_config_param("default_circle_points", 24))
        self.circle_eef_step = float(self.get_config_param("circle_eef_step", 0.008))
        self.circle_velocity_scaling = float(self.get_config_param("circle_velocity_scaling", 0.8))
        self.circle_acceleration_scaling = float(self.get_config_param("circle_acceleration_scaling", 0.8))
        self.cartesian_max_attempts = int(self.get_config_param("cartesian_max_attempts", 6))
        self.default_repetitions = int(self.get_config_param("default_repetitions", 2))
        self.default_transition_points = int(self.get_config_param("default_transition_points", 5))
        self.default_hold_steps = int(self.get_config_param("default_hold_steps", 2))
        self.waypoint_pause = float(self.get_config_param("waypoint_pause_sec", 0.20))
        self.action_wait_timeout = float(self.get_config_param("action_wait_timeout", 30.0))
        self.save_results = bool(self.get_config_param("save_results", True))
        self.results_dir = str(self.get_config_param("results_dir", "")).strip()
        self.run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.active_results_dir = None
        self.circle_group = None

        self.arm_client = actionlib.SimpleActionClient(self.arm_name + "/sgr_ctrl", SGRCtrlAction)
        self.rospy.loginfo("Waiting for motion action server: %s/sgr_ctrl", self.arm_name)
        if not self.arm_client.wait_for_server(rospy.Duration.from_sec(self.action_wait_timeout)):
            raise RuntimeError("Timed out waiting for action server %s/sgr_ctrl" % self.arm_name)

    def get_config_param(self, name, default):
        private_name = "~" + name
        legacy_name = "~motion_task/" + name
        if self.rospy.has_param(private_name):
            return self.rospy.get_param(private_name)
        if self.rospy.has_param(legacy_name):
            return self.rospy.get_param(legacy_name)
        return default

    def get_results_dir(self):
        if self.active_results_dir is not None:
            return self.active_results_dir
        if self.results_dir:
            base_dir = self.results_dir
        else:
            base_dir = os.path.join(os.getenv("ROS_HOME", os.path.expanduser("~/.ros")), "motion_task_results")
        run_dir = os.path.join(base_dir, self.run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        self.active_results_dir = run_dir
        self.rospy.loginfo("Resolved motion result directory: %s", self.active_results_dir)
        return self.active_results_dir

    def send_pose_goal(self, x, y, z, roll, pitch, yaw):
        goal = SGRCtrlGoal()
        goal.grasp_type = goal.GRASP_NONE
        goal.action_type = goal.ACTION_TYPE_XYZ_RPY
        goal.pos_x = x
        goal.pos_y = y
        goal.pos_z = z
        goal.pos_roll = roll
        goal.pos_pitch = pitch
        goal.pos_yaw = yaw
        state = self.arm_client.send_goal_and_wait(goal, rospy.Duration.from_sec(30))
        status_text = self.arm_client.get_goal_status_text()
        if state != GoalStatus.SUCCEEDED:
            self.arm_client.cancel_goal()
            raise RuntimeError(
                "Motion goal failed: state=%s text=%s pose=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)"
                % (state, status_text, x, y, z, roll, pitch, yaw)
            )
        return state, status_text

    def execute_pose_sequence(self, sequence):
        total = len(sequence)
        for index, step in enumerate(sequence, start=1):
            self.rospy.loginfo(
                "Executing motion waypoint %d/%d: x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f",
                index,
                total,
                step["x"],
                step["y"],
                step["z"],
                step["roll"],
                step["pitch"],
                step["yaw"],
            )
            self.send_pose_goal(step["x"], step["y"], step["z"], step["roll"], step["pitch"], step["yaw"])
            if self.waypoint_pause > 0.0:
                self.rospy.sleep(self.waypoint_pause)

    def make_pose(self, x, y, z, roll=None, pitch=None, yaw=None):
        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "roll": self.default_roll if roll is None else float(roll),
            "pitch": self.default_pitch if pitch is None else float(pitch),
            "yaw": self.default_yaw if yaw is None else float(yaw),
        }

    def make_arc_pose(self, center_x, center_y, radius, z, theta, pitch=None, yaw_bias=0.0):
        pitch_value = self.default_pitch if pitch is None else float(pitch)
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        yaw = yaw_bias + theta
        return self.make_pose(x, y, z, pitch=pitch_value, yaw=yaw)

    def to_geometry_pose(self, x, y, z, roll, pitch, yaw):
        quat = transformations.quaternion_from_euler(roll, pitch, yaw)
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    def get_circle_group(self):
        if self.circle_group is not None:
            return self.circle_group
        move_group_ns = "/" + self.robot_name
        robot_description = move_group_ns + "/robot_description"
        self.circle_group = moveit_commander.MoveGroupCommander(
            "sagittarius_arm",
            robot_description=robot_description,
            ns=move_group_ns,
        )
        self.circle_group.allow_replanning(True)
        self.circle_group.set_pose_reference_frame(self.robot_name + "/base_link")
        self.circle_group.set_goal_position_tolerance(0.001)
        self.circle_group.set_goal_orientation_tolerance(0.001)
        self.circle_group.set_max_acceleration_scaling_factor(self.circle_acceleration_scaling)
        self.circle_group.set_max_velocity_scaling_factor(self.circle_velocity_scaling)
        return self.circle_group

    def build_wave_hand_sequence(self, parameters):
        repetitions = max(1, int(parameters.get("repetitions", self.default_repetitions)))
        yaw_span = float(parameters.get("yaw_span_rad", self.get_config_param("wave_yaw_span_rad", 0.28)))
        base = self.make_pose(self.wave_x, self.wave_y, self.wave_z)
        left = self.make_pose(self.wave_x, self.wave_y, self.wave_z, yaw=self.default_yaw + yaw_span)
        right = self.make_pose(self.wave_x, self.wave_y, self.wave_z, yaw=self.default_yaw - yaw_span)
        sequence = [base]
        for _ in range(repetitions):
            sequence.extend([dict(left), dict(right), dict(left), dict(base)])
        return sequence

    def build_nod_sequence(self, parameters):
        repetitions = max(1, int(parameters.get("repetitions", self.default_repetitions)))
        pitch_span = float(parameters.get("pitch_span_rad", self.get_config_param("nod_pitch_span_rad", 0.10)))
        z_span = float(parameters.get("z_span_m", self.get_config_param("nod_z_span_m", 0.012)))
        x_span = float(parameters.get("x_span_m", self.get_config_param("nod_x_span_m", 0.018)))
        base = self.make_pose(self.nod_x, self.nod_y, self.nod_z, pitch=self.default_pitch - 0.03)
        down = self.make_pose(
            self.nod_x + x_span,
            self.nod_y,
            self.nod_z - z_span,
            pitch=self.default_pitch + pitch_span,
        )
        raise_up = self.make_pose(
            self.nod_x - x_span * 0.15,
            self.nod_y,
            self.nod_z + z_span * 0.12,
            pitch=self.default_pitch - pitch_span * 0.15,
        )
        sequence = [base]
        for _ in range(repetitions):
            sequence.extend([dict(down), dict(raise_up)])
        return sequence

    def build_draw_circle_sequence(self, parameters):
        radius = float(parameters.get("radius_m", self.default_circle_radius))
        points = max(8, int(parameters.get("points", self.default_circle_points)))
        revolutions = max(1, int(parameters.get("revolutions", 1)))
        direction = str(parameters.get("direction", "counterclockwise"))
        direction_sign = 1.0 if direction != "clockwise" else -1.0
        sequence = []
        total_points = revolutions * points
        for index in range(total_points + 1):
            theta = direction_sign * (2.0 * math.pi * float(index) / float(points))
            x = self.circle_x + radius * math.cos(theta)
            y = self.circle_y + radius * math.sin(theta)
            sequence.append(self.make_pose(x, y, self.circle_z))
        return sequence

    def execute_draw_circle_motion(self, prompt, parameters):
        sequence = self.build_draw_circle_sequence(parameters)
        self.rospy.loginfo(
            "Executing draw_circle as a single cartesian trajectory: parameters=%s waypoints=%d",
            parameters,
            len(sequence),
        )
        self.save_motion_plan(prompt, "draw_circle", parameters, sequence)
        start_pose = sequence[0]
        self.rospy.loginfo(
            "Moving to circle start pose: x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f",
            start_pose["x"],
            start_pose["y"],
            start_pose["z"],
            start_pose["roll"],
            start_pose["pitch"],
            start_pose["yaw"],
        )
        self.send_pose_goal(
            start_pose["x"],
            start_pose["y"],
            start_pose["z"],
            start_pose["roll"],
            start_pose["pitch"],
            start_pose["yaw"],
        )

        arm_group = self.get_circle_group()
        waypoints = [
            self.to_geometry_pose(
                step["x"],
                step["y"],
                step["z"],
                step["roll"],
                step["pitch"],
                step["yaw"],
            )
            for step in sequence[1:]
        ]
        arm_group.set_start_state_to_current_state()
        plan = None
        fraction = 0.0
        for _ in range(max(1, self.cartesian_max_attempts)):
            plan, fraction = arm_group.compute_cartesian_path(
                waypoints,
                self.circle_eef_step,
                0.0,
                True,
            )
            if fraction >= 0.999:
                break
        if fraction < 0.999 or plan is None:
            raise RuntimeError("Cartesian circle planning failed: fraction=%.3f" % fraction)
        arm_group.execute(plan, wait=True)
        arm_group.stop()
        arm_group.clear_pose_targets()
        self.rospy.loginfo("Motion task finished: draw_circle")

    def build_spin_wrist_sequence(self, parameters):
        repetitions = max(1, int(parameters.get("repetitions", self.default_repetitions)))
        radius = float(parameters.get("radius_m", self.get_config_param("arm_rotate_radius_m", 0.07)))
        angle_span = float(parameters.get("angle_span_rad", self.get_config_param("arm_rotate_angle_span_rad", 0.70)))
        center_x = float(parameters.get("center_x", self.get_config_param("arm_rotate_center_x", 0.12)))
        center_y = float(parameters.get("center_y", self.get_config_param("arm_rotate_center_y", 0.00)))
        rotate_z = float(parameters.get("z", self.get_config_param("arm_rotate_z", self.home_z)))
        rotate_pitch = float(parameters.get("pitch", self.get_config_param("arm_rotate_pitch", self.default_pitch)))
        base = self.make_arc_pose(center_x, center_y, radius, rotate_z, 0.0, pitch=rotate_pitch)
        left = self.make_arc_pose(center_x, center_y, radius, rotate_z, angle_span, pitch=rotate_pitch)
        right = self.make_arc_pose(center_x, center_y, radius, rotate_z, -angle_span, pitch=rotate_pitch)
        sequence = [base]
        for _ in range(repetitions):
            sequence.extend([dict(left), dict(base), dict(right), dict(base)])
        return sequence

    def build_sequence(self, task_type, parameters):
        if task_type == "wave_hand":
            return self.build_wave_hand_sequence(parameters)
        if task_type == "nod":
            return self.build_nod_sequence(parameters)
        if task_type == "draw_circle":
            return self.build_draw_circle_sequence(parameters)
        if task_type == "spin_wrist":
            return self.build_spin_wrist_sequence(parameters)
        raise RuntimeError("Unsupported motion task_type=%s" % task_type)

    def save_motion_plan(self, prompt, task_type, parameters, sequence):
        if not self.save_results:
            return
        results_dir = self.get_results_dir()
        json_path = os.path.join(results_dir, "motion_plan.json")
        payload = {
            "prompt": prompt,
            "task_type": task_type,
            "parameters": parameters,
            "sequence": sequence,
        }
        with open(json_path, "w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        self.rospy.loginfo("Saved motion task plan to %s", json_path)

    def execute_motion(self, prompt, task_type, parameters):
        if task_type == "draw_circle":
            self.execute_draw_circle_motion(prompt, parameters)
            return
        sequence = self.build_sequence(task_type, parameters)
        self.rospy.loginfo(
            "Executing motion task: task_type=%s parameters=%s waypoints=%d",
            task_type,
            parameters,
            len(sequence),
        )
        self.save_motion_plan(prompt, task_type, parameters, sequence)
        self.execute_pose_sequence(sequence)
        self.rospy.loginfo("Motion task finished: %s", task_type)
