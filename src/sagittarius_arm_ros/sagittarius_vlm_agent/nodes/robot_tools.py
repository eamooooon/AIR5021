#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import actionlib
import moveit_commander
import rospy
import yaml
from actionlib_msgs.msg import GoalStatus

from sagittarius_vlm_agent.msg import SGRCtrlAction, SGRCtrlGoal
from sdk_sagittarius_arm.srv import ServoRtInfo, ServoRtInfoRequest


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class RobotTools:
    def __init__(self, memory, vision_config, config):
        self.memory = memory
        self.robot_name = config["robot_name"]
        self.arm_name = config["arm_name"]
        self.action_wait_timeout = float(config["action_wait_timeout"])
        self.servo_info_wait_timeout = float(config["servo_info_wait_timeout"])
        self.move_group_wait_timeout = float(config["move_group_wait_timeout"])
        self.default_pick_roll = float(config["default_pick_roll"])
        self.default_pick_pitch = float(config["default_pick_pitch"])
        self.default_pick_yaw = float(config["default_pick_yaw"])
        self.use_detected_pick_yaw = as_bool(config["use_detected_pick_yaw"])
        self.require_reliable_pick_yaw = as_bool(config["require_reliable_pick_yaw"])
        self.pick_yaw_offset = float(config["pick_yaw_offset"])
        self.pick_yaw_scale = float(config["pick_yaw_scale"])
        self.observe_x = float(config["observe_x"])
        self.observe_y = float(config["observe_y"])
        self.observe_z = float(config["observe_z"])
        self.pick_z = float(config["pick_z"])
        self.pre_grasp_offset_z = float(config["pre_grasp_offset_z"])
        self.lift_offset_z = float(config["lift_offset_z"])
        self.place_z = float(config["place_z"])
        self.default_block_height = float(config["default_block_height"])
        self.min_estimated_block_height = float(config["min_estimated_block_height"])
        self.max_estimated_block_height = float(config["max_estimated_block_height"])
        self.place_on_target_clearance = float(config["place_on_target_clearance"])
        self.stack_pre_place_offset_z = float(config["stack_pre_place_offset_z"])
        self.max_place_approach_z = float(config["max_place_approach_z"])
        self.place_roll = float(config["place_roll"])
        self.place_pitch = float(config["place_pitch"])
        self.place_yaw = float(config["place_yaw"])
        self.open_gripper_width = float(config["open_gripper_width"])
        self.grasp_close_width = float(config["grasp_close_width"])
        self.min_gripper_width = float(config["min_gripper_width"])
        self.max_gripper_width = float(config["max_gripper_width"])
        self.grasp_payload_threshold = int(config["grasp_payload_threshold"])
        self.place_slots = [tuple(slot) for slot in config["place_slots"]]
        self.k1, self.b1, self.k2, self.b2 = self.load_linear_regression(vision_config)

        self.arm_client = actionlib.SimpleActionClient(self.arm_name + "/sgr_ctrl", SGRCtrlAction)
        rospy.loginfo("Waiting for action server: %s/sgr_ctrl", self.arm_name)
        if not self.arm_client.wait_for_server(rospy.Duration.from_sec(self.action_wait_timeout)):
            raise RuntimeError("Timed out waiting for action server %s/sgr_ctrl" % self.arm_name)

        self.servo_info_service = self.resolve_robot_resource("get_servo_info")
        rospy.loginfo("Waiting for servo info service: %s", self.servo_info_service)
        rospy.wait_for_service(self.servo_info_service, self.servo_info_wait_timeout)
        self.servo_info_srv = rospy.ServiceProxy(self.servo_info_service, ServoRtInfo)

        moveit_commander.roscpp_initialize([])
        self.gripper = moveit_commander.MoveGroupCommander(
            "sagittarius_gripper",
            robot_description=self.resolve_robot_resource("robot_description"),
            ns=self.resolve_robot_namespace(),
            wait_for_servers=self.move_group_wait_timeout,
        )
        self.gripper.set_goal_joint_tolerance(0.001)
        self.gripper.set_max_acceleration_scaling_factor(0.5)
        self.gripper.set_max_velocity_scaling_factor(0.5)

    def load_linear_regression(self, path):
        with open(path, "r") as handle:
            content = yaml.safe_load(handle)
        linear = content["LinearRegression"]
        return linear["k1"], linear["b1"], linear["k2"], linear["b2"]

    def resolve_robot_resource(self, resource_name):
        if resource_name.startswith("/"):
            return resource_name
        if self.robot_name:
            return rospy.resolve_name("%s/%s" % (self.robot_name, resource_name))
        return rospy.resolve_name(resource_name)

    def resolve_robot_namespace(self):
        if not self.robot_name:
            return rospy.get_namespace()
        return rospy.resolve_name(self.robot_name)

    def pixel_to_robot_xy(self, pixel_x, pixel_y):
        return self.k1 * pixel_y + self.b1, self.k2 * pixel_x + self.b2

    def estimate_block_height(self, proposal):
        points = proposal.get("rotated_rect", {}).get("box_points", [])
        if len(points) != 4:
            return self.default_block_height
        robot_points = [self.pixel_to_robot_xy(point[0], point[1]) for point in points]
        side_lengths = []
        for index in range(4):
            x1, y1 = robot_points[index]
            x2, y2 = robot_points[(index + 1) % 4]
            side_lengths.append(math.hypot(x2 - x1, y2 - y1))
        side_lengths = sorted(length for length in side_lengths if length > 0.0)
        if not side_lengths:
            return self.default_block_height
        estimated = sum(side_lengths[1:3]) / 2.0 if len(side_lengths) >= 3 else side_lengths[0]
        return max(self.min_estimated_block_height, min(estimated, self.max_estimated_block_height))

    def normalize_yaw(self, yaw):
        while yaw <= -math.pi:
            yaw += 2.0 * math.pi
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        return yaw

    def resolve_pick_yaw(self, obj):
        if not self.use_detected_pick_yaw:
            return self.default_pick_yaw, {
                "source": "default",
                "yaw_deg": float(obj.get("yaw_deg", 0.0)),
                "yaw_reliable": bool(obj.get("yaw_reliable", False)),
            }

        yaw_reliable = bool(obj.get("yaw_reliable", False))
        if self.require_reliable_pick_yaw and not yaw_reliable:
            return self.default_pick_yaw, {
                "source": "default_unreliable_detection",
                "yaw_deg": float(obj.get("yaw_deg", 0.0)),
                "yaw_reliable": yaw_reliable,
            }

        yaw_deg = float(obj.get("yaw_deg", 0.0))
        yaw = self.default_pick_yaw + self.pick_yaw_offset + math.radians(yaw_deg) * self.pick_yaw_scale
        yaw = self.normalize_yaw(yaw)
        return yaw, {
            "source": "detected",
            "yaw_deg": yaw_deg,
            "yaw_reliable": yaw_reliable,
            "pick_yaw": yaw,
            "pick_yaw_offset": self.pick_yaw_offset,
            "pick_yaw_scale": self.pick_yaw_scale,
        }

    def clip_gripper_width(self, width):
        return max(self.min_gripper_width, min(width, self.max_gripper_width))

    def open_width(self):
        if self.open_gripper_width == 0.0:
            return self.max_gripper_width
        return self.open_gripper_width

    def set_gripper_width(self, width):
        width = self.clip_gripper_width(width)
        self.gripper.set_joint_value_target([width, width])
        if not self.gripper.go():
            raise RuntimeError("Failed to move gripper to width %.4f" % width)
        rospy.sleep(1.0)
        return width

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
                "Pose goal failed: state=%s text=%s pose=(%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)"
                % (state, status_text, x, y, z, roll, pitch, yaw)
            )
        return state, status_text

    def move_to_observe(self):
        self.send_pose_goal(
            self.observe_x,
            self.observe_y,
            self.observe_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            self.default_pick_yaw,
        )
        return {"success": True, "pose": [self.observe_x, self.observe_y, self.observe_z]}

    def open_gripper(self):
        width = self.set_gripper_width(self.open_width())
        return {"success": True, "width": width}

    def close_gripper(self):
        width = self.set_gripper_width(self.grasp_close_width)
        return {"success": True, "width": width}

    def verify_grasp(self):
        result = self.servo_info_srv.call(ServoRtInfoRequest(servo_id=7))
        success = abs(result.payload) >= self.grasp_payload_threshold
        return {"success": True, "holding": success, "payload": result.payload}

    def pick_object(self, object_id):
        obj = self.memory.get_object(object_id)
        grasp_x, grasp_y = obj["robot_xy"]
        yaw, yaw_info = self.resolve_pick_yaw(obj)
        self.open_gripper()
        self.send_pose_goal(grasp_x, grasp_y, self.pick_z + self.pre_grasp_offset_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.send_pose_goal(grasp_x, grasp_y, self.pick_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.close_gripper()
        verification = self.verify_grasp()
        if not verification["holding"]:
            raise RuntimeError("Grasp verification failed for %s: payload=%s" % (object_id, verification["payload"]))
        self.send_pose_goal(grasp_x, grasp_y, self.pick_z + self.lift_offset_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.memory.held_object_id = object_id
        return {
            "success": True,
            "object_id": object_id,
            "verification": verification,
            "grasp_xy": [grasp_x, grasp_y],
            "yaw": yaw,
            "yaw_info": yaw_info,
        }

    def resolve_place_pose(self, target_object_id="", slot_index=None):
        if target_object_id:
            target = self.memory.get_object(target_object_id)
            target_height = float(target.get("estimated_block_height", self.default_block_height))
            place_z = self.place_z + target_height + self.place_on_target_clearance
            approach_z = min(place_z + self.stack_pre_place_offset_z, self.max_place_approach_z)
            approach_z = max(approach_z, place_z)
            return {
                "x": target["robot_xy"][0],
                "y": target["robot_xy"][1],
                "place_z": place_z,
                "approach_z": approach_z,
                "retreat_z": approach_z,
                "mode": "target_object",
                "components": {
                    "table_release_z": self.place_z,
                    "target_height": target_height,
                    "held_height": self.held_object_height(),
                    "place_on_target_clearance": self.place_on_target_clearance,
                    "stack_pre_place_offset_z": self.stack_pre_place_offset_z,
                    "max_place_approach_z": self.max_place_approach_z,
                    "target_object_id": target_object_id,
                    "held_object_id": self.memory.held_object_id,
                },
            }
        if slot_index is None:
            slot_index = 0
        slot_index = max(0, min(int(slot_index), len(self.place_slots) - 1))
        slot_x, slot_y = self.place_slots[slot_index]
        approach_z = min(self.place_z + self.pre_grasp_offset_z, self.max_place_approach_z)
        approach_z = max(approach_z, self.place_z)
        return {
            "x": slot_x,
            "y": slot_y,
            "place_z": self.place_z,
            "approach_z": approach_z,
            "retreat_z": approach_z,
            "mode": "slot_%d" % slot_index,
            "components": {
                "table_release_z": self.place_z,
                "pre_grasp_offset_z": self.pre_grasp_offset_z,
                "max_place_approach_z": self.max_place_approach_z,
                "slot_index": slot_index,
                "held_object_id": self.memory.held_object_id,
                "held_height": self.held_object_height(),
            },
        }

    def held_object_height(self):
        if not self.memory.held_object_id:
            return self.default_block_height
        held = self.memory.objects.get(self.memory.held_object_id, {})
        return float(held.get("estimated_block_height", self.default_block_height))

    def place_object(self, target_object_id="", slot_index=None):
        pose = self.resolve_place_pose(target_object_id, slot_index)
        place_x = pose["x"]
        place_y = pose["y"]
        place_z = pose["place_z"]
        approach_z = pose["approach_z"]
        retreat_z = pose["retreat_z"]
        mode = pose["mode"]
        self.memory.append_log("INFO", "Resolved place pose", pose)
        self.send_pose_goal(place_x, place_y, approach_z, self.place_roll, self.place_pitch, self.place_yaw)
        self.send_pose_goal(place_x, place_y, place_z, self.place_roll, self.place_pitch, self.place_yaw)
        self.open_gripper()
        self.send_pose_goal(place_x, place_y, retreat_z, self.place_roll, self.place_pitch, self.place_yaw)
        released = self.memory.held_object_id
        self.memory.held_object_id = ""
        return {
            "success": True,
            "released_object_id": released,
            "place_xy": [place_x, place_y],
            "place_z": place_z,
            "approach_z": approach_z,
            "retreat_z": retreat_z,
            "height_components": pose["components"],
            "mode": mode,
        }

    def return_home(self):
        goal = SGRCtrlGoal()
        goal.grasp_type = goal.GRASP_NONE
        goal.action_type = goal.ACTION_TYPE_DEFINE_SAVE
        state = self.arm_client.send_goal_and_wait(goal, rospy.Duration.from_sec(30))
        return {"success": state == GoalStatus.SUCCEEDED, "state": state}
