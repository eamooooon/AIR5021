#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import threading
import time
import json

import actionlib
import cv2
import moveit_commander
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from clean_desk_planner import OpenAICleanDeskPlanner
from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal
from sdk_sagittarius_arm.srv import ServoRtInfo, ServoRtInfoRequest


class CleanDeskExecutor:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.robot_name = rospy.get_param("~robot_name", "sgr532")
        self.arm_name = rospy.get_param("~arm_name", self.robot_name)
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.prompt = rospy.get_param(
            "~prompt",
            "clean the desk by picking all visible objects one by one and placing them into the storage area",
        )
        self.api_key = rospy.get_param("~openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        self.api_base = rospy.get_param("~api_base", "https://api.openai.com/v1")
        self.model = rospy.get_param("~model", "gpt-5.4")
        self.request_timeout = float(self.get_config_param("request_timeout", 30.0))
        self.min_confidence = float(self.get_config_param("min_confidence", 0.30))
        self.max_objects = int(self.get_config_param("max_objects", 6))
        self.debug_image = bool(self.get_config_param("debug_image", True))
        self.save_results = bool(self.get_config_param("save_results", True))
        self.results_dir = str(self.get_config_param("results_dir", "")).strip()
        self.default_pick_yaw = float(self.get_config_param("default_pick_yaw", 0.0))
        self.default_pick_pitch = float(self.get_config_param("default_pick_pitch", 1.57))
        self.default_pick_roll = float(self.get_config_param("default_pick_roll", 0.0))
        self.pick_z = float(self.get_config_param("pick_z", 0.020))
        self.pre_grasp_offset_z = float(self.get_config_param("pre_grasp_offset_z", 0.050))
        self.lift_offset_z = float(self.get_config_param("lift_offset_z", 0.080))
        self.observe_x = float(self.get_config_param("observe_x", 0.200))
        self.observe_y = float(self.get_config_param("observe_y", 0.000))
        self.observe_z = float(self.get_config_param("observe_z", 0.150))
        self.place_z = float(self.get_config_param("place_z", 0.050))
        self.place_yaw = float(self.get_config_param("place_yaw", 1.57))
        self.place_pitch = float(self.get_config_param("place_pitch", 1.57))
        self.place_roll = float(self.get_config_param("place_roll", 0.0))
        self.open_gripper_width = float(self.get_config_param("open_gripper_width", 0.0))
        self.min_gripper_width = float(self.get_config_param("min_gripper_width", -0.021))
        self.max_gripper_width = float(self.get_config_param("max_gripper_width", 0.0))
        self.grasp_close_width = float(self.get_config_param("grasp_close_width", -0.021))
        self.grasp_payload_threshold = int(self.get_config_param("grasp_payload_threshold", 24))
        self.action_wait_timeout = float(self.get_config_param("action_wait_timeout", 30.0))
        self.servo_info_wait_timeout = float(self.get_config_param("servo_info_wait_timeout", 15.0))
        self.move_group_wait_timeout = float(self.get_config_param("move_group_wait_timeout", 15.0))
        self.place_slots = [tuple(slot) for slot in self.get_config_param("place_slots", [[0.0, 0.2]])]
        self.run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.active_results_dir = None

        self.k1, self.b1, self.k2, self.b2 = self.load_linear_regression(rospy.get_param("~vision_config"))
        self.planner = OpenAICleanDeskPlanner(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.request_timeout,
            rospy_module=rospy,
        )

        self.arm_client = actionlib.SimpleActionClient(self.arm_name + "/sgr_ctrl", SGRCtrlAction)
        rospy.loginfo("Waiting for action server: %s/sgr_ctrl", self.arm_name)
        if not self.arm_client.wait_for_server(rospy.Duration.from_sec(self.action_wait_timeout)):
            raise RuntimeError("Timed out waiting for action server %s/sgr_ctrl" % self.arm_name)

        self.servo_info_service = self.resolve_robot_resource("get_servo_info")
        rospy.loginfo("Waiting for servo info service: %s", self.servo_info_service)
        rospy.wait_for_service(self.servo_info_service, self.servo_info_wait_timeout)
        self.servo_info_srv = rospy.ServiceProxy(self.servo_info_service, ServoRtInfo)

        moveit_commander.roscpp_initialize([])
        self.move_group_ns = self.resolve_robot_namespace()
        self.robot_description = self.resolve_robot_resource("robot_description")
        self.gripper = moveit_commander.MoveGroupCommander(
            "sagittarius_gripper",
            robot_description=self.robot_description,
            ns=self.move_group_ns,
            wait_for_servers=self.move_group_wait_timeout,
        )
        self.gripper.set_goal_joint_tolerance(0.001)
        self.gripper.set_max_acceleration_scaling_factor(0.5)
        self.gripper.set_max_velocity_scaling_factor(0.5)

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(
            "Result saving config: save_results=%s results_dir=%s",
            self.save_results,
            self.results_dir if self.results_dir else "<default under ROS_HOME>",
        )

    def get_config_param(self, name, default):
        private_name = "~" + name
        legacy_name = "~clean_desk/" + name
        if rospy.has_param(private_name):
            return rospy.get_param(private_name)
        if rospy.has_param(legacy_name):
            return rospy.get_param(legacy_name)
        return default

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

    def load_linear_regression(self, path):
        with open(path, "r") as handle:
            content = yaml.safe_load(handle)
        linear = content["LinearRegression"]
        return linear["k1"], linear["b1"], linear["k2"], linear["b2"]

    def get_results_dir(self):
        if self.active_results_dir is not None:
            return self.active_results_dir

        if self.results_dir:
            base_dir = self.results_dir
        else:
            base_dir = os.path.join(os.getenv("ROS_HOME", os.path.expanduser("~/.ros")), "clean_desk_results")
        run_dir = os.path.join(base_dir, self.run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        self.active_results_dir = run_dir
        rospy.loginfo("Resolved result directory: %s", self.active_results_dir)
        return self.active_results_dir

    def write_image_or_raise(self, path, frame):
        ok = cv2.imwrite(path, frame)
        if not ok:
            raise RuntimeError("Failed to write image to %s" % path)

    def save_raw_observation(self, raw_frame):
        if not self.save_results:
            return
        results_dir = self.get_results_dir()
        raw_path = os.path.join(results_dir, "observation_raw.jpg")
        self.write_image_or_raise(raw_path, raw_frame)
        rospy.loginfo("Saved raw observation image to %s", raw_path)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logerr("Failed to decode image: %s", exc)
            return
        with self.frame_lock:
            self.latest_frame = frame

    def wait_for_frame(self, timeout_sec=10.0):
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout_sec)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            rate.sleep()
        raise RuntimeError("Timed out waiting for image on %s" % self.image_topic)

    def pixel_to_robot_xy(self, pixel_x, pixel_y):
        return self.k1 * pixel_y + self.b1, self.k2 * pixel_x + self.b2

    def clip_gripper_width(self, width):
        return max(self.min_gripper_width, min(width, self.max_gripper_width))

    def set_gripper_width(self, width):
        width = self.clip_gripper_width(width)
        self.gripper.set_joint_value_target([width, width])
        if not self.gripper.go():
            raise RuntimeError("Failed to move gripper to width %.4f" % width)
        rospy.sleep(1.0)

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
        self.arm_client.send_goal_and_wait(goal, rospy.Duration.from_sec(30))

    def verify_grasp(self):
        result = self.servo_info_srv.call(ServoRtInfoRequest(servo_id=7))
        rospy.loginfo("Gripper payload feedback: %s", result.payload)
        return abs(result.payload) >= self.grasp_payload_threshold

    def annotate(self, frame, objects, summary=""):
        vis = frame.copy()
        cv2.putText(
            vis,
            "Detected objects: %d" % len(objects),
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        if summary:
            cv2.putText(
                vis,
                summary[:90],
                (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        for index, obj in enumerate(objects, start=1):
            x_min, y_min, x_max, y_max = [int(v) for v in obj.bbox]
            gx, gy = [int(v) for v in obj.grasp_pixel]
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(vis, (gx, gy), 4, (0, 0, 255), -1)
            label = "#%d %s %.2f" % (index, obj.label, obj.confidence)
            cv2.putText(vis, label, (x_min, max(24, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            slot_text = "slot[%d] -> (%.2f, %.2f)" % (
                index - 1,
                self.place_slots[min(index - 1, len(self.place_slots) - 1)][0],
                self.place_slots[min(index - 1, len(self.place_slots) - 1)][1],
            )
            cv2.putText(
                vis,
                slot_text,
                (x_min, min(y_max + 18, vis.shape[0] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
            )
        if self.debug_image:
            cv2.imshow("clean_desk_plan", vis)
            cv2.waitKey(1)
        return vis

    def save_plan_artifacts(self, raw_frame, annotated_frame, plan, objects):
        if not self.save_results:
            return

        results_dir = self.get_results_dir()
        raw_path = os.path.join(results_dir, "observation_raw.jpg")
        annotated_path = os.path.join(results_dir, "observation_annotated.jpg")
        json_path = os.path.join(results_dir, "plan.json")

        self.write_image_or_raise(raw_path, raw_frame)
        self.write_image_or_raise(annotated_path, annotated_frame)

        plan_payload = {
            "prompt": self.prompt,
            "summary": plan.summary,
            "detected_count": len(objects),
            "model": self.model,
            "api_base": self.api_base,
            "objects": [],
        }
        for index, obj in enumerate(objects):
            grasp_x, grasp_y = self.pixel_to_robot_xy(obj.grasp_pixel[0], obj.grasp_pixel[1])
            slot_x, slot_y = self.place_slots[min(index, len(self.place_slots) - 1)]
            plan_payload["objects"].append(
                {
                    "index": index + 1,
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                    "grasp_pixel": obj.grasp_pixel,
                    "grasp_xy": [grasp_x, grasp_y],
                    "place_slot_index": index,
                    "place_xy": [slot_x, slot_y],
                    "rationale": obj.rationale,
                }
            )

        with open(json_path, "w") as handle:
            json.dump(plan_payload, handle, indent=2, ensure_ascii=False)

        rospy.loginfo("Saved clean-desk artifacts to %s", results_dir)

    def normalize_objects(self, objects, frame_shape):
        height, width = frame_shape[:2]
        normalized = []
        for obj in objects:
            if obj.bbox is None or obj.grasp_pixel is None:
                rospy.logwarn("Skip object with empty bbox/grasp_pixel: %s", obj.label)
                continue
            if obj.confidence < self.min_confidence:
                rospy.logwarn("Skip low-confidence object %s: %.3f", obj.label, obj.confidence)
                continue

            x_min, y_min, x_max, y_max = [int(round(v)) for v in obj.bbox]
            gx, gy = [int(round(v)) for v in obj.grasp_pixel]
            x_min = max(0, min(x_min, width - 1))
            x_max = max(0, min(x_max, width - 1))
            y_min = max(0, min(y_min, height - 1))
            y_max = max(0, min(y_max, height - 1))
            gx = max(0, min(gx, width - 1))
            gy = max(0, min(gy, height - 1))
            if x_max <= x_min or y_max <= y_min:
                rospy.logwarn("Skip invalid bbox for object %s", obj.label)
                continue

            obj.bbox = [x_min, y_min, x_max, y_max]
            obj.grasp_pixel = [gx, gy]
            obj.gripper_width = min(max(obj.gripper_width, 0.0), 0.03)
            normalized.append(obj)

        normalized.sort(key=lambda item: (item.grasp_pixel[1], item.grasp_pixel[0]))
        return normalized[: self.max_objects]

    def execute_pick_and_place(self, obj, slot_index):
        grasp_x, grasp_y = self.pixel_to_robot_xy(obj.grasp_pixel[0], obj.grasp_pixel[1])
        slot_x, slot_y = self.place_slots[slot_index]
        open_width = self.max_gripper_width if self.open_gripper_width == 0.0 else self.open_gripper_width
        yaw = self.default_pick_yaw

        rospy.loginfo(
            "Executing object %d/%d: %s bbox=%s grasp_pixel=%s conf=%.3f",
            slot_index + 1,
            len(self.place_slots),
            obj.label,
            obj.bbox,
            obj.grasp_pixel,
            obj.confidence,
        )

        self.set_gripper_width(open_width)
        self.send_pose_goal(grasp_x, grasp_y, self.pick_z + self.pre_grasp_offset_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.send_pose_goal(grasp_x, grasp_y, self.pick_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.set_gripper_width(self.grasp_close_width)
        if not self.verify_grasp():
            raise RuntimeError("Grasp verification failed for %s" % obj.label)

        self.send_pose_goal(grasp_x, grasp_y, self.pick_z + self.lift_offset_z, self.default_pick_roll, self.default_pick_pitch, yaw)
        self.send_pose_goal(slot_x, slot_y, self.place_z + self.pre_grasp_offset_z, self.place_roll, self.place_pitch, self.place_yaw)
        self.send_pose_goal(slot_x, slot_y, self.place_z, self.place_roll, self.place_pitch, self.place_yaw)
        self.set_gripper_width(open_width)
        self.send_pose_goal(slot_x, slot_y, self.place_z + self.lift_offset_z, self.place_roll, self.place_pitch, self.place_yaw)

    def execute(self):
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or ~openai_api_key.")
        if not self.place_slots:
            raise RuntimeError("No place_slots configured")

        rospy.loginfo("Moving to observation pose for one-shot desk scan")
        self.send_pose_goal(
            self.observe_x,
            self.observe_y,
            self.observe_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            self.default_pick_yaw,
        )

        frame = self.wait_for_frame()
        rospy.loginfo("Captured observation frame: %dx%d", frame.shape[1], frame.shape[0])
        self.save_raw_observation(frame)
        plan = self.planner.plan(frame, self.prompt, cv2)
        objects = self.normalize_objects(plan.objects, frame.shape)
        if not objects:
            raise RuntimeError("VLM returned no valid objects for clean-desk task")
        if len(objects) > len(self.place_slots):
            rospy.logwarn("Detected %d objects but only %d place slots configured; truncating queue", len(objects), len(self.place_slots))
            objects = objects[: len(self.place_slots)]

        annotated = self.annotate(frame, objects, plan.summary)
        self.save_plan_artifacts(frame, annotated, plan, objects)
        rospy.loginfo("Desk cleaning queue summary: %s", plan.summary)
        rospy.loginfo("Desk cleaning queue: %s", [obj.label for obj in objects])

        for index, obj in enumerate(objects):
            self.execute_pick_and_place(obj, index)

        rospy.loginfo("Desk cleaning task finished without returning to observation pose")


def main():
    rospy.init_node("clean_desk_executor", anonymous=False)
    executor = CleanDeskExecutor()
    try:
        executor.execute()
    except Exception as exc:
        rospy.logerr("Clean desk execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
