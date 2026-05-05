#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import threading

import actionlib
import cv2
import moveit_commander
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from sagittarius_object_color_detector.msg import SGRCtrlAction, SGRCtrlGoal
from sdk_sagittarius_arm.srv import ServoRtInfo, ServoRtInfoRequest
from vlm_grasp_planner import OpenAIVlmPlanner


class VlmGraspExecutor:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.robot_name = rospy.get_param("~robot_name", "sgr532")
        self.arm_name = rospy.get_param("~arm_name", self.robot_name)
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.prompt = rospy.get_param("~prompt", "pick up the blue object")
        self.api_key = rospy.get_param("~openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        self.api_base = rospy.get_param("~api_base", "https://api.openai.com/v1")
        self.model = rospy.get_param("~model", "gpt-5.4")
        self.request_timeout = float(self.get_config_param("request_timeout", 30.0))
        self.min_confidence = float(self.get_config_param("min_confidence", 0.35))
        self.debug_image = bool(self.get_config_param("debug_image", True))
        self.use_vlm_yaw = bool(self.get_config_param("use_vlm_yaw", False))
        self.default_pick_yaw = float(self.get_config_param("default_pick_yaw", 0.0))
        self.default_pick_pitch = float(self.get_config_param("default_pick_pitch", 1.57))
        self.default_pick_roll = float(self.get_config_param("default_pick_roll", 0.0))
        self.pick_z = float(self.get_config_param("pick_z", 0.020))
        self.pre_grasp_offset_z = float(self.get_config_param("pre_grasp_offset_z", 0.050))
        self.lift_offset_z = float(self.get_config_param("lift_offset_z", 0.080))
        self.search_x = float(self.get_config_param("search_x", 0.200))
        self.search_y = float(self.get_config_param("search_y", 0.000))
        self.search_z = float(self.get_config_param("search_z", 0.150))
        self.place_x = float(self.get_config_param("place_x", 0.000))
        self.place_y = float(self.get_config_param("place_y", 0.200))
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

        self.k1, self.b1, self.k2, self.b2 = self.load_linear_regression(rospy.get_param("~vision_config"))
        self.planner = OpenAIVlmPlanner(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.request_timeout,
            rospy_module=rospy,
        )

        self.arm_client = actionlib.SimpleActionClient(self.arm_name + "/sgr_ctrl", SGRCtrlAction)
        rospy.loginfo("Waiting for action server: %s/sgr_ctrl", self.arm_name)
        if not self.arm_client.wait_for_server(rospy.Duration.from_sec(self.action_wait_timeout)):
            raise RuntimeError(
                "Timed out waiting for action server %s/sgr_ctrl after %.1fs"
                % (self.arm_name, self.action_wait_timeout)
            )

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

    def get_config_param(self, name, default):
        private_name = "~" + name
        legacy_name = "~vlm_grasp/" + name
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
        pos_x = self.k1 * pixel_y + self.b1
        pos_y = self.k2 * pixel_x + self.b2
        return pos_x, pos_y

    def clip_gripper_width(self, width):
        return max(self.min_gripper_width, min(width, self.max_gripper_width))

    def set_gripper_width(self, width):
        width = self.clip_gripper_width(width)
        self.gripper.set_joint_value_target([width, width])
        success = self.gripper.go()
        if not success:
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

    def annotate(self, frame, plan):
        if not self.debug_image or plan.bbox is None:
            return
        x_min, y_min, x_max, y_max = [int(v) for v in plan.bbox]
        gx, gy = [int(v) for v in plan.grasp_pixel]
        vis = frame.copy()
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.circle(vis, (gx, gy), 4, (0, 0, 255), -1)
        label = "%s %.2f" % (plan.target, plan.confidence)
        cv2.putText(vis, label, (x_min, max(24, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("vlm_grasp_plan", vis)
        cv2.waitKey(1)

    def normalize_plan(self, plan, frame_shape):
        height, width = frame_shape[:2]
        if plan.bbox is None or plan.grasp_pixel is None:
            raise RuntimeError("VLM could not find the target object")
        if plan.confidence < self.min_confidence:
            raise RuntimeError("VLM confidence too low: %.3f" % plan.confidence)

        x_min, y_min, x_max, y_max = [int(round(v)) for v in plan.bbox]
        gx, gy = [int(round(v)) for v in plan.grasp_pixel]

        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width - 1))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height - 1))
        gx = max(0, min(gx, width - 1))
        gy = max(0, min(gy, height - 1))

        if x_max <= x_min or y_max <= y_min:
            raise RuntimeError("Invalid bbox after clipping")

        plan.bbox = [x_min, y_min, x_max, y_max]
        plan.grasp_pixel = [gx, gy]
        plan.gripper_width = min(max(plan.gripper_width, 0.0), 0.03)
        if plan.grasp_type != "top_grasp":
            rospy.logwarn("Unsupported grasp_type=%s, fallback to top_grasp", plan.grasp_type)
            plan.grasp_type = "top_grasp"
        return plan

    def execute(self):
        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or ~openai_api_key.")

        rospy.loginfo("Moving arm to search pose")
        self.send_pose_goal(
            self.search_x,
            self.search_y,
            self.search_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            self.default_pick_yaw,
        )

        rospy.loginfo("Waiting for camera frame on %s", self.image_topic)
        frame = self.wait_for_frame()
        rospy.loginfo("Camera frame received: %dx%d", frame.shape[1], frame.shape[0])
        rospy.loginfo("Requesting VLM plan with model=%s api_base=%s", self.model, self.api_base)
        plan = self.normalize_plan(self.planner.plan(frame, self.prompt, cv2), frame.shape)
        rospy.loginfo("VLM plan received")
        self.annotate(frame, plan)

        grasp_x, grasp_y = self.pixel_to_robot_xy(plan.grasp_pixel[0], plan.grasp_pixel[1])
        yaw = plan.yaw_rad() if self.use_vlm_yaw else self.default_pick_yaw
        open_width = self.open_gripper_width
        if open_width == 0.0:
            open_width = self.max_gripper_width

        rospy.loginfo(
            "VLM plan: target=%s bbox=%s grasp_pixel=%s type=%s width=%.4f yaw_deg=%.2f conf=%.3f",
            plan.target,
            plan.bbox,
            plan.grasp_pixel,
            plan.grasp_type,
            plan.gripper_width,
            plan.yaw_deg,
            plan.confidence,
        )
        rospy.loginfo(
            "Robot grasp pose: x=%.4f y=%.4f z=%.4f yaw=%.4f rationale=%s",
            grasp_x,
            grasp_y,
            self.pick_z,
            yaw,
            plan.rationale,
        )

        rospy.loginfo("Opening gripper to width %.4f", open_width)
        self.set_gripper_width(open_width)
        rospy.loginfo("Moving to pre-grasp pose")
        self.send_pose_goal(
            grasp_x,
            grasp_y,
            self.pick_z + self.pre_grasp_offset_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            yaw,
        )
        rospy.loginfo("Moving to grasp pose")
        self.send_pose_goal(
            grasp_x,
            grasp_y,
            self.pick_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            yaw,
        )

        rospy.loginfo("Closing gripper to width %.4f", self.grasp_close_width)
        self.set_gripper_width(self.grasp_close_width)
        if not self.verify_grasp():
            raise RuntimeError("Grasp verification failed: payload threshold not reached")

        rospy.loginfo("Lifting object")
        self.send_pose_goal(
            grasp_x,
            grasp_y,
            self.pick_z + self.lift_offset_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            yaw,
        )
        rospy.loginfo("Moving to place pose")
        self.send_pose_goal(
            self.place_x,
            self.place_y,
            self.place_z,
            self.place_roll,
            self.place_pitch,
            self.place_yaw,
        )
        rospy.loginfo("Releasing gripper to width %.4f", open_width)
        self.set_gripper_width(open_width)
        rospy.loginfo("Returning to search pose")
        self.send_pose_goal(
            self.search_x,
            self.search_y,
            self.search_z,
            self.default_pick_roll,
            self.default_pick_pitch,
            self.default_pick_yaw,
        )


def main():
    rospy.init_node("vlm_grasp_executor", anonymous=False)
    executor = VlmGraspExecutor()
    try:
        executor.execute()
        rospy.loginfo("VLM grasp execution finished")
    except Exception as exc:
        rospy.logerr("VLM grasp execution failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
