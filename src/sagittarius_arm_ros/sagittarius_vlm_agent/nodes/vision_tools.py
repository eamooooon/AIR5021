#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAGITTARIUS_ARM_ROS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PERCEPTION_DIR = os.path.join(SAGITTARIUS_ARM_ROS_DIR, "sagittarius_perception")
OBJECT_NODES = os.path.join(PERCEPTION_DIR, "sagittarius_object_color_detector", "nodes")
CLEANER_NODES = os.path.join(PERCEPTION_DIR, "sagittarius_vlm_cleaner", "nodes")
for path in (OBJECT_NODES, CLEANER_NODES):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlm_grasp_planner import OpenAIVlmPlanner
from clean_desk_planner import OpenAICleanDeskPlanner


class VisionTools:
    def __init__(self, memory, api_base, api_key, model, timeout, image_topic, min_confidence, debug_image):
        self.memory = memory
        self.bridge = CvBridge()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.current_frame = None
        self.image_topic = image_topic
        self.min_confidence = min_confidence
        self.debug_image = debug_image
        self.single_planner = OpenAIVlmPlanner(api_base, api_key, model, timeout, rospy_module=rospy)
        self.multi_planner = OpenAICleanDeskPlanner(api_base, api_key, model, timeout, rospy_module=rospy)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

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

    def capture_image(self):
        frame = self.wait_for_frame()
        self.current_frame = frame
        step = self.memory.step
        path = ""
        if self.memory.save_results:
            path = os.path.join(self.memory.get_results_dir(), "step_%03d_raw.jpg" % step)
            if not cv2.imwrite(path, frame):
                raise RuntimeError("Failed to write image to %s" % path)
            self.memory.latest_image_path = path
        return {
            "success": True,
            "image_path": path,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

    def require_frame(self):
        if self.current_frame is None:
            self.capture_image()
        return self.current_frame

    def normalize_bbox_and_grasp(self, bbox, grasp_pixel, frame_shape):
        height, width = frame_shape[:2]
        if bbox is None or grasp_pixel is None:
            raise RuntimeError("VLM returned empty bbox/grasp_pixel")
        x_min, y_min, x_max, y_max = [int(round(v)) for v in bbox]
        gx, gy = [int(round(v)) for v in grasp_pixel]
        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width - 1))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height - 1))
        gx = max(0, min(gx, width - 1))
        gy = max(0, min(gy, height - 1))
        if x_max <= x_min or y_max <= y_min:
            raise RuntimeError("Invalid bbox after clipping")
        return [x_min, y_min, x_max, y_max], [gx, gy]

    def annotate_object(self, frame, label, bbox, grasp_pixel, confidence, suffix):
        if not self.debug_image and not self.memory.save_results:
            return ""
        vis = frame.copy()
        x_min, y_min, x_max, y_max = bbox
        gx, gy = grasp_pixel
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.circle(vis, (gx, gy), 4, (0, 0, 255), -1)
        cv2.putText(vis, "%s %.2f" % (label, confidence), (x_min, max(24, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.debug_image:
            cv2.imshow("vlm_agent_detection", vis)
            cv2.waitKey(1)
        if self.memory.save_results:
            path = os.path.join(self.memory.get_results_dir(), "step_%03d_%s.jpg" % (self.memory.step, suffix))
            if not cv2.imwrite(path, vis):
                raise RuntimeError("Failed to write image to %s" % path)
            return path
        return ""

    def detect_object(self, query, robot_tools):
        frame = self.require_frame()
        plan = self.single_planner.plan(frame, query, cv2)
        if plan.confidence < self.min_confidence:
            raise RuntimeError("VLM confidence too low for %s: %.3f" % (query, plan.confidence))
        bbox, grasp_pixel = self.normalize_bbox_and_grasp(plan.bbox, plan.grasp_pixel, frame.shape)
        robot_xy = robot_tools.pixel_to_robot_xy(grasp_pixel[0], grasp_pixel[1])
        object_id = self.memory.add_object(
            label=plan.target or query,
            bbox=bbox,
            grasp_pixel=grasp_pixel,
            robot_xy=list(robot_xy),
            confidence=plan.confidence,
            gripper_width=min(max(plan.gripper_width, 0.0), 0.03),
            yaw_deg=plan.yaw_deg,
            rationale=plan.rationale,
        )
        annotated_path = self.annotate_object(frame, plan.target or query, bbox, grasp_pixel, plan.confidence, object_id)
        return {
            "success": True,
            "object_id": object_id,
            "object": self.memory.objects[object_id],
            "annotated_image_path": annotated_path,
        }

    def detect_objects(self, query, robot_tools):
        frame = self.require_frame()
        plan = self.multi_planner.plan(frame, query, cv2)
        detected = []
        for obj in plan.objects:
            if obj.confidence < self.min_confidence:
                continue
            try:
                bbox, grasp_pixel = self.normalize_bbox_and_grasp(obj.bbox, obj.grasp_pixel, frame.shape)
            except RuntimeError as exc:
                rospy.logwarn("Skipping invalid object %s: %s", obj.label, exc)
                continue
            robot_xy = robot_tools.pixel_to_robot_xy(grasp_pixel[0], grasp_pixel[1])
            object_id = self.memory.add_object(
                label=obj.label,
                bbox=bbox,
                grasp_pixel=grasp_pixel,
                robot_xy=list(robot_xy),
                confidence=obj.confidence,
                gripper_width=min(max(obj.gripper_width, 0.0), 0.03),
                yaw_deg=obj.yaw_deg,
                rationale=obj.rationale,
            )
            detected.append(self.memory.objects[object_id])
        return {
            "success": True,
            "objects": detected,
            "summary": plan.summary,
        }
