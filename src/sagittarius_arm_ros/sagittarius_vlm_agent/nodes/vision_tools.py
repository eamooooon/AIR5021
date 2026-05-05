#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from block_proposer import annotate_proposals, propose_blocks
from object_selector import ObjectSelector


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
        self.selector = ObjectSelector(api_base, api_key, model, timeout, rospy_module=rospy)
        self.latest_proposals = []
        self.latest_proposal_image = None
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

    def save_or_show_proposals(self, frame, proposals, suffix="proposals", selected_id=""):
        if not self.debug_image and not self.memory.save_results:
            return ""
        vis = annotate_proposals(frame, proposals, selected_id=selected_id)
        if self.debug_image:
            cv2.imshow("vlm_agent_proposals", vis)
            cv2.waitKey(1)
        if self.memory.save_results:
            path = os.path.join(self.memory.get_results_dir(), "step_%03d_%s.jpg" % (self.memory.step, suffix))
            if not cv2.imwrite(path, vis):
                raise RuntimeError("Failed to write image to %s" % path)
            return path
        return ""

    def detect_object(self, query, robot_tools):
        detect_result = self.detect_objects(query, robot_tools)
        select_result = self.select_object(query)
        object_id = select_result["object_id"]
        return {
            "success": True,
            "object_id": object_id,
            "object": self.memory.objects[object_id],
            "objects": detect_result["objects"],
            "selection": select_result,
        }

    def detect_objects(self, query, robot_tools):
        frame = self.require_frame()
        proposal_result = propose_blocks(frame)
        proposals = proposal_result["proposals"]
        if not proposals:
            raise RuntimeError("No local block proposals found")
        detected = []
        for proposal in proposals:
            bbox, grasp_pixel = self.normalize_bbox_and_grasp(proposal["bbox"], proposal["grasp_pixel"], frame.shape)
            robot_xy = robot_tools.pixel_to_robot_xy(grasp_pixel[0], grasp_pixel[1])
            estimated_block_height = robot_tools.estimate_block_height(proposal)
            object_id = self.memory.add_object(
                label=proposal["id"],
                bbox=bbox,
                grasp_pixel=grasp_pixel,
                robot_xy=list(robot_xy),
                confidence=1.0,
                gripper_width=0.02,
                yaw_deg=proposal["yaw_deg"],
                rationale="Local block proposal generated from foreground segmentation.",
                object_id=proposal["id"],
                extra={
                    "proposal": proposal,
                    "yaw_reliable": proposal["yaw_reliable"],
                    "yaw_confidence": proposal["yaw_confidence"],
                    "mean_bgr": proposal["mean_bgr"],
                    "estimated_block_height": estimated_block_height,
                },
            )
            detected.append(self.memory.objects[object_id])
        self.latest_proposals = proposals
        self.latest_proposal_image = annotate_proposals(frame, proposals)
        annotated_path = self.save_or_show_proposals(frame, proposals, suffix="proposals")
        return {
            "success": True,
            "objects": detected,
            "proposal_count": len(detected),
            "annotated_image_path": annotated_path,
            "summary": "Generated %d local block proposals for query: %s" % (len(detected), query),
        }

    def select_object(self, query):
        frame = self.require_frame()
        if not self.latest_proposals:
            raise RuntimeError("No proposals available. Call detect_objects first.")
        annotated = annotate_proposals(frame, self.latest_proposals)
        result = self.selector.select(query, self.latest_proposals, annotated)
        object_id = result.get("object_id", "")
        if object_id not in self.memory.objects:
            raise RuntimeError("Selector returned invalid object_id=%s" % object_id)
        confidence = float(result.get("confidence", 0.0))
        selected = self.memory.objects[object_id]
        selected["selection_confidence"] = confidence
        selected["selection_rationale"] = result.get("rationale", "")
        selected_path = self.save_or_show_proposals(frame, self.latest_proposals, suffix="selected_%s" % object_id, selected_id=object_id)
        return {
            "success": True,
            "object_id": object_id,
            "confidence": confidence,
            "rationale": result.get("rationale", ""),
            "object": selected,
            "annotated_image_path": selected_path,
        }
