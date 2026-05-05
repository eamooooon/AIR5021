#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CameraIntrinsicsCalibration:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.output_path = rospy.get_param(
            "~output_path",
            os.path.join(os.path.expanduser("~"), ".ros", "sagittarius_camera_intrinsics.yaml"),
        )
        self.inner_cols = int(rospy.get_param("~inner_cols", 4))
        self.inner_rows = int(rospy.get_param("~inner_rows", 5))
        self.square_size = float(rospy.get_param("~square_size", 0.020))
        self.min_samples = int(rospy.get_param("~min_samples", 12))
        self.preview = bool(rospy.get_param("~preview", True))

        self.bridge = CvBridge()
        self.object_points = []
        self.image_points = []
        self.image_size = None
        self.latest_frame = None
        self.latest_corners = None
        self.latest_found = False

        objp = np.zeros((self.inner_rows * self.inner_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.inner_cols, 0:self.inner_rows].T.reshape(-1, 2)
        objp *= self.square_size
        self.pattern_points = objp

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(
            "Camera intrinsic calibration started: topic=%s board_inner=%dx%d square=%.3fm output=%s",
            self.image_topic,
            self.inner_cols,
            self.inner_rows,
            self.square_size,
            self.output_path,
        )
        rospy.loginfo("Keys in preview window: s=save sample, c=calibrate, q=quit")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logerr("Failed to decode image: %s", exc)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray,
            (self.inner_cols, self.inner_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        self.latest_frame = frame
        self.latest_corners = corners
        self.latest_found = bool(found)

    def save_sample(self):
        if not self.latest_found or self.latest_corners is None:
            rospy.logwarn("No complete chessboard detected; sample not saved")
            return False
        self.object_points.append(self.pattern_points.copy())
        self.image_points.append(self.latest_corners.copy())
        rospy.loginfo("Saved sample %d/%d", len(self.object_points), self.min_samples)
        return True

    def calibrate(self):
        if len(self.object_points) < 3:
            raise RuntimeError("Need at least 3 samples; %d saved" % len(self.object_points))
        if self.image_size is None:
            raise RuntimeError("No image size available")

        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None,
        )
        content = {
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "board_squares": [int(self.inner_cols + 1), int(self.inner_rows + 1)],
            "board_inner_corners": [int(self.inner_cols), int(self.inner_rows)],
            "square_size": float(self.square_size),
            "sample_count": int(len(self.object_points)),
            "rms_reprojection_error": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        }

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output_path, "w") as handle:
            yaml.safe_dump(content, handle, default_flow_style=False, sort_keys=False)

        rospy.loginfo("Calibration saved to %s", self.output_path)
        rospy.loginfo("RMS reprojection error: %.6f", rms)
        rospy.loginfo("camera_matrix: %s", camera_matrix.tolist())
        rospy.loginfo("dist_coeffs: %s", dist_coeffs.reshape(-1).tolist())
        return content

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.preview and self.latest_frame is not None:
                display = self.latest_frame.copy()
                if self.latest_found and self.latest_corners is not None:
                    cv2.drawChessboardCorners(
                        display,
                        (self.inner_cols, self.inner_rows),
                        self.latest_corners,
                        self.latest_found,
                    )
                status = "found" if self.latest_found else "not found"
                text = "samples=%d  board=%s  s:save c:calibrate q:quit" % (
                    len(self.object_points),
                    status,
                )
                cv2.putText(display, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.imshow("camera_intrinsics_calibration", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    self.save_sample()
                elif key == ord("c"):
                    try:
                        self.calibrate()
                    except Exception as exc:
                        rospy.logerr("Calibration failed: %s", exc)
                elif key == ord("q"):
                    rospy.signal_shutdown("User requested quit")
            rate.sleep()
        cv2.destroyAllWindows()


def main():
    rospy.init_node("camera_intrinsics_calibration", anonymous=False)
    node = CameraIntrinsicsCalibration()
    node.spin()


if __name__ == "__main__":
    main()
