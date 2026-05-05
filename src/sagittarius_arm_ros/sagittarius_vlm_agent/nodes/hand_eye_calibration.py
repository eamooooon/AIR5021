#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import cv2
import numpy as np
import yaml


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T.dot(translation)
    return inverse


def rotation_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz.dot(ry).dot(rx)


def quaternion_to_rotation(quaternion):
    x, y, z, w = [float(v) for v in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_transform(pose):
    if "matrix" in pose:
        return np.asarray(pose["matrix"], dtype=np.float64).reshape(4, 4)
    translation = pose.get("translation", pose.get("xyz", [0.0, 0.0, 0.0]))
    if "rotation_matrix" in pose:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    elif "rpy" in pose:
        rotation = rotation_from_rpy(*[float(v) for v in pose["rpy"]])
    elif "quaternion_xyzw" in pose:
        rotation = quaternion_to_rotation(pose["quaternion_xyzw"])
    elif "quaternion" in pose:
        rotation = quaternion_to_rotation(pose["quaternion"])
    elif "rvec" in pose:
        rotation, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1))
    else:
        rotation = np.eye(3, dtype=np.float64)
    return make_transform(rotation, translation)


def transform_to_dict(transform):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return {
        "matrix": transform.tolist(),
        "translation": transform[:3, 3].tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
    }


def average_transforms(transforms):
    if not transforms:
        raise ValueError("No transforms to average")
    translations = np.array([transform[:3, 3] for transform in transforms], dtype=np.float64)
    rotations = np.array([transform[:3, :3] for transform in transforms], dtype=np.float64)
    rotation_mean = rotations.mean(axis=0)
    u, _, vt = np.linalg.svd(rotation_mean)
    rotation = u.dot(vt)
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u.dot(vt)
    return make_transform(rotation, translations.mean(axis=0))


def marker_object_points(marker_size):
    half = float(marker_size) / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def solve_cam_to_cal_from_corners(corners, marker_size, camera_matrix, dist_coeffs):
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if image_points.shape[0] != 4:
        raise ValueError("Expected four marker corners")
    object_points = marker_object_points(marker_size)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    return make_transform(rotation, tvec.reshape(3))


def estimate_eye_to_hand(samples, tool_to_cal=None):
    if len(samples) < 2:
        raise ValueError("At least two calibration samples are required for a useful estimate")

    base_to_tool = [sample["base_H_tool"] for sample in samples]
    cam_to_cal = [sample["cam_H_cal"] for sample in samples]

    if tool_to_cal is not None:
        estimates = [
            cam_to_cal[i].dot(tool_to_cal).dot(invert_transform(base_to_tool[i]))
            for i in range(len(samples))
        ]
        return average_transforms(estimates), estimates

    if len(samples) < 3:
        raise ValueError("AX=XB hand-eye calibration needs at least three diverse poses when tool_H_cal is unknown")

    tool_to_base = [invert_transform(transform) for transform in base_to_tool]
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in tool_to_base],
        [transform[:3, 3] for transform in tool_to_base],
        [transform[:3, :3] for transform in cam_to_cal],
        [transform[:3, 3] for transform in cam_to_cal],
        method=cv2.CALIB_HAND_EYE_HORAUD,
    )
    base_to_cam = make_transform(rotation, np.asarray(translation, dtype=np.float64).reshape(3))
    return invert_transform(base_to_cam), []


class HandEyeCalibration:
    def __init__(self, camera_matrix, dist_coeffs, cam_to_base, base_plane_z=0.0, source_path=""):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self.cam_H_base = np.asarray(cam_to_base, dtype=np.float64).reshape(4, 4)
        self.base_H_cam = invert_transform(self.cam_H_base)
        self.base_plane_z = float(base_plane_z)
        self.source_path = source_path

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as handle:
            content = yaml.safe_load(handle) or {}
        if not bool(content.get("enabled", True)):
            raise RuntimeError("hand-eye calibration is disabled in YAML")
        camera_matrix = content["camera_matrix"]
        dist_coeffs = content.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        base_plane_z = content.get("base_plane_z", 0.0)
        if "cam_H_base" in content:
            cam_to_base = pose_to_transform(content["cam_H_base"])
        elif "base_H_cam" in content:
            cam_to_base = invert_transform(pose_to_transform(content["base_H_cam"]))
        else:
            cam_to_base = build_calibration_from_yaml_content(content)["cam_H_base"]
        return cls(camera_matrix, dist_coeffs, cam_to_base, base_plane_z=base_plane_z, source_path=path)

    def pixel_to_base_point(self, pixel_x, pixel_y, plane_z=None):
        if plane_z is None:
            plane_z = self.base_plane_z
        points = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs)
        ray_cam = np.array([undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0], dtype=np.float64)
        origin_base = self.base_H_cam[:3, 3]
        direction_base = self.base_H_cam[:3, :3].dot(ray_cam)
        if abs(direction_base[2]) < 1e-9:
            raise RuntimeError("Camera ray is parallel to base plane")
        scale = (float(plane_z) - origin_base[2]) / direction_base[2]
        if scale <= 0.0:
            raise RuntimeError("Camera ray intersects the base plane behind the camera")
        return origin_base + scale * direction_base

    def pixel_to_robot_xy(self, pixel_x, pixel_y, plane_z=None):
        point = self.pixel_to_base_point(pixel_x, pixel_y, plane_z=plane_z)
        return float(point[0]), float(point[1])


def build_calibration_from_yaml_content(content):
    camera_matrix = np.asarray(content["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(content.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    marker_size = float(content.get("marker_size", 0.04))
    tool_to_cal = pose_to_transform(content["tool_H_cal"]) if "tool_H_cal" in content else None
    samples = []
    for raw in content.get("samples", []):
        cam_to_cal = pose_to_transform(raw["cam_H_cal"]) if "cam_H_cal" in raw else solve_cam_to_cal_from_corners(
            raw["corners"], marker_size, camera_matrix, dist_coeffs
        )
        samples.append({"base_H_tool": pose_to_transform(raw["base_H_tool"]), "cam_H_cal": cam_to_cal})
    cam_to_base, per_sample = estimate_eye_to_hand(samples, tool_to_cal=tool_to_cal)
    return {
        "cam_H_base": cam_to_base,
        "per_sample_cam_H_base": per_sample,
    }


def write_calibration_yaml(input_path, output_path):
    with open(input_path, "r") as handle:
        content = yaml.safe_load(handle) or {}
    result = build_calibration_from_yaml_content(content)
    content["cam_H_base"] = transform_to_dict(result["cam_H_base"])
    content["enabled"] = True
    if result["per_sample_cam_H_base"]:
        content["per_sample_cam_H_base"] = [transform_to_dict(transform) for transform in result["per_sample_cam_H_base"]]
    with open(output_path, "w") as handle:
        yaml.safe_dump(content, handle, default_flow_style=False, sort_keys=False)
    return result
