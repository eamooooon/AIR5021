#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hand_eye_calibration import HandEyeCalibration, estimate_eye_to_hand, invert_transform, make_transform


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    rotation, _ = cv2.Rodrigues(axis.reshape(3, 1) * float(angle))
    return rotation


def synthetic_samples(cam_to_base, tool_to_cal, count):
    samples = []
    for index in range(count):
        rotation = rodrigues([1.0 + (index % 2), 0.5 + index, 0.8], 0.18 + 0.12 * index)
        translation = np.array(
            [0.14 + 0.025 * index, -0.08 + 0.018 * index, 0.10 + 0.01 * ((index * 3) % 5)],
            dtype=np.float64,
        )
        base_to_tool = make_transform(rotation, translation)
        cam_to_cal = cam_to_base.dot(base_to_tool).dot(invert_transform(tool_to_cal))
        samples.append({"base_H_tool": base_to_tool, "cam_H_cal": cam_to_cal})
    return samples


def assert_close(name, actual, expected, tolerance):
    error = float(np.linalg.norm(actual - expected))
    print("%s error: %.10f" % (name, error))
    if error > tolerance:
        raise AssertionError("%s error %.10f > %.10f" % (name, error, tolerance))


def main():
    base_to_cam = make_transform(
        rodrigues([0.0, 0.0, 1.0], math.radians(16.0)).dot(np.diag([1.0, -1.0, -1.0])),
        [0.12, -0.04, 0.35],
    )
    cam_to_base = invert_transform(base_to_cam)
    tool_to_cal = make_transform(rodrigues([0.0, 1.0, 0.0], math.radians(6.0)), [0.012, 0.018, 0.028])

    two_sample_estimate, per_sample = estimate_eye_to_hand(synthetic_samples(cam_to_base, tool_to_cal, 2), tool_to_cal=tool_to_cal)
    assert_close("two-sample translation", two_sample_estimate[:3, 3], cam_to_base[:3, 3], 1e-9)
    assert_close("two-sample rotation", two_sample_estimate[:3, :3], cam_to_base[:3, :3], 1e-9)
    print("per-sample estimates: %d" % len(per_sample))

    axxb_estimate, _ = estimate_eye_to_hand(synthetic_samples(cam_to_base, tool_to_cal, 6), tool_to_cal=None)
    assert_close("AX=XB translation", axxb_estimate[:3, 3], cam_to_base[:3, 3], 1e-8)
    assert_close("AX=XB rotation", axxb_estimate[:3, :3], cam_to_base[:3, :3], 1e-8)

    camera_matrix = np.array([[620.0, 0.0, 320.0], [0.0, 620.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    calibration = HandEyeCalibration(camera_matrix, [0.0, 0.0, 0.0, 0.0, 0.0], cam_to_base, base_plane_z=0.02)
    point = calibration.pixel_to_base_point(320, 240)
    print("center pixel on base plane: [%.6f, %.6f, %.6f]" % (point[0], point[1], point[2]))
    if abs(point[2] - 0.02) > 1e-9:
        raise AssertionError("pixel ray did not land on requested base plane")


if __name__ == "__main__":
    main()
