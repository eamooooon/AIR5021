#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import math
from dataclasses import dataclass


@dataclass
class VlmGraspPlan:
    target: str
    bbox: list
    grasp_pixel: list
    grasp_type: str
    gripper_width: float
    yaw_deg: float
    confidence: float
    rationale: str = ""

    @classmethod
    def from_dict(cls, payload):
        return cls(
            target=str(payload.get("target", "")),
            bbox=payload.get("bbox"),
            grasp_pixel=payload.get("grasp_pixel"),
            grasp_type=str(payload.get("grasp_type", "top_grasp")),
            gripper_width=float(payload.get("gripper_width", 0.020)),
            yaw_deg=float(payload.get("yaw_deg", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            rationale=str(payload.get("rationale", "")),
        )

    def yaw_rad(self):
        return math.radians(self.yaw_deg)


def encode_jpeg_base64(frame, cv2_module):
    ok, buffer = cv2_module.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode camera frame")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def safe_json_loads(text):
    text = text.strip()
    if not text:
        raise RuntimeError("Empty JSON payload from VLM")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise
