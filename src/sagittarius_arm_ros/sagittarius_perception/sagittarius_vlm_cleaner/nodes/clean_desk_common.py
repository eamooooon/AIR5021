#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
from dataclasses import dataclass


@dataclass
class DeskObject:
    label: str
    bbox: list
    grasp_pixel: list
    confidence: float
    gripper_width: float = 0.02
    yaw_deg: float = 0.0
    rationale: str = ""

    @classmethod
    def from_dict(cls, payload):
        return cls(
            label=str(payload.get("label", "")),
            bbox=payload.get("bbox"),
            grasp_pixel=payload.get("grasp_pixel"),
            confidence=float(payload.get("confidence", 0.0)),
            gripper_width=float(payload.get("gripper_width", 0.02)),
            yaw_deg=float(payload.get("yaw_deg", 0.0)),
            rationale=str(payload.get("rationale", "")),
        )


@dataclass
class DeskCleaningPlan:
    objects: list
    summary: str = ""

    @classmethod
    def from_dict(cls, payload):
        objects = [DeskObject.from_dict(item) for item in payload.get("objects", [])]
        return cls(objects=objects, summary=str(payload.get("summary", "")))


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
