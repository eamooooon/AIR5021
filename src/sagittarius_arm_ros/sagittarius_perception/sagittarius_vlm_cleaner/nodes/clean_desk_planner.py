#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import socket
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from clean_desk_common import DeskCleaningPlan, encode_jpeg_base64, safe_json_loads


SYSTEM_PROMPT = """
You are a robot tabletop cleaning planner for a 6-axis arm with a parallel gripper.
You will receive one RGB image and one natural-language instruction.
Identify every visible movable tabletop object that should be cleaned from the desk.

Return exactly one JSON object, with no markdown and no extra text.
Schema:
{
  "objects": [
    {
      "label": "string",
      "bbox": [x_min, y_min, x_max, y_max],
      "grasp_pixel": [x, y],
      "confidence": 0.0,
      "gripper_width": 0.020,
      "yaw_deg": 0.0,
      "rationale": "short reason"
    }
  ],
  "summary": "short sentence"
}

Rules:
- Return all visible movable tabletop objects relevant to cleaning the desk.
- Sort objects from nearest/front-most to farthest/back-most if possible. If unclear, sort left-to-right.
- bbox and grasp_pixel must use image pixel coordinates.
- confidence must be in [0, 1].
- gripper_width must be in [0.0, 0.03].
- yaw_deg must be in [-180, 180].
- If nothing should be picked, return {"objects": [], "summary": "..."}.
""".strip()


class OpenAICleanDeskPlanner:
    def __init__(self, api_base, api_key, model, timeout, rospy_module=None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.rospy = rospy_module

    def _logwarn(self, message):
        if self.rospy is not None:
            self.rospy.logwarn(message)

    def _response_body(self, exc):
        try:
            return exc.read().decode("utf-8")
        except Exception:
            return ""

    def _extract_responses_output_text(self, response):
        output_text = response.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    text = content.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                if content.get("type") == "json_schema":
                    text = content.get("json", "")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        return ""

    def _extract_chat_content_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        chunks.append(item["text"])
                    elif isinstance(item.get("text"), dict) and isinstance(item["text"].get("value"), str):
                        chunks.append(item["text"]["value"])
            if chunks:
                return "\n".join(chunks)
        raise RuntimeError("Unsupported chat completion content format: %r" % type(content).__name__)

    def _use_responses_first(self):
        return self.api_base.rstrip("/") == "https://api.openai.com/v1"

    def _post_json(self, endpoint, payload):
        request = urllib.request.Request(
            self.api_base + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _plan_with_responses(self, payload):
        response = self._post_json("/responses", payload)
        output_text = self._extract_responses_output_text(response)
        if not output_text:
            raise RuntimeError("Responses API returned no output_text")
        return DeskCleaningPlan.from_dict(safe_json_loads(output_text))

    def _plan_with_chat_completions(self, payload):
        response = self._post_json("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        content = self._extract_chat_content_text(content)
        return DeskCleaningPlan.from_dict(safe_json_loads(content))

    def plan(self, frame, prompt, cv2_module):
        image_b64 = encode_jpeg_base64(frame, cv2_module)
        responses_payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Instruction: %s\nReturn only the JSON object." % prompt},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,%s" % image_b64},
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        chat_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Instruction: %s\nReturn only the JSON object." % prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % image_b64}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }

        if self._use_responses_first():
            attempts = [
                ("Responses API", self._plan_with_responses, responses_payload),
                ("Chat Completions API", self._plan_with_chat_completions, chat_payload),
            ]
        else:
            attempts = [
                ("Chat Completions API", self._plan_with_chat_completions, chat_payload),
                ("Responses API", self._plan_with_responses, responses_payload),
            ]

        errors = []
        for label, method, payload in attempts:
            try:
                self._logwarn("Trying %s via %s" % (label, self.api_base))
                return method(payload)
            except urllib.error.HTTPError as exc:
                message = "%s failed with HTTP %s: %s" % (label, exc.code, self._response_body(exc))
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                message = "%s timed out or could not connect: %s" % (label, exc)
            except Exception as exc:
                message = "%s failed: %s" % (label, exc)
            errors.append(message)
            self._logwarn(message)

        raise RuntimeError(" ; ".join(errors))
