#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import socket
import urllib.error
import urllib.request

import cv2

from agent_common import compact_for_prompt, safe_json_loads


SYSTEM_PROMPT = """
You are selecting a robot grasp target from pre-detected object proposals.
You must not invent pixel coordinates or new object IDs.
Choose exactly one object_id from the provided proposals.

Return exactly one JSON object, with no markdown and no extra text:
{
  "object_id": "obj_1",
  "confidence": 0.0,
  "rationale": "short reason"
}

Rules:
- Select only from the proposal IDs shown in the image and proposal list.
- Use the user's query, object appearance, position, and relationships to choose.
- If two objects have similar colors, use position, relative location, shade/brightness, and the labeled proposal boxes.
- If no proposal matches, return {"object_id": "", "confidence": 0.0, "rationale": "..."}.
""".strip()


def encode_jpeg_base64(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


class ObjectSelector:
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
        return ""

    def _extract_chat_content_text(self, content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(item.get("text", ""))
            if chunks:
                return "\n".join(chunks)
        raise RuntimeError("Unsupported chat completion content format")

    def _use_responses_first(self):
        return self.api_base == "https://api.openai.com/v1"

    def select(self, query, proposals, annotated_frame):
        image_b64 = encode_jpeg_base64(annotated_frame)
        proposal_payload = [
            {
                "id": item["id"],
                "bbox": item["bbox"],
                "grasp_pixel": item["grasp_pixel"],
                "mean_bgr": item.get("mean_bgr"),
                "yaw_deg": item.get("yaw_deg"),
                "yaw_reliable": item.get("yaw_reliable"),
            }
            for item in proposals
        ]
        user_text = (
            "User query: %s\n\n"
            "Object proposals:\n%s\n\n"
            "Return only the JSON object."
        ) % (query, compact_for_prompt(proposal_payload, 5000))

        responses_payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
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
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % image_b64}},
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }
        attempts = [
            ("Responses API", "/responses", responses_payload),
            ("Chat Completions API", "/chat/completions", chat_payload),
        ]
        if not self._use_responses_first():
            attempts.reverse()

        errors = []
        for label, endpoint, payload in attempts:
            try:
                self._logwarn("Trying object selector via %s" % label)
                response = self._post_json(endpoint, payload)
                if endpoint == "/responses":
                    text = self._extract_responses_output_text(response)
                else:
                    text = self._extract_chat_content_text(response["choices"][0]["message"]["content"])
                result = safe_json_loads(text)
                result.setdefault("object_id", "")
                result.setdefault("confidence", 0.0)
                result.setdefault("rationale", "")
                return result
            except urllib.error.HTTPError as exc:
                message = "%s failed with HTTP %s: %s" % (label, exc.code, self._response_body(exc))
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                message = "%s timed out or could not connect: %s" % (label, exc)
            except Exception as exc:
                message = "%s failed: %s" % (label, exc)
            errors.append(message)
            self._logwarn(message)
        raise RuntimeError(" ; ".join(errors))
