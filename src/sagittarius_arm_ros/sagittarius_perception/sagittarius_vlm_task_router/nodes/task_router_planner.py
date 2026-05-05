#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task_router_common import TaskIntent


SYSTEM_PROMPT = """
You are a task router for a robot arm.
Read the user's instruction and classify it into one supported task.

Return exactly one JSON object, with no markdown and no extra text.
Schema:
{
  "task_type": "clean_desk" | "wave_hand" | "nod" | "draw_circle" | "spin_wrist" | "unknown",
  "parameters": {
    "repetitions": 1,
    "radius_m": 0.04,
    "points": 24,
    "direction": "counterclockwise"
  },
  "rationale": "short reason"
}

Rules:
- Use clean_desk for prompts about cleaning, picking all objects, or tidying the table.
- Use wave_hand for waving or greeting motions.
- Use nod for nodding or bowing motions.
- Use draw_circle for prompts about making a circular arm motion in the air.
- Use spin_wrist for prompts about rotating, twisting, or spinning the wrist.
- If the user does not explicitly ask for multiple repetitions, set repetitions to 1.
- If the task is unsupported, return unknown.
- Keep parameters minimal and only include relevant fields.
""".strip()


class PromptTaskRouter:
    def __init__(self, api_base, api_key, model, timeout, use_llm_routing=True, rospy_module=None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.use_llm_routing = use_llm_routing
        self.rospy = rospy_module

    def _logwarn(self, message):
        if self.rospy is not None:
            self.rospy.logwarn(message)

    def _response_body(self, exc):
        try:
            return exc.read().decode("utf-8")
        except Exception:
            return ""

    def _extract_chat_content_text(self, content):
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        chunks.append(item["text"])
                    elif isinstance(item.get("text"), dict) and isinstance(item["text"].get("value"), str):
                        chunks.append(item["text"]["value"])
            if chunks:
                return "\n".join(chunks).strip()
        raise RuntimeError("Unsupported chat completion content format")

    def _post_chat(self, payload):
        request = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _route_with_llm(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Instruction: %s\nReturn only the JSON object." % prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = self._post_chat(payload)
        content = response["choices"][0]["message"]["content"]
        content = self._extract_chat_content_text(content)
        return TaskIntent.from_dict(json.loads(content))

    def _extract_radius(self, prompt):
        lower = prompt.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|m|meter|meters)", lower)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit in ("cm", "centimeter", "centimeters"):
                return value / 100.0
            return value
        return 0.04

    def _extract_repetitions(self, prompt):
        lower = prompt.lower()
        match = re.search(r"(\d+)\s*(times|repetitions|rounds|圈|次)", lower)
        if match:
            return max(1, int(match.group(1)))
        if "twice" in lower or "两次" in prompt or "两圈" in prompt:
            return 2
        if "once" in lower or "one time" in lower or "一次" in prompt:
            return 1
        return 1

    def _contains_phrase_or_words(self, prompt, phrases=None, word_groups=None):
        lower = prompt.lower()
        phrases = phrases or []
        word_groups = word_groups or []

        if any(phrase in lower for phrase in phrases):
            return True

        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", lower)
        tokens = set(normalized.split())
        for group in word_groups:
            if all(word in tokens for word in group):
                return True
        return False

    def _route_with_rules(self, prompt):
        lower = prompt.lower()

        clean_keywords = [
            "clean the desk",
            "clean desk",
            "clear the desk",
            "tidy the desk",
            "pick all",
            "清扫",
            "清理桌面",
            "收拾桌面",
            "整理桌面",
        ]
        if any(keyword in lower or keyword in prompt for keyword in clean_keywords):
            return TaskIntent(
                task_type="clean_desk",
                parameters={},
                rationale="Matched clean-desk keywords",
            )

        if self._contains_phrase_or_words(
            prompt,
            phrases=["say hello", "挥手", "打招呼", "招手"],
            word_groups=[["wave"], ["greet"], ["hello"]],
        ):
            parameters = {
                "repetitions": self._extract_repetitions(prompt),
            }
            return TaskIntent(
                task_type="wave_hand",
                parameters=parameters,
                rationale="Matched wave-hand keywords",
            )

        if self._contains_phrase_or_words(
            prompt,
            phrases=["点头", "鞠躬"],
            word_groups=[["nod"], ["bow"]],
        ):
            return TaskIntent(
                task_type="nod",
                parameters={"repetitions": self._extract_repetitions(prompt)},
                rationale="Matched nod keywords",
            )

        if self._contains_phrase_or_words(
            prompt,
            phrases=["draw a circle", "circle motion", "move in a circle", "画圈", "圆周", "转圈"],
            word_groups=[["circular"], ["circle"], ["loop"]],
        ):
            return TaskIntent(
                task_type="draw_circle",
                parameters={
                    "radius_m": self._extract_radius(prompt),
                    "revolutions": max(1, self._extract_repetitions(prompt)),
                    "points": 24,
                    "direction": "counterclockwise",
                },
                rationale="Matched circular-motion keywords",
            )

        if self._contains_phrase_or_words(
            prompt,
            phrases=["转动手腕", "旋转手腕"],
            word_groups=[
                ["spin", "wrist"],
                ["rotate", "wrist"],
                ["rotate", "the", "wrist"],
                ["twist", "wrist"],
                ["rotate", "hand"],
            ],
        ):
            return TaskIntent(
                task_type="spin_wrist",
                parameters={"repetitions": self._extract_repetitions(prompt)},
                rationale="Matched spin-wrist keywords",
            )

        return TaskIntent(task_type="unknown", parameters={}, rationale="No supported task matched")

    def route(self, prompt):
        if self.use_llm_routing and self.api_key:
            try:
                intent = self._route_with_llm(prompt)
                if intent.task_type:
                    return intent
            except urllib.error.HTTPError as exc:
                self._logwarn("LLM task routing failed with HTTP %s: %s" % (exc.code, self._response_body(exc)))
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                self._logwarn("LLM task routing network failure: %s" % exc)
            except Exception as exc:
                self._logwarn("LLM task routing failed: %s" % exc)

        return self._route_with_rules(prompt)
