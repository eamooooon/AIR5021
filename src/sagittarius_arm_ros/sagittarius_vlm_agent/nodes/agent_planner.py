#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import urllib.error
import urllib.request

from agent_common import compact_for_prompt, safe_json_loads


SYSTEM_PROMPT = """
You are a robot agent controller.
You cannot directly control the robot.
You can only call one tool at a time from the provided tool list.
Return exactly one JSON object, with no markdown and no extra text.
Do not output Python code.
Do not invent tools.
Use memory object IDs when referring to detected objects.
If required visual information is missing, call capture_image or detect_object first.
Call finish_task when the user request has been completed or cannot be completed safely.

Output schema:
{
  "tool": "detect_object",
  "arguments": {"query": "green block"},
  "rationale": "short reason"
}
""".strip()


class AgentPlanner:
    def __init__(self, api_base, api_key, model, timeout, rospy_module=None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.rospy = rospy_module

    def _logwarn(self, message):
        if self.rospy is not None:
            self.rospy.logwarn(message)

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

    def next_tool_call(self, prompt, memory_snapshot, tool_schemas):
        user_text = (
            "User request:\n%s\n\n"
            "Available tools:\n%s\n\n"
            "Current memory:\n%s\n\n"
            "Return exactly one JSON tool call."
        ) % (
            prompt,
            compact_for_prompt(tool_schemas, 5000),
            compact_for_prompt(memory_snapshot, 7000),
        )
        responses_payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            "text": {"format": {"type": "json_object"}},
        }
        chat_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
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
                self._logwarn("Trying agent planner via %s" % label)
                response = self._post_json(endpoint, payload)
                if endpoint == "/responses":
                    text = self._extract_responses_output_text(response)
                else:
                    text = self._extract_chat_content_text(response["choices"][0]["message"]["content"])
                call = safe_json_loads(text)
                if "tool" not in call:
                    raise RuntimeError("Planner JSON does not contain tool")
                call.setdefault("arguments", {})
                call.setdefault("rationale", "")
                return call
            except urllib.error.HTTPError as exc:
                message = "%s failed with HTTP %s: %s" % (label, exc.code, self._response_body(exc))
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                message = "%s timed out or could not connect: %s" % (label, exc)
            except Exception as exc:
                message = "%s failed: %s" % (label, exc)
            errors.append(message)
            self._logwarn(message)
        raise RuntimeError(" ; ".join(errors))
