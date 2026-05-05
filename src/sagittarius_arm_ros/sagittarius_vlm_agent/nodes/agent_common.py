#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re


def safe_json_loads(text):
    text = text.strip()
    if not text:
        raise RuntimeError("Empty JSON payload")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def object_key(label):
    key = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    return key or "object"


def compact_for_prompt(value, max_chars=6000):
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... truncated ..."
