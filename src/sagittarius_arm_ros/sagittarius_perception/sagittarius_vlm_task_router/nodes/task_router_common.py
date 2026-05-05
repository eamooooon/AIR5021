#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field


@dataclass
class TaskIntent:
    task_type: str
    parameters: dict = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_dict(cls, payload):
        return cls(
            task_type=str(payload.get("task_type", "")).strip(),
            parameters=dict(payload.get("parameters", {}) or {}),
            rationale=str(payload.get("rationale", "")).strip(),
        )
