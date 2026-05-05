#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hand_eye_calibration import transform_to_dict, write_calibration_yaml


def main():
    parser = argparse.ArgumentParser(description="Build an eye-to-hand calibration YAML for Sagittarius VLM agent.")
    parser.add_argument("--input", required=True, help="YAML with camera intrinsics, marker size, and calibration samples")
    parser.add_argument("--output", required=True, help="Output YAML containing cam_H_base")
    args = parser.parse_args()

    result = write_calibration_yaml(args.input, args.output)
    print(json.dumps(transform_to_dict(result["cam_H_base"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
