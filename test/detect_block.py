#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

import cv2
import numpy as np


def color_score_bgr(image, color):
    b, g, r = cv2.split(image.astype(np.float32))
    color = color.lower()
    if color == "green":
        score = 2.0 * g - r - b
    elif color == "red":
        score = 2.0 * r - g - b
    elif color == "blue":
        score = 2.0 * b - r - g
    elif color == "yellow":
        score = r + g - 1.4 * b - 0.25 * np.abs(r - g)
    else:
        raise RuntimeError("Unsupported color: %s" % color)
    score = score - float(score.min())
    peak = float(score.max())
    if peak > 0.0:
        score = score / peak
    return (score * 255.0).astype(np.uint8)


def make_mask(image, color):
    score = color_score_bgr(image, color)
    score = cv2.GaussianBlur(score, (5, 5), 0)
    threshold, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, threshold


def contour_center(contour, bbox):
    moments = cv2.moments(contour)
    x_min, y_min, x_max, y_max = bbox
    if moments["m00"] > 0.0:
        return [
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        ]
    return [
        int(round((x_min + x_max) * 0.5)),
        int(round((y_min + y_max) * 0.5)),
    ]


def detect_block(image, color="green", min_area=500.0):
    mask, threshold = make_mask(image, color)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    candidates = []
    image_area = float(image.shape[0] * image.shape[1])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        bbox_area = float(w * h)
        fill_ratio = area / bbox_area
        aspect = float(w) / float(h)
        if fill_ratio < 0.35:
            continue
        if aspect < 0.35 or aspect > 2.8:
            continue
        if bbox_area > 0.75 * image_area:
            continue
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        candidates.append(
            {
                "contour": contour,
                "bbox": bbox,
                "area": area,
                "fill_ratio": fill_ratio,
                "aspect": aspect,
                "score": area * fill_ratio,
            }
        )

    if not candidates:
        return {
            "success": False,
            "error": "No block candidate found",
            "threshold": float(threshold),
            "mask": mask,
        }

    best = max(candidates, key=lambda item: item["score"])
    grasp_pixel = contour_center(best["contour"], best["bbox"])
    return {
        "success": True,
        "bbox": best["bbox"],
        "grasp_pixel": grasp_pixel,
        "area": best["area"],
        "fill_ratio": best["fill_ratio"],
        "aspect": best["aspect"],
        "threshold": float(threshold),
        "candidate_count": len(candidates),
        "mask": mask,
    }


def annotate(image, result, color_name):
    vis = image.copy()
    if result["success"]:
        x_min, y_min, x_max, y_max = result["bbox"]
        gx, gy = result["grasp_pixel"]
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.circle(vis, (gx, gy), 5, (0, 0, 255), -1)
        label = "%s bbox=%s grasp=%s" % (color_name, result["bbox"], result["grasp_pixel"])
        cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1)
    else:
        cv2.putText(vis, result["error"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def serializable_result(result):
    return {key: value for key, value in result.items() if key != "mask" and key != "contour"}


def main():
    parser = argparse.ArgumentParser(description="Offline colored block detector for saved RGB images.")
    parser.add_argument("--images", required=True, help="Image path or glob pattern.")
    parser.add_argument("--color", default="green", choices=["green", "red", "blue", "yellow"])
    parser.add_argument("--out-dir", default="test/output")
    parser.add_argument("--min-area", type=float, default=500.0)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths and os.path.exists(args.images):
        paths = [args.images]
    if not paths:
        raise SystemExit("No images matched: %s" % args.images)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            summary.append({"image": path, "success": False, "error": "Failed to read image"})
            continue
        result = detect_block(image, args.color, args.min_area)
        payload = serializable_result(result)
        payload["image"] = path
        summary.append(payload)

        stem = os.path.splitext(os.path.basename(path))[0]
        run_name = os.path.basename(os.path.dirname(path))
        prefix = "%s_%s" % (run_name, stem)
        cv2.imwrite(os.path.join(args.out_dir, prefix + "_annotated.jpg"), annotate(image, result, args.color))
        cv2.imwrite(os.path.join(args.out_dir, prefix + "_mask.jpg"), result["mask"])

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote %s" % summary_path)


if __name__ == "__main__":
    main()
