#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

import cv2
import numpy as np


def estimate_background_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = image.shape[:2]
    border = max(12, min(height, width) // 20)
    samples = np.concatenate(
        [
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def foreground_score(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = estimate_background_lab(image)
    delta = lab - bg.reshape(1, 1, 3)

    # Weighted Lab distance from table/background. This is color-independent:
    # any sufficiently non-table object can become a proposal.
    score = np.sqrt(0.35 * delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)
    score = cv2.GaussianBlur(score, (5, 5), 0)
    score = score - float(score.min())
    peak = float(score.max())
    if peak > 0.0:
        score = score / peak
    return (score * 255.0).astype(np.uint8)


def build_foreground_mask(image):
    score = foreground_score(image)
    threshold, otsu_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu can be too conservative on low-contrast blocks. Keep a soft floor.
    floor = max(22, int(round(threshold * 0.72)))
    _, mask = cv2.threshold(score, floor, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, score, threshold, floor


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


def rotated_rect_from_contour(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    if w < h:
        yaw_deg = angle
    else:
        yaw_deg = angle + 90.0
    while yaw_deg <= -90.0:
        yaw_deg += 180.0
    while yaw_deg > 90.0:
        yaw_deg -= 180.0
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(int).tolist()
    long_side = max(float(w), float(h))
    short_side = max(1.0, min(float(w), float(h)))
    rectangularity = abs(long_side - short_side) / long_side
    return {
        "center": [float(cx), float(cy)],
        "size": [float(w), float(h)],
        "angle_deg": float(yaw_deg),
        "rectangularity": float(rectangularity),
        "box_points": box,
    }


def mean_bgr(image, mask, bbox):
    x_min, y_min, x_max, y_max = bbox
    roi = image[y_min:y_max, x_min:x_max]
    roi_mask = mask[y_min:y_max, x_min:x_max]
    pixels = roi[roi_mask > 0]
    if len(pixels) == 0:
        pixels = roi.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    return [float(mean[0]), float(mean[1]), float(mean[2])]


def propose_blocks(image, min_area=500.0, max_area_ratio=0.55):
    mask, score, otsu_threshold, used_threshold = build_foreground_mask(image)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    image_area = float(image.shape[0] * image.shape[1])
    proposals = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        bbox_area = float(w * h)
        if bbox_area > max_area_ratio * image_area:
            continue
        fill_ratio = area / bbox_area
        aspect = float(w) / float(h)
        if fill_ratio < 0.22:
            continue
        if aspect < 0.25 or aspect > 4.0:
            continue

        bbox = [int(x), int(y), int(x + w), int(y + h)]
        center = contour_center(contour, bbox)
        rotated_rect = rotated_rect_from_contour(contour)
        touches_border = x <= 1 or y <= 1 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1
        yaw_confidence = rotated_rect["rectangularity"]
        yaw_reliable = (not touches_border) and yaw_confidence >= 0.08
        proposals.append(
            {
                "bbox": bbox,
                "grasp_pixel": center,
                "rotated_rect": rotated_rect,
                "yaw_deg": rotated_rect["angle_deg"],
                "yaw_reliable": bool(yaw_reliable),
                "yaw_confidence": float(yaw_confidence),
                "touches_border": bool(touches_border),
                "area": area,
                "bbox_area": bbox_area,
                "fill_ratio": fill_ratio,
                "aspect": aspect,
                "mean_bgr": mean_bgr(image, mask, bbox),
                "score": area * fill_ratio,
            }
        )

    proposals.sort(key=lambda item: item["bbox"][0])
    for index, proposal in enumerate(proposals, start=1):
        proposal["id"] = "obj_%d" % index

    return {
        "success": bool(proposals),
        "proposals": proposals,
        "mask": mask,
        "score_image": score,
        "otsu_threshold": float(otsu_threshold),
        "used_threshold": int(used_threshold),
    }


def annotate(image, result):
    vis = image.copy()
    for proposal in result["proposals"]:
        x_min, y_min, x_max, y_max = proposal["bbox"]
        gx, gy = proposal["grasp_pixel"]
        box_points = np.array(proposal["rotated_rect"]["box_points"], dtype=np.int32)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.polylines(vis, [box_points], True, (255, 0, 0), 2)
        cv2.circle(vis, (gx, gy), 5, (0, 0, 255), -1)
        reliable = "ok" if proposal["yaw_reliable"] else "low"
        label = "%s yaw=%.1f(%s) bbox=%s" % (proposal["id"], proposal["yaw_deg"], reliable, proposal["bbox"])
        y_text = max(24, y_min - 6)
        cv2.putText(vis, label, (x_min, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(vis, label, (x_min, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
    if not result["proposals"]:
        cv2.putText(vis, "No proposals", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def serializable_result(result):
    return {
        "success": result["success"],
        "otsu_threshold": result["otsu_threshold"],
        "used_threshold": result["used_threshold"],
        "proposals": result["proposals"],
    }


def main():
    parser = argparse.ArgumentParser(description="Color-independent block proposal test.")
    parser.add_argument("--images", required=True, help="Image path or glob pattern.")
    parser.add_argument("--out-dir", default="test/output_proposals")
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
        result = propose_blocks(image, min_area=args.min_area)
        payload = serializable_result(result)
        payload["image"] = path
        summary.append(payload)

        stem = os.path.splitext(os.path.basename(path))[0]
        run_name = os.path.basename(os.path.dirname(path))
        prefix = "%s_%s" % (run_name, stem)
        cv2.imwrite(os.path.join(args.out_dir, prefix + "_proposals.jpg"), annotate(image, result))
        cv2.imwrite(os.path.join(args.out_dir, prefix + "_mask.jpg"), result["mask"])
        cv2.imwrite(os.path.join(args.out_dir, prefix + "_score.jpg"), result["score_image"])

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Wrote %s" % summary_path)


if __name__ == "__main__":
    main()
