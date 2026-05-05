#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import mimetypes
import os
import socket
import sys
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://api.aibold.art/v1"
from PIL import Image, ImageDraw


def normalize_base(base_url):
    return base_url.rstrip("/")


def candidate_bases(base_url):
    base_url = normalize_base(base_url)
    bases = [base_url]
    if not base_url.endswith("/v1"):
        bases.append(base_url + "/v1")
    return bases


def build_chat_payload(model, prompt):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }


def build_chat_vision_payload(model, prompt, image_data_url):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0,
    }


def build_responses_payload(model, prompt):
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "text": {"format": {"type": "text"}},
    }


def build_responses_vision_payload(model, prompt, image_data_url):
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        "text": {"format": {"type": "text"}},
    }


def build_default_test_image_data_url():
    image = Image.new("RGB", (128, 128), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 28, 108, 108), fill=(220, 40, 40), outline=(20, 20, 20), width=3)
    draw.text((34, 8), "red box", fill=(0, 0, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64,%s" % image_b64


def load_image_data_url(image_path):
    if image_path:
        with open(image_path, "rb") as handle:
            image_bytes = handle.read()
        mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return "data:%s;base64,%s" % (mime_type, image_b64)

    return build_default_test_image_data_url()


def post_json(url, api_key, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, body


def shorten_data_urls(value, keep_chars=80):
    if isinstance(value, dict):
        return {key: shorten_data_urls(val, keep_chars) for key, val in value.items()}
    if isinstance(value, list):
        return [shorten_data_urls(item, keep_chars) for item in value]
    if isinstance(value, str) and value.startswith("data:") and len(value) > keep_chars:
        return value[:keep_chars] + "...<trimmed %d chars>" % (len(value) - keep_chars)
    return value


def extract_text_from_content_parts(parts):
    if not isinstance(parts, list):
        return None

    chunks = []
    for item in parts:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            chunks.append(item["text"])
        elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
            chunks.append(item["text"])
        elif isinstance(item.get("text"), dict) and isinstance(item["text"].get("value"), str):
            chunks.append(item["text"]["value"])

    if chunks:
        return "\n".join(chunks)
    return None


def extract_output_text(parsed):
    candidates = []

    output_text = parsed.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        candidates.append(("output_text", output_text.strip()))

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            candidates.append(("choices[0].message.content", content.strip()))
        content_text = extract_text_from_content_parts(content)
        if content_text:
            candidates.append(("choices[0].message.content[*].text", content_text))

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            candidates.append(("choices[0].message.reasoning_content", reasoning.strip()))
        reasoning_text = extract_text_from_content_parts(reasoning)
        if reasoning_text:
            candidates.append(("choices[0].message.reasoning_content[*].text", reasoning_text))

    output = parsed.get("output")
    if isinstance(output, list):
        for output_index, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            content_text = extract_text_from_content_parts(content)
            if content_text:
                candidates.append(("output[%d].content[*].text" % output_index, content_text))

    for path, text in candidates:
        if text:
            return path, text
    return None, None


def try_request(label, url, api_key, payload, timeout):
    print("=" * 80)
    print("Test:", label)
    print("URL :", url)
    print("Body:", json.dumps(shorten_data_urls(payload), ensure_ascii=False, indent=2))
    try:
        status, body = post_json(url, api_key, payload, timeout)
        print("HTTP:", status)
        print("Resp:", body)
        try:
            parsed = json.loads(body)
            path, text = extract_output_text(parsed)
            if path and text:
                print("Text path:", path)
                print("Text out :", text)
            else:
                print("Text path: <none>")
                print("Text out : <none extracted>")
        except Exception as exc:
            print("Text parse error:", repr(exc))
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print("HTTPError:", exc.code)
        print("Resp     :", body)
        return False
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        print("NetworkError:", exc)
        return False
    except Exception as exc:
        print("Error:", repr(exc))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test an OpenAI-compatible API endpoint without ROS."
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Base URL, default: %s" % DEFAULT_API_BASE,
    )
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: API test success.",
        help="Prompt used for test requests",
    )
    parser.add_argument(
        "--vision-prompt",
        default="Describe this image in one short sentence.",
        help="Prompt used for image-input requests",
    )
    parser.add_argument(
        "--image",
        help="Optional local image path. If omitted, a built-in test image is used.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--mode",
        choices=["chat", "responses", "both", "chat-vision", "responses-vision", "vision"],
        default="both",
        help="Which API style to test",
    )
    args = parser.parse_args()

    image_data_url = load_image_data_url(args.image)
    if args.image:
        print("Using image:", os.path.abspath(args.image))
    else:
        print("Using built-in JPEG test image")
    print("Image MIME:", image_data_url.split(';', 1)[0].replace('data:', ''))

    any_success = False
    for base in candidate_bases(args.api_base):
        if args.mode in ("chat", "both"):
            any_success = try_request(
                "chat/completions",
                base + "/chat/completions",
                args.api_key,
                build_chat_payload(args.model, args.prompt),
                args.timeout,
            ) or any_success

        if args.mode in ("responses", "both"):
            any_success = try_request(
                "responses",
                base + "/responses",
                args.api_key,
                build_responses_payload(args.model, args.prompt),
                args.timeout,
            ) or any_success

        if args.mode in ("chat-vision", "vision"):
            any_success = try_request(
                "chat/completions with image input",
                base + "/chat/completions",
                args.api_key,
                build_chat_vision_payload(args.model, args.vision_prompt, image_data_url),
                args.timeout,
            ) or any_success

        if args.mode in ("responses-vision", "vision"):
            any_success = try_request(
                "responses with image input",
                base + "/responses",
                args.api_key,
                build_responses_vision_payload(args.model, args.vision_prompt, image_data_url),
                args.timeout,
            ) or any_success

    if any_success:
        print("=" * 80)
        print("At least one request succeeded.")
        return 0

    print("=" * 80)
    print("All requests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
