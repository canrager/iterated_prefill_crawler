#!/usr/bin/env python3
"""Extract actual chat logs from an OpenRouter exported chat JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object")
    return data


def _extract_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _build_character_index(characters: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(characters, dict):
        return {}

    index: dict[str, dict[str, Any]] = {}
    for character_id, payload in characters.items():
        if not isinstance(payload, dict):
            continue
        model_info = payload.get("modelInfo")
        if not isinstance(model_info, dict):
            model_info = {}
        index[character_id] = {
            "characterId": character_id,
            "model": payload.get("model"),
            "permaslug": model_info.get("permaslug"),
            "supports_reasoning": model_info.get("supports_reasoning"),
        }
    return index


def _build_item_index(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, dict):
        return {}

    index: dict[str, dict[str, Any]] = {}
    for item_id, payload in items.items():
        if isinstance(payload, dict):
            index[item_id] = payload
    return index


def _extract_message_content(message: dict[str, Any], item_index: dict[str, dict[str, Any]]) -> str:
    message_items = message.get("items")
    if not isinstance(message_items, list):
        return ""

    contents: list[str] = []
    for item_ref in message_items:
        if not isinstance(item_ref, dict):
            continue
        if item_ref.get("type") != "message":
            continue
        item_id = item_ref.get("id")
        if not isinstance(item_id, str):
            continue
        item = item_index.get(item_id)
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        text = _extract_text(data.get("content"))
        if text:
            contents.append(text)
    return "".join(contents)


def extract_chat(export: dict[str, Any]) -> dict[str, Any]:
    characters = _build_character_index(export.get("characters"))
    item_index = _build_item_index(export.get("items"))
    raw_messages = export.get("messages")
    if not isinstance(raw_messages, dict):
        raise ValueError("Expected 'messages' to be an object")

    output_messages: list[dict[str, Any]] = []
    for message_id, payload in raw_messages.items():
        if not isinstance(payload, dict):
            continue

        role = payload.get("type")
        if role not in {"user", "assistant", "system"}:
            continue

        message: dict[str, Any] = {
            "messageId": message_id,
            "timestamp": payload.get("createdAt"),
            "role": role,
            "content": _extract_message_content(payload, item_index),
        }

        if role != "user":
            character_id = payload.get("characterId")
            if isinstance(character_id, str):
                message["characterId"] = character_id
                character = characters.get(character_id)
                if character:
                    message["model"] = character.get("model")
                    message["permaslug"] = character.get("permaslug")
                    message["supports_reasoning"] = character.get("supports_reasoning")

        output_messages.append(message)

    output_messages.sort(key=lambda msg: (str(msg.get("timestamp")), str(msg.get("messageId"))))

    models = sorted(
        (
            character
            for character in characters.values()
            if character.get("model") is not None or character.get("permaslug") is not None
        ),
        key=lambda item: str(item.get("characterId")),
    )

    return {
        "title": export.get("title"),
        "models": models,
        "messages": output_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract actual chat messages and model metadata from an OpenRouter chat export."
    )
    parser.add_argument("input_json", type=Path, help="Path to the OpenRouter export JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for output. Default: 2",
    )
    args = parser.parse_args()

    export = _load_json(args.input_json)
    cleaned = extract_chat(export)
    rendered = json.dumps(cleaned, indent=args.indent, ensure_ascii=False) + "\n"

    if args.output is None:
        print(rendered, end="")
        return

    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
