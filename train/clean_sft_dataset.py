"""Validate and clean the SFT dataset into (original_text, detox_text) pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple


PROMPT_PREFIX = "다음 커뮤니티 텍스트를 중립적이고 예의 있는 표현으로 순화해 주세요.\n"
ORIGIN_MARKER = "원문:\n"


def extract_original(user_content: str) -> str:
    """
    Pull the raw user text out of the formatted prompt.

    We keep everything after '원문:\\n' until we hit a known footer such as
    '메타정보:' or '출력:'.
    """
    if ORIGIN_MARKER not in user_content:
        return user_content.strip()

    _, tail = user_content.split(ORIGIN_MARKER, 1)
    for stop in ("\n메타정보:", "\n출력:"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
            break
    return tail.strip()


def parse_item(line: str) -> Tuple[str, str]:
    """Return (original_text, detox_text) or raise ValueError."""
    obj = json.loads(line)

    messages = obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages missing")

    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if user_msg is None or asst_msg is None:
        raise ValueError("user/assistant message missing")

    user_content = user_msg.get("content")
    asst_content = asst_msg.get("content")
    if not isinstance(user_content, str) or not isinstance(asst_content, str):
        raise ValueError("user/assistant content missing")

    original_text = extract_original(user_content)
    detox_text = asst_content.strip()
    if not original_text:
        raise ValueError("empty original_text")
    if not detox_text:
        raise ValueError("empty detox_text")

    return original_text, detox_text


def clean_file(input_path: Path, output_path: Path) -> None:
    ok = 0
    bad = 0
    with input_path.open() as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                original, detox = parse_item(line)
                fout.write(
                    json.dumps(
                        {"original_text": original, "detox_text": detox},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ok += 1
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"[WARN] Line {line_no}: {e}", file=sys.stderr)
    print(f"Processed {ok} items; skipped {bad}. Output -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check sft_dataset and strip prompt/urls into original_text & detox_text fields."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sft_dataset.json"),
        help="Path to the source dataset (jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sft_dataset_clean.jsonl"),
        help="Destination path for cleaned jsonl",
    )
    args = parser.parse_args()
    clean_file(args.input, args.output)


if __name__ == "__main__":
    main()
