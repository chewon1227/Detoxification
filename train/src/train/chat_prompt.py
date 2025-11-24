"""Shared utilities for building the converse-style chat prompt."""

from __future__ import annotations

import re
from typing import Dict, Tuple

_PROMPT_PATTERN = re.compile(
    r"원문:\s*(?P<body>.*?)(?:\n메타정보:|\n출력:|$)", re.DOTALL
)


def _format_persona(persona: Dict[str, str]) -> str:
    """Serialize persona key/value pairs into the format used in converse.py."""
    return "\n".join(f"{key}: {value}" for key, value in persona.items())


def default_personas(system_prompt: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return default assistant/user personas derived from the system prompt."""
    assistant = {
        "name": "DetoxAssistant",
        "description": system_prompt.strip(),
    }
    user = {
        "name": "User",
        "description": "An online community participant sharing raw text.",
    }
    return assistant, user


def build_converse_prompt(
    init_persona: Dict[str, str], target_persona: Dict[str, str], context: str
) -> str:
    """Replicate the chat prompt format used inside converse.py."""
    init_description = _format_persona(init_persona)
    target_description = _format_persona(target_persona)
    target_name = target_persona.get("name", "User")
    init_name = init_persona.get("name", "Assistant")

    return (
        # f"You are {init_name}.\n"
        # f"This is a brief description of {init_name}.\n"
        f"아래는 당신의 페르소나에 대한 간략한 설명이다."
        f"[페르소나]\n"
        f"{init_description}\n\n"
        # f"This is a brief description of {target_name}.\n"
        # f"{target_description}\n\n"
        # f"{target_name} said: \n"
        f"아래는 디시인사이드에 게시된 게시글이다. 아래는 너의 의견에 반대한다: "
        f"[게시글] {context.strip()}\n\n"
        # f"In this case, What will you say to {target_name}?\n"
    )


def build_converse_prompt_from_text(user_text: str, system_prompt: str) -> str:
    """Convenience wrapper that builds personas from defaults and adds context."""
    init_persona, target_persona = default_personas(system_prompt)
    context = f"{target_persona['name']}: {user_text.strip()}"
    return build_converse_prompt(init_persona, target_persona, context)


def strip_embedded_prompt(text: str) -> str:
    """Remove the verbose instruction wrapper embedded in dataset entries."""
    if not text:
        return ""
    match = _PROMPT_PATTERN.search(text)
    if match:
        return match.group("body").strip()
    # If the template markers are missing, fall back to trimming whitespace.
    return text.strip()
