"""Shared utilities for building the converse-style chat prompt."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_PROMPT_PATTERN = re.compile(
    r"원문:\s*(?P<body>.*?)(?:\n메타정보:|\n출력:|$)", re.DOTALL
)

DC_SYSTEM_PROMPT = (
    "너는 '디시인사이드' 갤러리 유저다. "
    "주어진 [텍스트] 내용을 바탕으로 반말(비속어, 음슴체)로 댓글을 남겨라. "
    "두서없는 내용이면 핵심만 추려서 한 문장으로 정리하고, '~임', '~함', 'ㅋㅋ', "
    "'ㄹㅇ', 'ㅂㅅ' 같은 표현을 섞어 거칠게 이야기하라. 없는 내용은 지어내지 않는다."
)


def _format_persona(persona: Dict[str, str]) -> str:
    """Serialize persona key/value pairs with Korean labels."""
    label_map = {
        "name": "이름",
        "age": "나이",
        "gender": "성별",
        "religion": "종교",
        "race": "인종",
        "topic": "주제",
        "stance": "입장",
        "background": "배경",
    }
    lines = []
    for key, value in persona.items():
        if value in (None, ""):
            continue
        label = label_map.get(key, key)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


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
    """Replicate the chat prompt format using Korean instructions."""
    init_description = _format_persona(init_persona)
    target_description = _format_persona(target_persona)
    target_name = target_persona.get("name", "User")
    init_name = init_persona.get("name", "Assistant")

    return (
        f"당신은 {init_name}입니다.\n"
        f"다음은 {init_name}에 대한 간단한 정보입니다.\n"
        f"{init_description}\n\n"
        f"다음은 {target_name}에 대한 간단한 정보입니다.\n"
        f"{target_description}\n\n"
        f"{target_name}의 발화:\n"
        f"{context.strip()}\n\n"
        f"위 상황을 바탕으로 {target_name}에게 예의 바르고 설득력 있게 한국어로 대답하세요.\n"
    )


def build_converse_prompt_from_text(user_text: str, system_prompt: str) -> str:
    """Convenience wrapper that builds personas from defaults and adds context."""
    init_persona, target_persona = default_personas(system_prompt)
    context = f"{target_persona['name']}: {user_text.strip()}"
    return build_converse_prompt(init_persona, target_persona, context)


def _build_dc_messages(context_text: str) -> List[Dict[str, str]]:
    clean_context = context_text.strip()
    user_prompt = (
        "아래 [텍스트]를 읽고 디시 말투로 핵심만 찔러서 말해라.\n"
        "엉뚱한 소리 하지 말고 한 문장으로 정리해.\n\n"
        f"[텍스트]\n{clean_context}\n\n"
        "댓글:"
    )
    return [
        {"role": "system", "content": DC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_dc_comment_prompt(tokenizer, context_text: str) -> str:
    """Build the DC-style comment prompt used by the RAG inference script."""
    messages = _build_dc_messages(context_text)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def strip_embedded_prompt(text: str) -> str:
    """Remove the verbose instruction wrapper embedded in dataset entries."""
    if not text:
        return ""
    match = _PROMPT_PATTERN.search(text)
    if match:
        return match.group("body").strip()
    # If the template markers are missing, fall back to trimming whitespace.
    return text.strip()
