"""Generate SFT-ready datasets that neutralize toxic community text using OpenAI or HyperClova."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests
from openai import AuthenticationError, OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = """
역할: 커뮤니티 글/댓글을 폭력적이거나 혐오적이지 않은 중립적이고 존중하는 말투로 고쳐준다.

지침:
- 핵심 사실은 유지하되 욕설/비하/성적 모욕은 삭제하거나 완곡어로 치환한다.
- 반말·명령조를 존대/제안 형태로 완화한다.
- 지나치게 짧은 답변, 음슴체, 혐오 밈은 자연스러운 서술문으로 정리한다.
- 새로운 정보를 추가하거나 삭제하지 않는다.
- 원문이 비어 있으면 "내용 없음"을 반환한다.
출력은 고친 문장만 작성한다 (설명/메모 금지).
""".strip()


@dataclass
class Sample:
    text: str
    kind: str  # "post" or "comment"
    source_url: Optional[str]
    gallery: Optional[str]
    date: Optional[str]
    index: int


@dataclass
class HyperClovaConfig:
    endpoint: str
    api_key: str
    api_key_id: str
    top_p: float
    top_k: int
    repeat_penalty: float


def load_dataset(path: Path, limit: Optional[int]) -> List[Sample]:
    raw = json.loads(path.read_text())
    samples: List[Sample] = []
    for row_idx, row in enumerate(raw):
        main = row.get("main")
        comments = row.get("comments") or []
        sample_meta = {
            "source_url": row.get("source_url"),
            "gallery": row.get("gallery"),
            "date": row.get("date"),
        }
        if isinstance(main, str) and main.strip():
            samples.append(
                Sample(
                    text=main.strip(),
                    kind="post",
                    index=row_idx,
                    **sample_meta,
                )
            )
        for comment_idx, comment in enumerate(comments):
            if not isinstance(comment, str) or not comment.strip():
                continue
            samples.append(
                Sample(
                    text=comment.strip(),
                    kind="comment",
                    index=comment_idx,
                    **sample_meta,
                )
            )
        if limit and len(samples) >= limit:
            return samples[:limit]
    return samples


def build_user_prompt(sample: Sample) -> str:
    header = "입력 유형: 게시글 본문" if sample.kind == "post" else "입력 유형: 댓글"
    context_bits = []
    if sample.gallery:
        context_bits.append(f"갤러리: {sample.gallery}")
    if sample.date:
        context_bits.append(f"작성 시각: {sample.date}")
    if sample.source_url:
        context_bits.append(f"출처: {sample.source_url}")
    context = "\n".join(context_bits)

    return "\n".join(
        [
            "다음 커뮤니티 텍스트를 중립적이고 예의 있는 표현으로 순화해 주세요.",
            header,
            f"원문:\n{sample.text}",
            ("메타정보:\n" + context) if context else "",
            "출력: 고쳐 쓴 문장만 작성",
        ]
    ).strip()


def call_openai(
    client: OpenAI,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            # Some recent models require max_completion_tokens instead of max_tokens.
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except AuthenticationError as e:
        raise SystemExit(
            "OpenAI 인증 실패: OPENAI_API_KEY를 올바르게 설정했는지 확인하세요."
        ) from e


def _extract_hyperclova_content(data: dict) -> str:
    if isinstance(data, dict):
        if data.get("choices"):
            msg = data["choices"][0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
        if data.get("result"):
            msg = data["result"].get("message") or data["result"].get("output") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return content
        if data.get("message") and isinstance(data["message"], dict):
            content = data["message"].get("content")
            if isinstance(content, str):
                return content
    raise SystemExit(f"HyperClova 응답 파싱 실패: {str(data)[:300]}")


def call_hyperclova(
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    config: HyperClovaConfig,
) -> str:
    if not (config.endpoint and config.api_key and config.api_key_id):
        raise SystemExit(
            "HyperClova 설정이 누락되었습니다: --hyperclova-endpoint, --hyperclova-api-key, --hyperclova-api-key-id 혹은 동일한 환경변수를 설정하세요."
        )
    headers = {
        "X-NCP-CLOVASTUDIO-API-KEY": config.api_key,
        "X-NCP-CLOVASTUDIO-API-KEY-ID": config.api_key_id,
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "maxTokens": max_tokens,
        "topP": config.top_p,
        "topK": config.top_k,
        "repeatPenalty": config.repeat_penalty,
        "stopBefore": [],
        "includeAiFilters": True,
    }
    resp = requests.post(config.endpoint, headers=headers, json=payload, timeout=30)
    if resp.status_code == 401:
        raise SystemExit("HyperClova 인증 실패: 키/키ID를 다시 확인하세요.")
    if not resp.ok:
        raise SystemExit(
            f"HyperClova 요청 실패 ({resp.status_code}): {resp.text[:300]}"
        )
    try:
        data = resp.json()
    except ValueError:
        raise SystemExit(f"HyperClova JSON 파싱 실패: {resp.text[:300]}")
    content = _extract_hyperclova_content(data)
    return content.strip()


def _ensure_chat_template(tokenizer: AutoTokenizer, messages: list) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback to simple Tagged prompt.
    parts = []
    for m in messages:
        role = m.get("role", "").upper()
        parts.append(f"[{role}]\n{m.get('content','')}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts)


def call_hyperclova_local(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = _ensure_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip prompt prefix if present
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt) :].strip()
    return decoded


def load_hf_model(
    model_name: str,
    device: str,
    dtype: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def call_model(
    provider: str,
    client: Optional[OpenAI],
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    hyperclova_config: Optional[HyperClovaConfig],
    hf_bundle: Optional[Tuple] = None,
) -> str:
    if provider == "hyperclova":
        if hyperclova_config is None:
            raise SystemExit("HyperClova 설정이 없습니다.")
        return call_hyperclova(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            config=hyperclova_config,
        )
    if provider == "hyperclova-local":
        if hf_bundle is None:
            raise SystemExit("HyperClova 로컬 모델이 초기화되지 않았습니다.")
        tokenizer, model_obj, device = hf_bundle
        return call_hyperclova_local(
            tokenizer=tokenizer,
            model=model_obj,
            device=device,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if client is None:
        raise SystemExit("OpenAI 클라이언트가 초기화되지 않았습니다.")
    return call_openai(
        client=client,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def save_jsonl(output_path: Path, rows: Iterable[dict]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_samples(
    dataset_path: Path,
    output_path: Path,
    limit: Optional[int],
    model: str,
    temperature: float,
    max_tokens: int,
    sleep: float,
    dry_run: bool,
    include_system: bool,
    provider: str,
    hyperclova_config: Optional[HyperClovaConfig],
    hf_bundle: Optional[Tuple] = None,
) -> None:
    samples = load_dataset(dataset_path, limit)
    client = OpenAI() if provider == "openai" else None
    rows = []
    for sample in tqdm(samples, desc="Labeling"):
        user_prompt = build_user_prompt(sample)
        if dry_run:
            rewritten = sample.text  # Placeholder to allow offline testing.
        else:
            rewritten = call_model(
                provider=provider,
                client=client,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                hyperclova_config=hyperclova_config,
                hf_bundle=hf_bundle,
            )
            if sleep:
                time.sleep(sleep)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": rewritten},
        ]

        if not include_system:
            messages = [m for m in messages if m["role"] != "system"]

        rows.append(
            {
                "messages": messages,
                "metadata": {
                    "kind": sample.kind,
                    "source_url": sample.source_url,
                    "gallery": sample.gallery,
                    "date": sample.date,
                    "source_index": sample.index,
                },
            }
        )
    save_jsonl(output_path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT dataset for polite/neutral rewrites of community text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("merged_dataset.json"),
        help="Path to merged_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sft_dataset.jsonl"),
        help="Destination jsonl path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum samples to label")
    parser.add_argument(
        "--provider",
        choices=["openai", "hyperclova", "hyperclova-local"],
        default="openai",
        help="Which backend to call for labeling.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for labeling",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (set if you hit rate limits)",
    )
    parser.add_argument(
        "--hf-model",
        default="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        help="Local HF model id/path for hyperclova-local provider.",
    )
    parser.add_argument(
        "--hf-device",
        default="cuda",
        help="Device for local HF model (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--hf-dtype",
        default=None,
        help="torch dtype string for local HF model (e.g., bfloat16, float16).",
    )
    parser.add_argument(
        "--hyperclova-endpoint",
        default=os.environ.get("HYPERCLOVA_ENDPOINT"),
        help="HyperClova chat-completion endpoint (e.g., https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003)",
    )
    parser.add_argument(
        "--hyperclova-api-key",
        default=os.environ.get("HYPERCLOVA_API_KEY"),
        help="HyperClova API key (X-NCP-CLOVASTUDIO-API-KEY)",
    )
    parser.add_argument(
        "--hyperclova-api-key-id",
        default=os.environ.get("HYPERCLOVA_API_KEY_ID"),
        help="HyperClova API key ID (X-NCP-CLOVASTUDIO-API-KEY-ID)",
    )
    parser.add_argument(
        "--hyperclova-top-p",
        type=float,
        default=0.8,
        help="topP for HyperClova (default 0.8)",
    )
    parser.add_argument(
        "--hyperclova-top-k",
        type=int,
        default=0,
        help="topK for HyperClova (default 0)",
    )
    parser.add_argument(
        "--hyperclova-repeat-penalty",
        type=float,
        default=1.05,
        help="repeatPenalty for HyperClova (default 1.05)",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Store the system message in each sample (default: dropped in output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and echo the input text (structural test only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hyperclova_config = None
    hf_bundle = None
    if args.provider == "hyperclova":
        hyperclova_config = HyperClovaConfig(
            endpoint=args.hyperclova_endpoint,
            api_key=args.hyperclova_api_key,
            api_key_id=args.hyperclova_api_key_id,
            top_p=args.hyperclova_top_p,
            top_k=args.hyperclova_top_k,
            repeat_penalty=args.hyperclova_repeat_penalty,
        )
    if args.provider == "hyperclova-local":
        hf_bundle = load_hf_model(
            model_name=args.hf_model,
            device=args.hf_device,
            dtype=args.hf_dtype,
        )
    generate_samples(
        dataset_path=args.input,
        output_path=args.output,
        limit=args.limit,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep=args.sleep,
        dry_run=args.dry_run,
        include_system=args.include_system,
        provider=args.provider,
        hyperclova_config=hyperclova_config,
        hf_bundle=hf_bundle,
    )


if __name__ == "__main__":
    main()
