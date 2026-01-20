from typing import Dict

from .llm_client import openai_client
from . import prompts
from .settings import LLM_MODEL


def generate_answer(question: str, context: str) -> str:
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not set.")
    client = openai_client()
    user_prompt = prompts.RAG_RESPOSE_USER_PROMPT.format(context=context, question=question)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompts.RAG_RESPOSE_SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def answer_with_context(payload: Dict[str, object]) -> str:
    question = str(payload.get("question", "")).strip()
    context = str(payload.get("context", "")).strip()
    return generate_answer(question, context)
