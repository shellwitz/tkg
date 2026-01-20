import os


def openai_client(api_key_env: str = "MODEL_API_KEY", base_url_env: str = "MODEL_BASE_URL"):
    api_key = os.getenv(api_key_env, "")
    base_url = os.getenv(base_url_env, "") or None
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required for LLM/embedding calls.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required. Install it to run this action.") from exc
    return OpenAI(api_key=api_key, base_url=base_url)
