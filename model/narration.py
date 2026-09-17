"""
AquaScan — LLM Narration Layer
Turns a raw classification into a short, plain-language explanation
using the Groq API (free tier, OpenAI-compatible endpoint).
"""

import os
import requests

FALLBACK_TEMPLATE = (
    "This appears to be {predicted_class} (severity: {severity}). "
    "Please report or dispose of this waste appropriately to prevent waterway contamination."
)


def _resolve_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve Groq API key from argument, environment, or Streamlit secrets."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            sec_key = st.secrets["GROQ_API_KEY"]
            if sec_key and sec_key.strip():
                return sec_key.strip()
    except Exception:
        pass

    return None


def get_explanation(
    predicted_class: str,
    confidence: float,
    severity: str,
    api_key: str | None = None,
) -> tuple[str, bool]:
    """
    Calls Groq to generate a short explanation of why this waste type
    matters and what to do about it.
    
    Returns:
        tuple[str, bool]: (explanation_text, is_ai_generated)
    """
    key = _resolve_api_key(api_key)
    fallback = FALLBACK_TEMPLATE.format(predicted_class=predicted_class, severity=severity)

    if not key:
        return fallback, False

    prompt = f"""
You are an environmental waste classification assistant.
A user photographed waste near a water body. The ML model classified it as:
- Type: {predicted_class} (confidence: {confidence:.0%})
- Severity: {severity}

Respond with EXACTLY:
1. One sentence: what this waste type is
2. One sentence: why it's harmful to water ecosystems specifically
3. One sentence: one concrete action the user or local authority can take

Keep the total response under 80 words. Do not add disclaimers or caveats.
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
            },
            timeout=8,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content, True
    except Exception:
        return fallback, False
