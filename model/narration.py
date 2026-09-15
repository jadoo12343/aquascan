"""
AquaScan — LLM Narration Layer
Turns a raw classification into a short, plain-language explanation
using the Groq API (free tier, OpenAI-compatible endpoint).
"""

import os
import requests

FALLBACK_TEMPLATE = (
    "This appears to be {predicted_class} (severity: {severity}). "
    "Please report or dispose of this waste appropriately."
)

def get_explanation(predicted_class: str, confidence: float, severity: str) -> str:
    """
    Calls Groq to generate a short explanation of why this waste type
    matters and what to do about it. Falls back to a plain templated
    string if the API key is missing or the call fails, so a network
    hiccup never breaks the live demo.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    fallback = FALLBACK_TEMPLATE.format(predicted_class=predicted_class, severity=severity)

    if not api_key:
        return fallback

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
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback
