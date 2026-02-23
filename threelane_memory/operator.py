"""GSW-style Semantic Operator – extracts structured event semantics from text."""

import json
from threelane_memory.llm_interface import invoke_llm
from threelane_memory.schemas import SemanticExtraction


OPERATOR_PROMPT = """\
You are a semantic operator extracting structured event information from a user's message.
The user is telling you about THEMSELVES unless they explicitly mention someone else.

Return STRICT JSON with the following keys:

{
  "summary": string,
  "emotion": string,
  "importance": float (0-1),
  "entities": [string],
  "roles": [{"entity": string, "role": string}],
  "actions": [{"actor": string, "verb": string, "object": string|null}],
  "states": [{"entity": string, "attribute": string, "value": string}],
  "location": string|null,
  "time": ISO timestamp string|null
}

CRITICAL RULES:
- summary MUST contain ALL concrete facts, names, and numbers from the text.
  BAD:  "A person states their age."
  GOOD: "The speaker is 20 years old."
  BAD:  "Someone has a pet."
  GOOD: "Max is the speaker's dog and is 3 years old."
- entities must be specific named things, not generic words like "person" or "speaker".
  If the user says "my dog Max", entities = ["Max"].
  If the user says "my age is 20", entities = ["speaker"].
- states MUST capture every attribute-value pair mentioned.
  "My age is 20" → states: [{"entity": "speaker", "attribute": "age", "value": "20 years old"}]
  "Max is 3 years old" → states: [{"entity": "Max", "attribute": "age", "value": "3 years old"}]
- Only extract explicit information. Do not hallucinate missing fields.
- importance: 0.1 = casual, 0.5 = notable, 0.9 = major life event.
- emotion must be simple: neutral, happy, sad, stressed, excited, etc.
- Return ONLY valid JSON, no markdown fences, no commentary.
"""


def operator_extract(text: str) -> SemanticExtraction:
    """Run the semantic operator on *text* and return structured extraction."""
    prompt = OPERATOR_PROMPT + f"\n\nText:\n{text}"
    response = invoke_llm(prompt)

    # Strip markdown fences if the LLM wraps them
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0]

    try:
        data: SemanticExtraction = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Operator output is not valid JSON: {exc}\n---\n{response}")

    # ── Normalize: guard against local LLMs returning null for list fields ──
    for key in ("entities", "roles", "actions", "states"):
        if not data.get(key):
            data[key] = []
    data.setdefault("summary", text)
    data.setdefault("emotion", "neutral")
    data.setdefault("importance", 0.5)
    data.setdefault("location", None)

    return data
