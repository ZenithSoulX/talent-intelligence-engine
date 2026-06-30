import os
import json
import re
import anthropic

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Export it before running this script."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client

_SYSTEM_PROMPT = """You are a structured data extractor for job descriptions.
Your job is to read a job description and return a single valid JSON object — nothing else.
No preamble. No markdown fences. No explanation. Only the JSON.

Return this exact schema (use null for missing fields, empty list [] for missing arrays):

{
  "role_title": string,
  "company_name": string,
  "employment_type": string,           // "full_time" | "contract" | "part_time" | null
  "location": {
    "cities": [string],
    "remote_ok": bool,
    "relocation_supported": bool,
    "visa_sponsorship": bool
  },
  "experience": {
    "min_years": int | null,
    "max_years": int | null,
    "preferred_years": int | null
  },
  "required_skills": [string],         // hard requirements — absence is a dealbreaker
  "preferred_skills": [string],        // nice-to-haves
  "hard_disqualifiers": [string],      // explicit things that rule a candidate OUT
  "domain_keywords": [string],         // important domain terms (retrieval, ranking, LLMs, etc.)
  "product_company_required": bool,    // true if pure services-company background is disqualifying
  "production_ml_required": bool,      // true if research-only background is disqualifying
  "notice_period": {
    "preferred_days": int | null,
    "max_buyout_days": int | null
  },
  "seniority_level": string,           // "junior" | "mid" | "senior" | "staff" | "principal"
  "culture_signals": [string],         // things like "async-first", "move fast", "disagree openly"
  "ideal_candidate_summary": string    // 2-3 sentence plain-English summary of the ideal hire
}
"""

def parse_jd(jd_text: str) -> dict:
    """
    Takes raw job description text and returns a structured dict.

    This runs once during preprocessing — not at query time — so the
    Anthropic API latency is fine here.

    Args:
        jd_text: The raw job description string (markdown or plain text).

    Returns:
        A dict matching the schema defined in _SYSTEM_PROMPT.
        If parsing fails, returns a minimal fallback dict with an
        'parse_error' key set to the error message.
    """
    client = _get_client()

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": jd_text.strip()
            }
        ]
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

    parsed = json.loads(raw)
    return parsed


def load_jd_text(path: str) -> str:
    """Simple helper to read the JD file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_parsed_jd(parsed: dict, out_path: str) -> None:
    """Saves the structured dict to a JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"Saved parsed JD → {out_path}")


if __name__ == "__main__":
    import sys

    jd_path   = "data/job_description.md"
    out_path  = "artifacts/jd_parsed.json"

    print(f"Reading JD from: {jd_path}")
    jd_text = load_jd_text(jd_path)

    print("Calling Anthropic API to parse JD...")
    parsed = parse_jd(jd_text)

    print("\n--- Parsed JD (preview) ---")
    print(json.dumps(parsed, indent=2))

    save_parsed_jd(parsed, out_path)
