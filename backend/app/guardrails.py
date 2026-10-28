import re
from fastapi import HTTPException

PATTERNS = [r"ignore (all )?(previous|prior) instructions", r"reveal (the )?(system prompt|hidden prompt)",
            r"bypass (all )?safety", r"act as (an )?unrestricted"]

def validate_paper_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) > max_chars: raise HTTPException(413, f"Text exceeds {max_chars} characters.")
    if len(text) < 80: raise HTTPException(422, "Provide at least 80 characters.")
    return text

def detect_prompt_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in PATTERNS)

def safe_external_text(value, limit=4000):
    if not isinstance(value, str): raise TypeError("External tool output must be a string.")
    return value[:limit]
