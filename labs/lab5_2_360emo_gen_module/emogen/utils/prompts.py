def sanitize_prompt_for_filename(prompt: str) -> str:
    """Sanitize prompt string for use in filenames."""
    sanitized = prompt.replace("<s0><s1>", "").replace("<s0>", "").replace("<s1>", "")
    sanitized = sanitized.replace(" ", "_")
    for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        sanitized = sanitized.replace(char, "")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized


# Linux single-component filename limit is typically 255 bytes; reserve room for suffixes like _a*_v*_seed*.pt
FILENAME_PROMPT_MAX_CHARS = 120


def prompt_stub_for_filename(sanitized: str, max_chars: int = FILENAME_PROMPT_MAX_CHARS) -> str:
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[:max_chars].rstrip("_")


def prompt_filename_stem(prompt: str, max_chars: int = FILENAME_PROMPT_MAX_CHARS) -> str:
    """Sanitize prompt and truncate for safe use as part of a filename (matches get_residual.py output)."""
    return prompt_stub_for_filename(sanitize_prompt_for_filename(prompt), max_chars)
