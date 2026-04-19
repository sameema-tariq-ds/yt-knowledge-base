import re
import unicodedata
from typing import Optional


INVALID_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_filename(name: str, *, replacement: str = "_", max_length: int = 255, default: str = "file",) -> str:
    """
    Sanitize a string to make it safe for use as a filename.

    Features:
    - Removes unsafe characters
    - Normalizes unicode → ASCII
    - Prevents empty filenames
    - Enforces max length
    - Strips leading/trailing dots and spaces

    Args:
        name: Raw input string
        replacement: Replacement for invalid characters
        max_length: Maximum filename length
        default: Fallback if result is empty

    Returns:
        Safe filename string
    """

    if not isinstance(name, str):
        raise TypeError("Filename must be a string")

    # Normalize unicode → ASCII (e.g., é → e)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # Replace invalid characters
    name = INVALID_CHARS_PATTERN.sub(replacement, name)

    # Collapse multiple replacements
    name = re.sub(rf"{re.escape(replacement)}+", replacement, name)

    # Strip leading/trailing dots, spaces, underscores
    name = name.strip(" ._-")

    # Prevent empty filename
    if not name:
        name = default

    # Enforce max length
    if len(name) > max_length:
        name = name[:max_length].rstrip(" ._-")

    return name