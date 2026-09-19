"""Document cleaning and boilerplate removal for Vietnamese university policy documents."""

from __future__ import annotations

import re

# Standalone boilerplate patterns to eliminate
_CTA_NAVIGATION_PATTERN = re.compile(
    r"^\s*\[(?:"
    r"Toggle|"
    r"Đăng ký(?:\s*ngay)?(?:\s*→)?|"
    r"Khám phá thêm(?:\s*→)?|"
    r"Tìm hiểu thêm(?:\s*→)?|"
    r"Xem thêm(?:\s*→)?|"
    r"xem thêm(?:\s*→)?|"
    r"Xem chi tiết(?:\s*→)?|"
    r"Tại đây|"
    r"Trở về đầu trang|"
    r"Quay lại|"
    r"Chia sẻ(?: bài viết)?"
    r")\](?:\([^)]*\))?\s*$",
    re.IGNORECASE,
)

_TOC_HEADER_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Mục\s+lục(?:\*\*)?\s*$",
    re.IGNORECASE,
)

_FILTER_PATTERN = re.compile(
    r"^\s*Lọc:\s*$",
    re.IGNORECASE,
)

_BREADCRUMB_PATTERN = re.compile(
    r"^\s*Trang\s+chủ\s*[>/»]\s*.*$",
    re.IGNORECASE,
)

_CLOUDFLARE_EMAIL_PATTERN = re.compile(
    r"\[\[email\s+protected\]\]\(/cdn-cgi/l/email-protection#[^)]*\)",
    re.IGNORECASE,
)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def remove_boilerplate(text: str) -> str:
    """Remove common web scraping artifacts, navigation links, and TOC boilerplate.

    Preserves tables, dates, monetary amounts, GPA, and policy numbering structures.

    Args:
        text: Raw document body text.

    Returns:
        Cleaned text with boilerplate stripped.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize Cloudflare obfuscated email links to readable email marker
    cleaned = _CLOUDFLARE_EMAIL_PATTERN.sub("[email protected]", text)

    # Inline toggle link removal if embedded in line
    cleaned = re.sub(r"\[Toggle\]\(#\)", "", cleaned, flags=re.IGNORECASE)

    output_lines: list[str] = []
    lines = cleaned.splitlines()

    for line in lines:
        stripped = line.strip()

        # Remove standalone table of contents markers (e.g. "Mục lục", "### Mục lục")
        if _TOC_HEADER_PATTERN.match(stripped):
            continue

        # Remove standalone web CTA buttons (e.g., "[Đăng ký ngay →](...)")
        if _CTA_NAVIGATION_PATTERN.match(stripped):
            continue

        # Remove web filter UI artifacts (e.g., "Lọc:")
        if _FILTER_PATTERN.match(stripped):
            continue

        # Remove breadcrumb navigation
        if _BREADCRUMB_PATTERN.match(stripped):
            continue

        output_lines.append(line)

    return "\n".join(output_lines)


def normalize_headings(text: str) -> str:
    """Normalize Markdown heading hierarchy and strip bold wrappers inside headings.

    Cleans:
        - '#### **CHƯƠNG I: NHỮNG VẤN ĐỀ CHUNG**' -> '#### CHƯƠNG I: NHỮNG VẤN ĐỀ CHUNG'
        - '# **Giá trị cốt lõi**' -> '# Giá trị cốt lõi'
        - Consecutive identical H1/H2 headings caused by web page titles.

    Args:
        text: Document text.

    Returns:
        Text with standardized heading formatting.
    """
    if not text or not isinstance(text, str):
        return ""

    lines = text.splitlines()
    normalized_lines: list[str] = []
    last_heading_clean: str | None = None
    last_heading_level: int = 0

    for line in lines:
        stripped = line.strip()

        # Check for Markdown heading
        match = _HEADING_PATTERN.match(stripped)
        if match:
            hashes, title = match.groups()
            level = len(hashes)

            # Unwrap bold or italic markers inside heading title
            title_clean = title.strip()
            if title_clean.startswith("**") and title_clean.endswith("**") and len(title_clean) > 4:
                title_clean = title_clean[2:-2].strip()
            elif title_clean.startswith("*") and title_clean.endswith("*") and len(title_clean) > 2:
                title_clean = title_clean[1:-1].strip()

            # Remove trailing hashes (e.g., '## Heading ##' -> '## Heading')
            title_clean = re.sub(r"\s+#+$", "", title_clean).strip()

            # Deduplicate immediate successive identical headings
            clean_canonical = title_clean.lower().strip()
            if clean_canonical == last_heading_clean and level == last_heading_level:
                continue

            last_heading_clean = clean_canonical
            last_heading_level = level
            normalized_lines.append(f"{hashes} {title_clean}")
        else:
            # If line is non-empty and not just university branding banner, reset last_heading tracking
            if stripped and stripped.lower() not in ("fpt university", "trường đại học fpt"):
                last_heading_clean = None
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def normalize_whitespace(text: str, max_blank_lines: int = 2) -> str:
    """Collapse excessive blank lines to a maximum of max_blank_lines (default 2).

    Also trims trailing whitespace on individual lines and strips outer blank lines.

    Args:
        text: Text to normalize.
        max_blank_lines: Maximum allowed consecutive blank lines between text.

    Returns:
        Whitespace-normalized text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.splitlines()]
    joined = "\n".join(lines)

    # In markdown, N blank lines corresponds to (N + 1) newlines.
    # When max_blank_lines = 2, at most 3 newlines are allowed.
    # Collapse any sequence of (max_blank_lines + 2) or more newlines.
    pattern = r"\n{" + str(max_blank_lines + 2) + r",}"
    replacement = "\n" * (max_blank_lines + 1)
    collapsed = re.sub(pattern, replacement, joined)

    # Trim leading/trailing blank lines
    result = collapsed.strip()
    return result + "\n" if result else ""


def clean_document(text: str) -> str:
    """Clean university policy and web documents end-to-end.

    Performs:
    1. Boilerplate and navigation removal.
    2. Heading hierarchy and styling normalization.
    3. Blank line collapse while fully preserving markdown tables,
       dates, monetary numbers, GPAs, and legal policy formatting.

    Args:
        text: Raw document body string.

    Returns:
        Cleaned, high-quality Markdown text ready for indexing.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Strip boilerplate, CTAs, and TOC artifacts
    no_boilerplate = remove_boilerplate(text)

    # Step 2: Normalize heading hierarchy and bold headers
    standardized_headings = normalize_headings(no_boilerplate)

    # Step 3: Normalize whitespace and collapse excessive blank lines
    cleaned = normalize_whitespace(standardized_headings, max_blank_lines=2)

    return cleaned
