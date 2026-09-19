from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUTPUT_DIR = Path("data/university")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_DOMAIN = "daihoc.fpt.edu.vn"

# Corpus chính thức ban đầu.
# Có thể bổ sung URL sau.
SOURCES = [
    {
        "url": "https://daihoc.fpt.edu.vn/hoat-dong-nha-truong/tin-tuc-chung/quy-che-dao-tao-dai-hoc-chinh-quy/",
        "category": "academic_regulation",
        "audience": "student",
        "department": "academic_affairs",
        "document_version": "2023",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/quy-che-tuyen-sinh-2026/",
        "category": "admission_regulation",
        "audience": "student",
        "department": "admissions",
        "document_version": "2026",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/hoc-phi-tai-campus-tp-ho-chi-minh/",
        "category": "tuition",
        "audience": "student",
        "department": "finance",
        "document_version": "2026",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/hoc-bong/",
        "category": "scholarship",
        "audience": "student",
        "department": "scholarship",
        "document_version": "2026",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/hoc-bong/faq-hoc-bong/",
        "category": "scholarship_faq",
        "audience": "student",
        "department": "scholarship",
        "document_version": "2026",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/hoc-bong/hoc-bong-tinh-hoa-cong-nghe/",
        "category": "scholarship_policy",
        "audience": "student",
        "department": "scholarship",
        "document_version": "2026",
    },
    {
        "url": "https://daihoc.fpt.edu.vn/hoc-bong/hoc-bong-ban-linh-the-he/",
        "category": "scholarship_policy",
        "audience": "student",
        "department": "scholarship",
        "document_version": "2026",
    },
]


def make_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "FPTU-Day07-Student-RAG-Crawler/1.0 "
                "(educational project; public pages only)"
            )
        }
    )

    return session


def validate_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Invalid scheme: {url}")

    if parsed.netloc != ALLOWED_DOMAIN:
        raise ValueError(
            f"Only {ALLOWED_DOMAIN} is allowed. Got: {parsed.netloc}"
        )


def extract_title(soup: BeautifulSoup) -> str:
    # Ưu tiên H1.
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return title

    # Fallback OpenGraph.
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # Fallback HTML title.
    if soup.title:
        return soup.title.get_text(" ", strip=True)

    return "Untitled Document"


def extract_published_date(soup: BeautifulSoup) -> str | None:
    meta_names = [
        ("property", "article:published_time"),
        ("name", "date"),
        ("name", "publish_date"),
    ]

    for attr, value in meta_names:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return tag["content"].strip()

    time_tag = soup.find("time")
    if time_tag:
        if time_tag.get("datetime"):
            return time_tag["datetime"].strip()

        text = time_tag.get_text(" ", strip=True)
        if text:
            return text

    return None


def remove_noise(node: BeautifulSoup) -> None:
    # Xóa các tag chắc chắn không cần.
    for tag in list(
        node.select(
            "script, style, noscript, iframe, svg, "
            "header, footer, nav, aside, form, button"
        )
    ):
        try:
            tag.decompose()
        except Exception:
            pass

    noisy_patterns = [
        "breadcrumb",
        "share",
        "social",
        "cookie",
        "popup",
        "sidebar",
        "related",
        "recommended",
        "menu",
        "navigation",
        "comment",
        "advertisement",
    ]

    # Sau decompose(), một số descendant Tag có attrs=None.
    # Vì vậy phải kiểm tra trước khi gọi .get().
    for element in list(node.find_all(True)):
        try:
            if element.attrs is None:
                continue

            classes = element.get("class") or []
            if isinstance(classes, str):
                classes = [classes]

            classes_text = " ".join(classes)
            element_id = element.get("id") or ""

            combined = f"{classes_text} {element_id}".lower()

            if any(pattern in combined for pattern in noisy_patterns):
                element.decompose()

        except (AttributeError, TypeError):
            continue
def find_main_content(soup: BeautifulSoup):
    """
    FPTU dùng WordPress/Elementor ở nhiều trang.
    Thử selector cụ thể trước rồi mới fallback sang main/body.
    """
    selectors = [
        "article",
        ".elementor-widget-theme-post-content",
        ".entry-content",
        ".post-content",
        ".td-post-content",
        "main",
    ]

    for selector in selectors:
        node = soup.select_one(selector)

        if node:
            text = node.get_text(" ", strip=True)

            # Tránh chọn container rỗng.
            if len(text) >= 300:
                return node

    return soup.body or soup


def clean_markdown(text: str) -> str:
    # Chuẩn hóa newline.
    text = text.replace("\xa0", " ")

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Không để >2 dòng trắng liên tục.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Xóa dòng chỉ chứa Image.
    text = re.sub(
        r"(?im)^\s*image\s*:?.*$",
        "",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")

    if not path:
        return "homepage"

    slug = path.split("/")[-1]

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")

    return slug or "document"


def yaml_string(value) -> str:
    """
    Quote đơn giản để YAML front matter không lỗi.
    """
    if value is None:
        return '""'

    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')

    return f'"{value}"'


def crawl_one(
    session: requests.Session,
    source: dict,
) -> Path:
    url = source["url"]

    validate_url(url)

    print(f"\n[GET] {url}")

    response = session.get(
        url,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    title = extract_title(soup)
    published_at = extract_published_date(soup)

    main_node = find_main_content(soup)

    # Clone lại HTML để thao tác cleanup không ảnh hưởng object gốc.
    content_soup = BeautifulSoup(str(main_node), "lxml")

    remove_noise(content_soup)

    markdown = md(
        str(content_soup),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )

    markdown = clean_markdown(markdown)

    if len(markdown) < 300:
        raise RuntimeError(
            f"Extracted content is suspiciously short: {len(markdown)} chars"
        )

    retrieved_at = datetime.now().astimezone().date().isoformat()

    front_matter = f"""---
title: {yaml_string(title)}
source_url: {yaml_string(url)}
retrieved_at: {yaml_string(retrieved_at)}
document_version: {yaml_string(source["document_version"])}
published_at: {yaml_string(published_at or "")}
audience: {yaml_string(source["audience"])}
department: {yaml_string(source["department"])}
category: {yaml_string(source["category"])}
language: "vi"
source_domain: "{ALLOWED_DOMAIN}"
---

"""

    final_content = front_matter + f"# {title}\n\n" + markdown + "\n"

    filename = slug_from_url(url) + ".md"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(
        final_content,
        encoding="utf-8",
    )

    print(f"[OK] {output_path}")
    print(f"     title = {title}")
    print(f"     chars = {len(markdown):,}")

    return output_path


def main():
    session = make_session()

    successes = []
    failures = []

    for index, source in enumerate(SOURCES, start=1):
        try:
            print(f"\n===== {index}/{len(SOURCES)} =====")

            path = crawl_one(
                session=session,
                source=source,
            )

            successes.append(path)

        except Exception as exc:
            print(f"[ERROR] {source['url']}")
            print(f"        {type(exc).__name__}: {exc}")

            failures.append(
                {
                    "url": source["url"],
                    "error": str(exc),
                }
            )

        # Crawl nhẹ nhàng, không spam server.
        time.sleep(1.5)

    print("\n" + "=" * 60)
    print("CRAWL SUMMARY")
    print("=" * 60)

    print(f"Success: {len(successes)}")
    print(f"Failed : {len(failures)}")

    for path in successes:
        print(f"  ✓ {path}")

    for item in failures:
        print(f"  ✗ {item['url']}")
        print(f"    {item['error']}")


if __name__ == "__main__":
    main()