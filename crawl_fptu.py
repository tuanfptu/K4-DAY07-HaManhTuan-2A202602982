from __future__ import annotations

import argparse
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
ALLOWED_DOMAIN = "daihoc.fpt.edu.vn"

SOURCES = [
    {
        "filename": "01-academic-regulations.md",
        "url": "https://daihoc.fpt.edu.vn/hoat-dong-nha-truong/tin-tuc-chung/quy-che-dao-tao-dai-hoc-chinh-quy/",
        "category": "academic_regulation",
        "audience": "student",
        "campus": "all",
        "department": "academic_affairs",
        "document_version": "public-web-version",
        "coverage": "credits,re-study,retake,preservation,exam,gpa,graduation,ojt",
    },
    {
        "filename": "02-fap-and-academic-procedures.md",
        "url": "https://daihoc.fpt.edu.vn/tin-tuc-chung-2/huong-dan-su-dung-cong-thong-tin-dao-tao-fap-cho-tan-sinh-vien-dai-hoc-fpt/",
        "category": "academic_services",
        "audience": "student",
        "campus": "all",
        "department": "academic_affairs",
        "document_version": "2021-guide",
        "coverage": "fap,schedule,attendance,grades,finance,online-requests",
    },
    {
        "filename": "03-tuition-hcm.md",
        "url": "https://daihoc.fpt.edu.vn/hoc-phi-tai-campus-tp-ho-chi-minh/",
        "category": "tuition",
        "audience": "student",
        "campus": "hcm",
        "department": "finance",
        "document_version": "2026",
        "coverage": "tuition,fees,financial-policy",
    },
    {
        "filename": "04-scholarship-faq.md",
        "url": "https://daihoc.fpt.edu.vn/hoc-bong/faq-hoc-bong/",
        "category": "scholarship",
        "audience": "student",
        "campus": "all",
        "department": "scholarship",
        "document_version": "2026",
        "coverage": "scholarship,eligibility,application,maintenance",
    },
    {
        "filename": "05-student-services-hcm.md",
        "url": "https://daihoc.fpt.edu.vn/hcm/lien-he/",
        "category": "student_services",
        "audience": "student",
        "campus": "hcm",
        "department": "student_services",
        "document_version": "current",
        "coverage": "student-services,administrative-support,contact,location",
    },
    {
        "filename": "06-campus-facilities-hcm.md",
        "url": "https://daihoc.fpt.edu.vn/hcm/",
        "category": "campus_services",
        "audience": "student",
        "campus": "hcm",
        "department": "campus_operations",
        "document_version": "current",
        "coverage": "campus,facilities,classrooms,student-experience",
    },
    {
        "filename": "07-ojt-regulations.md",
        "url": "https://daihoc.fpt.edu.vn/thong-bao-huong-dan/ojt-spring-2026-thong-bao-tham-du-orientation-ojt/",
        "category": "ojt",
        "audience": "student",
        "campus": "hcm",
        "department": "corporate_relations",
        "document_version": "spring-2026",
        "coverage": "ojt,eligibility,credits,orientation,requirements",
    },
    {
        "filename": "08-ojt-registration.md",
        "url": "https://daihoc.fpt.edu.vn/thong-bao-huong-dan/ojt-spring-2026-thong-bao-ve-viec-huong-dan-sinh-vien-dang-ky-doanh-nghiep-ojt/",
        "category": "ojt_registration",
        "audience": "student",
        "campus": "hcm",
        "department": "corporate_relations",
        "document_version": "spring-2026",
        "coverage": "ojt,company-registration,deadline,support",
    },
    {
        "filename": "09-international-exchange-hcm.md",
        "url": "https://daihoc.fpt.edu.vn/hcm/chuong-trinh-trao-doi-exchange/",
        "category": "international_exchange",
        "audience": "student",
        "campus": "hcm",
        "department": "international_collaboration_pdp",
        "document_version": "current",
        "coverage": "exchange,international,pdp,global-experience",
    },
    {
        "filename": "10-departments-and-support-routing-hcm.md",
        "url": "https://daihoc.fpt.edu.vn/hcm/ban-lanh-dao-campus-hcm/",
        "category": "support_routing",
        "audience": "student",
        "campus": "hcm",
        "department": "campus_management",
        "document_version": "current",
        "coverage": "departments,student-affairs,student-services,it,pdp,international",
    },
]

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "FPTU-HCM-Day07-RAG-Crawler/2.0 (educational project; public official pages only)"
    })
    return session

def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Invalid scheme: {url}")
    if parsed.netloc != ALLOWED_DOMAIN:
        raise ValueError(f"Only {ALLOWED_DOMAIN} is allowed. Got: {parsed.netloc}")

def extract_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
        if title:
            return title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    if soup.title:
        return soup.title.get_text(" ", strip=True)
    return "Untitled Document"

def extract_published_date(soup: BeautifulSoup) -> str | None:
    for attr, value in [
        ("property", "article:published_time"),
        ("name", "date"),
        ("name", "publish_date"),
    ]:
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
    for tag in list(node.select(
        "script, style, noscript, iframe, svg, header, footer, nav, aside, form, button"
    )):
        try:
            tag.decompose()
        except Exception:
            pass

    noisy_patterns = [
        "breadcrumb", "share", "social", "cookie", "popup", "sidebar",
        "related", "recommended", "menu", "navigation", "comment", "advertisement",
    ]
    for element in list(node.find_all(True)):
        try:
            if element.attrs is None:
                continue
            classes = element.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            combined = f"{' '.join(classes)} {element.get('id') or ''}".lower()
            if any(pattern in combined for pattern in noisy_patterns):
                element.decompose()
        except (AttributeError, TypeError):
            continue

def find_main_content(soup: BeautifulSoup):
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
        if node and len(node.get_text(" ", strip=True)) >= 250:
            return node
    return soup.body or soup

def clean_markdown(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"(?im)^\s*image\s*:?.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def yaml_string(value) -> str:
    if value is None:
        return '""'
    value = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'

def build_front_matter(source: dict, title: str, published_at: str | None) -> str:
    retrieved_at = datetime.now().astimezone().date().isoformat()
    return f"""---
title: {yaml_string(title)}
source_url: {yaml_string(source['url'])}
retrieved_at: {yaml_string(retrieved_at)}
document_version: {yaml_string(source['document_version'])}
published_at: {yaml_string(published_at or '')}
audience: {yaml_string(source['audience'])}
campus: {yaml_string(source['campus'])}
department: {yaml_string(source['department'])}
category: {yaml_string(source['category'])}
coverage: {yaml_string(source['coverage'])}
language: "vi"
source_domain: "{ALLOWED_DOMAIN}"
source_type: "official_public_web"
---

"""

def crawl_one(session: requests.Session, source: dict) -> Path:
    validate_url(source["url"])
    print(f"\n[GET] {source['url']}")
    response = session.get(source["url"], timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    title = extract_title(soup)
    published_at = extract_published_date(soup)
    main_node = find_main_content(soup)

    content_soup = BeautifulSoup(str(main_node), "lxml")
    remove_noise(content_soup)
    markdown = md(
        str(content_soup),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    markdown = clean_markdown(markdown)

    if len(markdown) < 250:
        raise RuntimeError(f"Extracted content is suspiciously short: {len(markdown)} chars")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / source["filename"]
    output_path.write_text(
        build_front_matter(source, title, published_at)
        + f"# {title}\n\n"
        + markdown
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] {output_path}")
    print(f"     category = {source['category']}")
    print(f"     campus   = {source['campus']}")
    print(f"     chars    = {len(markdown):,}")
    return output_path

def clean_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.md"):
        path.unlink()
        print(f"[CLEAN] removed {path}")

def write_manifest(successes: list[tuple[dict, Path]]) -> None:
    lines = [
        "# FPTU HCM Student Assistant — Corpus Manifest",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "| # | File | Category | Campus | Coverage | Source |",
        "|---|---|---|---|---|---|",
    ]
    for i, (source, path) in enumerate(successes, start=1):
        lines.append(
            f"| {i} | `{path.name}` | {source['category']} | {source['campus']} | "
            f"{source['coverage']} | {source['url']} |"
        )
    (OUTPUT_DIR / "CORPUS_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean_output_dir()

    session = make_session()
    successes: list[tuple[dict, Path]] = []
    failures: list[dict] = []

    for index, source in enumerate(SOURCES, start=1):
        print(f"\n===== {index}/{len(SOURCES)}: {source['category']} =====")
        try:
            path = crawl_one(session, source)
            successes.append((source, path))
        except Exception as exc:
            print(f"[ERROR] {source['url']}")
            print(f"        {type(exc).__name__}: {exc}")
            failures.append({"source": source, "error": str(exc)})
        time.sleep(1.5)

    write_manifest(successes)

    print("\n" + "=" * 64)
    print("FPTU HCM CORPUS SUMMARY")
    print("=" * 64)
    print(f"Success: {len(successes)}/{len(SOURCES)}")
    print(f"Failed : {len(failures)}")
    print(f"Output : {OUTPUT_DIR}")

    for source, path in successes:
        print(f"  ✓ {path.name:<42} {source['category']}")
    for item in failures:
        print(f"  ✗ {item['source']['category']}: {item['error']}")

    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
