#!/usr/bin/env python3
"""Corpus validation script for FPT University RAG dataset.

Validates:
1. Markdown documents in data/university/:
   - UTF-8 encoding integrity (no decode errors or replacement chars)
   - Valid YAML frontmatter syntax
   - Required frontmatter fields: title, source_url, retrieved_at, document_version, audience
   - audience is one of: student, faculty, staff, all
   - doc_id matches filename stem
   - Non-empty body content (>100 characters)
2. sources.csv inventory file:
   - File existence and valid CSV structure
   - Required columns: doc_id, title, source_url, retrieved_at, document_version,
     audience, campus, department, category, language
   - 1-to-1 bidirectional mapping with .md files in the corpus
   - Metadata field consistency between frontmatter and sources.csv

Usage:
    python scripts/validate_corpus.py
    python scripts/validate_corpus.py --data-dir data/university
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allowed audience values according to corpus specifications
ALLOWED_AUDIENCES: frozenset[str] = frozenset({"student", "faculty", "staff", "all"})

# Required frontmatter fields in each .md file
REQUIRED_FRONTMATTER_FIELDS: list[str] = [
    "title",
    "source_url",
    "retrieved_at",
    "document_version",
    "audience",
]

# Required columns in sources.csv
REQUIRED_CSV_COLUMNS: list[str] = [
    "doc_id",
    "title",
    "source_url",
    "retrieved_at",
    "document_version",
    "audience",
    "campus",
    "department",
    "category",
    "language",
]

# ANSI color codes
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"


def _supports_color() -> bool:
    """Detect if stdout supports ANSI color escape codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        # Enable Windows 10/11 native ANSI escape processing
        try:
            os.system("")
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR: bool = _supports_color()


def color_text(text: str, color_code: str) -> str:
    """Wrap text in ANSI color escape codes if color is enabled."""
    if not USE_COLOR:
        return text
    return f"{color_code}{text}{ANSI_RESET}"


def tag_ok() -> str:
    """Return formatted green OK tag."""
    return color_text("[  OK  ]", ANSI_GREEN + ANSI_BOLD)


def tag_fail() -> str:
    """Return formatted red FAIL tag."""
    return color_text("[ FAIL ]", ANSI_RED + ANSI_BOLD)


def tag_warn() -> str:
    """Return formatted yellow WARN tag."""
    return color_text("[ WARN ]", ANSI_YELLOW + ANSI_BOLD)


def tag_info() -> str:
    """Return formatted cyan INFO tag."""
    return color_text("[ INFO ]", ANSI_CYAN + ANSI_BOLD)


@dataclass
class CheckResult:
    """Result of an individual validation check."""
    target: str
    check_name: str
    passed: bool
    message: str
    is_warning: bool = False


@dataclass
class ValidationReport:
    """Aggregated validation results across all checks."""
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, target: str, check_name: str, passed: bool, message: str, is_warning: bool = False) -> None:
        """Add a check result to the report."""
        self.checks.append(CheckResult(
            target=target,
            check_name=check_name,
            passed=passed,
            message=message,
            is_warning=is_warning,
        ))

    @property
    def total_count(self) -> int:
        """Total number of checks performed."""
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for c in self.checks if c.passed and not c.is_warning)

    @property
    def failed_count(self) -> int:
        """Number of failed checks."""
        return sum(1 for c in self.checks if not c.passed and not c.is_warning)

    @property
    def warning_count(self) -> int:
        """Number of warning checks."""
        return sum(1 for c in self.checks if c.is_warning)

    @property
    def is_success(self) -> bool:
        """True if no checks failed."""
        return self.failed_count == 0


def parse_yaml_frontmatter(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """Parse YAML frontmatter and body content from markdown text.

    Returns:
        tuple of (metadata_dict, body_content, error_message)
    """
    if not text.startswith("---"):
        return None, text, "File does not begin with YAML frontmatter delimiter ('---')"

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text, "File has unclosed YAML frontmatter delimiter ('---')"

    frontmatter_raw = parts[1].strip()
    body_content = parts[2]

    # Try PyYAML first if available
    try:
        import yaml
        try:
            parsed = yaml.safe_load(frontmatter_raw)
            if not isinstance(parsed, dict):
                return None, body_content, f"Frontmatter did not parse as dictionary, got {type(parsed).__name__}"
            return parsed, body_content, None
        except Exception as e:
            return None, body_content, f"YAML parse error: {e}"
    except ImportError:
        # Fallback regex parser for key: value pairs if PyYAML is not installed
        parsed: dict[str, Any] = {}
        for line in frontmatter_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-zA-Z0-9_\-]+)\s*:\s*(.*)$", line)
            if match:
                key, val = match.group(1), match.group(2).strip()
                # Strip quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                parsed[key] = val
        return parsed, body_content, None


def validate_markdown_file(file_path: Path, report: ValidationReport) -> dict[str, Any] | None:
    """Validate a single markdown file against all corpus requirements.

    Checks:
    - Encoding: UTF-8 with no replacement characters
    - Frontmatter: Valid YAML structure and delimiter
    - Required fields: title, source_url, retrieved_at, document_version, audience
    - audience: one of 'student', 'faculty', 'staff', 'all'
    - doc_id: matches filename stem
    - Body content: non-empty (>100 characters)

    Returns:
        The extracted frontmatter dictionary if parsing succeeded, else None.
    """
    filename = file_path.name
    stem = file_path.stem

    # 1. Encoding check
    try:
        raw_bytes = file_path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="strict")
        if "\ufffd" in text:
            report.add(filename, "Encoding", False, "Contains Unicode replacement character (\\ufffd)")
            return None
        report.add(filename, "Encoding", True, f"UTF-8 valid ({len(raw_bytes):,} bytes)")
    except UnicodeDecodeError as e:
        report.add(filename, "Encoding", False, f"UTF-8 decoding failed: {e}")
        return None
    except Exception as e:
        report.add(filename, "Encoding", False, f"Error reading file: {e}")
        return None

    # 2. YAML frontmatter parsing
    fm, body, parse_err = parse_yaml_frontmatter(text)
    if parse_err is not None or fm is None:
        report.add(filename, "Frontmatter Syntax", False, parse_err or "Unknown parse error")
        return None
    report.add(filename, "Frontmatter Syntax", True, f"Parsed {len(fm)} frontmatter keys")

    # 3. Required frontmatter fields
    missing_fields = [k for k in REQUIRED_FRONTMATTER_FIELDS if k not in fm or str(fm[k]).strip() == ""]
    if missing_fields:
        report.add(
            filename,
            "Required Fields",
            False,
            f"Missing required fields: {', '.join(missing_fields)}",
        )
    else:
        report.add(
            filename,
            "Required Fields",
            True,
            f"All {len(REQUIRED_FRONTMATTER_FIELDS)} required fields present",
        )

    # 4. Audience validation
    audience = str(fm.get("audience", "")).strip().lower()
    if not audience:
        report.add(filename, "Audience Value", False, "Audience field is empty or missing")
    elif audience not in ALLOWED_AUDIENCES:
        report.add(
            filename,
            "Audience Value",
            False,
            f"Invalid audience '{audience}'. Must be one of: {sorted(ALLOWED_AUDIENCES)}",
        )
    else:
        report.add(filename, "Audience Value", True, f"Audience '{audience}' is valid")

    # 5. doc_id matches filename stem
    fm_doc_id = fm.get("doc_id")
    if fm_doc_id is not None:
        if str(fm_doc_id).strip() != stem:
            report.add(
                filename,
                "doc_id Matching",
                False,
                f"Frontmatter doc_id '{fm_doc_id}' does not match filename stem '{stem}'",
            )
        else:
            report.add(filename, "doc_id Matching", True, f"doc_id matches stem '{stem}'")
    else:
        # doc_id not explicitly declared in frontmatter; fallback to stem
        report.add(
            filename,
            "doc_id Matching",
            True,
            f"Implicit doc_id from stem '{stem}'",
        )

    # 6. Non-empty body content (>100 characters)
    stripped_body = body.strip()
    body_length = len(stripped_body)
    if body_length <= 100:
        report.add(
            filename,
            "Body Content",
            False,
            f"Body content too short: {body_length} characters (must be > 100)",
        )
    else:
        report.add(
            filename,
            "Body Content",
            True,
            f"Body content non-empty ({body_length:,} characters > 100)",
        )

    return fm


def validate_sources_csv(
    csv_path: Path,
    md_files: list[Path],
    frontmatters: dict[str, dict[str, Any]],
    report: ValidationReport,
) -> None:
    """Validate sources.csv existence, schema, and 1-to-1 mapping with corpus documents.

    Checks:
    - File exists and is non-empty
    - Header contains all REQUIRED_CSV_COLUMNS
    - 1-to-1 mapping: every .md file is in CSV, every CSV row corresponds to an .md file
    - No duplicate doc_ids in CSV
    - Consistency: values in CSV match corresponding frontmatter fields
    """
    target = csv_path.name

    # 1. Existence check
    if not csv_path.exists():
        report.add(target, "File Existence", False, f"File not found at: {csv_path}")
        return

    if csv_path.stat().st_size == 0:
        report.add(target, "File Existence", False, "File is empty")
        return

    report.add(target, "File Existence", True, f"Found {csv_path.name} ({csv_path.stat().st_size:,} bytes)")

    # 2. Read and parse CSV
    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except Exception as e:
        report.add(target, "CSV Parsing", False, f"Failed to read CSV: {e}")
        return

    report.add(target, "CSV Parsing", True, f"Successfully parsed {len(rows)} data rows")

    # 3. Column schema check
    missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in fieldnames]
    if missing_cols:
        report.add(
            target,
            "Required Columns",
            False,
            f"Missing required columns: {', '.join(missing_cols)}",
        )
    else:
        report.add(
            target,
            "Required Columns",
            True,
            f"All {len(REQUIRED_CSV_COLUMNS)} required columns present",
        )

    # 4. 1-to-1 Mapping check
    expected_doc_ids = {p.stem for p in md_files}
    csv_doc_ids = [r.get("doc_id", "").strip() for r in rows if r.get("doc_id")]
    unique_csv_doc_ids = set(csv_doc_ids)

    # Duplicate doc_ids
    if len(csv_doc_ids) != len(unique_csv_doc_ids):
        duplicates = [i for i in unique_csv_doc_ids if csv_doc_ids.count(i) > 1]
        report.add(target, "No Duplicates", False, f"Duplicate doc_ids found: {duplicates}")
    else:
        report.add(target, "No Duplicates", True, f"All {len(csv_doc_ids)} doc_ids are unique")

    # Mapping comparison
    missing_in_csv = expected_doc_ids - unique_csv_doc_ids
    extra_in_csv = unique_csv_doc_ids - expected_doc_ids

    if missing_in_csv or extra_in_csv:
        details = []
        if missing_in_csv:
            details.append(f"In .md but missing in CSV: {sorted(missing_in_csv)}")
        if extra_in_csv:
            details.append(f"In CSV but missing in .md: {sorted(extra_in_csv)}")
        report.add(
            target,
            "1-to-1 Mapping",
            False,
            "; ".join(details),
        )
    else:
        report.add(
            target,
            "1-to-1 Mapping",
            True,
            f"Perfect 1-to-1 match ({len(unique_csv_doc_ids)} files <-> {len(rows)} CSV rows)",
        )

    # 5. Metadata field consistency between frontmatter and sources.csv
    consistency_errors: list[str] = []
    fields_to_check = [
        "title",
        "source_url",
        "retrieved_at",
        "document_version",
        "audience",
        "campus",
        "department",
        "category",
        "language",
    ]

    for row in rows:
        row_doc_id = row.get("doc_id", "").strip()
        fm = frontmatters.get(row_doc_id)
        if not fm:
            continue

        for col in fields_to_check:
            if col in row and col in fm:
                csv_val = str(row[col]).strip()
                fm_val = str(fm[col]).strip()
                if csv_val != fm_val:
                    consistency_errors.append(
                        f"doc '{row_doc_id}' field '{col}' mismatch: CSV='{csv_val}' vs FM='{fm_val}'"
                    )

    if consistency_errors:
        report.add(
            target,
            "Metadata Consistency",
            False,
            f"{len(consistency_errors)} field mismatch(es): {consistency_errors[:3]}",
        )
    else:
        report.add(
            target,
            "Metadata Consistency",
            True,
            f"All fields strictly match frontmatter across {len(rows)} documents",
        )


def run_validation(data_dir: Path, sources_csv_path: Path | None = None) -> ValidationReport:
    """Execute complete corpus validation pipeline."""
    report = ValidationReport()

    # Verify directory existence
    if not data_dir.exists() or not data_dir.is_dir():
        report.add(
            str(data_dir),
            "Directory Exists",
            False,
            f"Corpus directory does not exist: {data_dir}",
        )
        return report

    report.add(str(data_dir.name), "Directory Exists", True, f"Found directory: {data_dir}")

    # Discover .md files
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        report.add(
            str(data_dir.name),
            "Document Discovery",
            False,
            f"No .md files found in: {data_dir}",
        )
        return report

    report.add(
        str(data_dir.name),
        "Document Discovery",
        True,
        f"Found {len(md_files)} markdown documents",
    )

    # Validate each markdown file
    frontmatters: dict[str, dict[str, Any]] = {}
    for file_path in md_files:
        fm = validate_markdown_file(file_path, report)
        if fm is not None:
            # Key by stem
            frontmatters[file_path.stem] = fm

    # Validate sources.csv
    csv_target = sources_csv_path or (data_dir / "sources.csv")
    validate_sources_csv(csv_target, md_files, frontmatters, report)

    return report


def print_report(report: ValidationReport, data_dir: Path) -> None:
    """Print beautifully formatted validation report to stdout."""
    header_line = "=" * 78
    sub_line = "-" * 78

    print(header_line)
    print(color_text(f" FPT UNIVERSITY RAG CORPUS VALIDATION - {data_dir}", ANSI_BOLD))
    print(header_line)

    current_target = ""
    for check in report.checks:
        if check.target != current_target:
            current_target = check.target
            print(f"\n{color_text('>>> ' + current_target, ANSI_BOLD)}")

        if check.is_warning:
            status = tag_warn()
        elif check.passed:
            status = tag_ok()
        else:
            status = tag_fail()

        print(f"  {status} {check.check_name:<24} : {check.message}")

    print("\n" + header_line)
    print(color_text(" VALIDATION SUMMARY", ANSI_BOLD))
    print(sub_line)
    print(f"  Total checks performed : {report.total_count}")
    print(f"  Passed checks          : {color_text(str(report.passed_count), ANSI_GREEN if report.passed_count else '')}")
    print(f"  Failed checks          : {color_text(str(report.failed_count), ANSI_RED if report.failed_count else '')}")
    if report.warning_count:
        print(f"  Warnings               : {color_text(str(report.warning_count), ANSI_YELLOW)}")

    print(sub_line)
    if report.is_success:
        print(color_text("  RESULT: ALL CHECKS PASSED (exit code 0)", ANSI_GREEN + ANSI_BOLD))
    else:
        print(color_text(f"  RESULT: FAILED WITH {report.failed_count} ERROR(S) (exit code 1)", ANSI_RED + ANSI_BOLD))
    print(header_line + "\n")


def main() -> int:
    """Command-line entrypoint for corpus validation."""
    parser = argparse.ArgumentParser(
        description="Validate FPT University RAG corpus (.md files and sources.csv)."
    )
    # Default path relative to repository root or script location
    default_corpus_dir = Path(__file__).resolve().parent.parent / "data" / "university"
    if not default_corpus_dir.exists():
        default_corpus_dir = Path("data/university")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_corpus_dir,
        help="Path to directory containing university corpus markdown files (default: data/university)",
    )
    parser.add_argument(
        "--sources-csv",
        type=Path,
        default=None,
        help="Path to sources.csv (default: <data-dir>/sources.csv)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output in terminal",
    )

    args = parser.parse_args()

    global USE_COLOR
    if args.no_color:
        USE_COLOR = False

    data_dir = args.data_dir.resolve()
    csv_path = args.sources_csv.resolve() if args.sources_csv else None

    report = run_validation(data_dir=data_dir, sources_csv_path=csv_path)
    print_report(report, data_dir)

    return 0 if report.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
