from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
WEB_CACHE_DIR = DATA_DIR / "web"

SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md"}
SKIPPED_WEB_SUFFIXES = {
    ".7z",
    ".avi",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".xls",
    ".xlsx",
    ".zip",
}
DEFAULT_ALLOWED_DOMAINS = (
    "gla.ac.in",
    "www.gla.ac.in",
    "glauniversity.in",
    "www.glauniversity.in",
    "student.glauniversity.in",
)
USER_AGENT = "GLAChatbotCrawler/1.0 (+https://www.gla.ac.in/)"


def ensure_data_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    cleaned = Path(filename or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("-.")
    return cleaned or "upload"


def is_supported_upload(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES


def save_uploaded_bytes(filename: str, content: bytes) -> Path:
    ensure_data_directories()
    safe_name = sanitize_filename(filename)
    destination = UPLOADS_DIR / safe_name
    destination.write_bytes(content)
    return destination


def normalize_url(url: str) -> str:
    trimmed = url.strip()
    if not trimmed:
        return ""

    if not trimmed.startswith(("http://", "https://")):
        trimmed = f"https://{trimmed}"

    without_fragment, _ = urldefrag(trimmed)
    parsed = urlparse(without_fragment)
    normalized_path = parsed.path or "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path.rstrip("/")

    return parsed._replace(path=normalized_path, params="", query="").geturl()


def is_allowed_domain(url: str, allowed_domains: set[str]) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)


def should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(suffix) for suffix in SKIPPED_WEB_SUFFIXES)


def build_allowed_domains(start_url: str, configured_domains: str | None) -> set[str]:
    domains = {
        domain.strip().lower()
        for domain in (configured_domains or "").split(",")
        if domain.strip()
    }
    if domains:
        return domains

    hostname = (urlparse(start_url).hostname or "").lower()
    if hostname:
        return {hostname}

    return set(DEFAULT_ALLOWED_DOMAINS)


def _get_robot_parser(page_url: str) -> RobotFileParser:
    parsed = urlparse(page_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = requests.get(
            robots_url,
            timeout=10,
            headers={"User-Agent": USER_AGENT},
        )
        if response.ok:
            parser.parse(response.text.splitlines())
    except requests.RequestException:
        return parser

    return parser


def _extract_text_and_links(html: str, page_url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "form", "iframe"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else page_url
    root = soup.find("main") or soup.find("article") or soup.body or soup

    text_blocks: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        block = " ".join(element.stripped_strings)
        if not block:
            continue
        if len(block) < 20 and element.name in {"p", "li", "td"}:
            continue
        text_blocks.append(block)

    if not text_blocks:
        fallback_text = root.get_text("\n", strip=True)
        text_blocks = [segment.strip() for segment in fallback_text.splitlines() if segment.strip()]

    deduped_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for block in text_blocks:
        normalized_block = re.sub(r"\s+", " ", block).strip()
        lowercase_block = normalized_block.lower()
        if lowercase_block in seen_blocks:
            continue
        seen_blocks.add(lowercase_block)
        deduped_blocks.append(normalized_block)

    discovered_links: list[str] = []
    seen_links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        candidate = normalize_url(urljoin(page_url, anchor["href"]))
        if not candidate or candidate in seen_links:
            continue
        seen_links.add(candidate)
        discovered_links.append(candidate)

    return title, "\n\n".join(deduped_blocks), discovered_links


def _record_path_for_url(url: str) -> Path:
    parsed = urlparse(url)
    slug_source = f"{parsed.netloc}{parsed.path}".strip("/") or "home"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug_source).strip("-").lower()[:80] or "page"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return WEB_CACHE_DIR / f"{slug}-{digest}.json"


def crawl_site(start_url: str, max_pages: int, allowed_domains: set[str]) -> dict[str, Any]:
    ensure_data_directories()

    normalized_start = normalize_url(start_url)
    if not normalized_start:
        raise ValueError("A valid website URL is required.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    robot_parsers: dict[str, RobotFileParser] = {}
    queue: deque[str] = deque([normalized_start])
    visited: set[str] = set()
    saved_urls: list[str] = []

    while queue and len(saved_urls) < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        if not is_allowed_domain(current_url, allowed_domains) or should_skip_url(current_url):
            continue

        hostname = (urlparse(current_url).hostname or "").lower()
        if hostname not in robot_parsers:
            robot_parsers[hostname] = _get_robot_parser(current_url)

        if not robot_parsers[hostname].can_fetch(USER_AGENT, current_url):
            continue

        try:
            response = session.get(current_url, timeout=15)
            response.raise_for_status()
        except requests.RequestException:
            continue

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            continue

        title, content, links = _extract_text_and_links(response.text, current_url)
        if len(content) < 250:
            continue

        record = {
            "url": current_url,
            "title": title,
            "content": content,
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        _record_path_for_url(current_url).write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved_urls.append(current_url)

        for link in links:
            if link not in visited and is_allowed_domain(link, allowed_domains) and not should_skip_url(link):
                queue.append(link)

    if not saved_urls:
        raise ValueError("No crawlable HTML pages were collected from that URL.")

    return {
        "start_url": normalized_start,
        "pages_indexed": len(saved_urls),
        "urls": saved_urls,
    }
