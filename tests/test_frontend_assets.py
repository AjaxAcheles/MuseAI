"""The browser must never reach the network.

MuseAI is a local-only application. Every asset the browser loads is served from
``museai/web/static``. These tests fail if a template, stylesheet, or first-party script
acquires a reference to a CDN, a font host, or any other remote origin.

The vendored libraries under ``static/vendor/`` are exempt from the text scan: their minified
bundles carry upstream license headers and source-map comments containing URLs that are never
requested at runtime. They are checked for existence instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1] / "museai" / "web"
TEMPLATES = WEB_ROOT / "templates"
STATIC = WEB_ROOT / "static"
VENDOR = STATIC / "vendor"

# Substrings that can only mean "this asset is fetched from somewhere else".
_REMOTE_MARKERS = (
    "http://",
    "https://",
    "cdnjs",
    "jsdelivr",
    "unpkg",
    "fonts.googleapis",
    "fonts.gstatic",
    "cdn.",
)

# Protocol-relative URLs: src="//example.com/x.js". Anchored on a quote or paren so that
# a JS line comment (`// foo`) or a CSS comment does not trip it.
_PROTOCOL_RELATIVE = re.compile(r"""["'(]\s*//[a-z0-9-]+\.""", re.IGNORECASE)

# Every href/src in a template must resolve locally: either a url_for(...) call, a Jinja
# expression, a root-relative path, or a fragment/mailto-free anchor.
_ASSET_REF = re.compile(r"""<(?:script|link|img|source)\b[^>]*?\b(?:src|href)\s*=\s*["']([^"']*)["']""", re.IGNORECASE)


def _first_party_files() -> list[Path]:
    """Templates, stylesheets, and our own scripts -- never the vendored bundles."""
    files = sorted(TEMPLATES.rglob("*.html"))
    files += sorted(STATIC.rglob("*.css"))
    files += sorted(STATIC.rglob("*.js"))
    return [path for path in files if VENDOR not in path.parents]


def _ids(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(WEB_ROOT)) for path in paths]


FIRST_PARTY = _first_party_files()


def test_first_party_file_set_is_not_empty() -> None:
    """Guard the guard: a bad glob must not silently pass every other test here."""
    assert FIRST_PARTY, "no templates/css/js were discovered -- the asset scan is not running"


@pytest.mark.parametrize("path", FIRST_PARTY, ids=_ids(FIRST_PARTY))
def test_no_remote_asset_reference(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for marker in _REMOTE_MARKERS:
            assert marker not in lowered, f"{path.name}:{lineno} references remote origin {marker!r}: {line.strip()}"
        assert not _PROTOCOL_RELATIVE.search(line), f"{path.name}:{lineno} uses a protocol-relative URL: {line.strip()}"


@pytest.mark.parametrize("path", sorted(TEMPLATES.rglob("*.html")), ids=lambda p: p.name)
def test_template_assets_resolve_locally(path: Path) -> None:
    """Every <script>/<link>/<img> source is a local static path."""
    for ref in _ASSET_REF.findall(path.read_text(encoding="utf-8")):
        ref = ref.strip()
        if not ref or ref.startswith("#"):
            continue
        local = ref.startswith("{{") or ref.startswith("/") or ref.startswith("data:")
        assert local, f"{path.name}: asset reference {ref!r} is not a local static path"


def test_css_imports_nothing_remote() -> None:
    for path in sorted(STATIC.rglob("*.css")):
        if VENDOR in path.parents:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "@import" not in text, f"{path.name}: @import may pull a remote stylesheet"
        assert "@font-face" not in text, f"{path.name}: @font-face may pull a remote font"


def test_vendored_assets_exist_on_disk() -> None:
    """base.html references these by name; a missing file means a blank page offline."""
    for filename in ("bootstrap.min.css", "marked.min.js", "purify.min.js"):
        asset = VENDOR / filename
        assert asset.is_file(), f"vendored asset {filename} is missing"
        assert asset.stat().st_size > 0, f"vendored asset {filename} is empty"


def test_vendor_readme_documents_every_bundle() -> None:
    """A vendored bundle without recorded provenance cannot be audited or updated."""
    readme = VENDOR / "README.md"
    assert readme.is_file(), "museai/web/static/vendor/README.md is missing"
    text = readme.read_text(encoding="utf-8")
    for bundle in sorted(VENDOR.glob("*.min.*")):
        assert bundle.name in text, f"{bundle.name} is vendored but undocumented in vendor/README.md"


def test_every_referenced_vendor_file_is_vendored() -> None:
    """A template naming vendor/x.js that does not exist would silently fetch nothing."""
    referenced = set()
    for path in TEMPLATES.rglob("*.html"):
        referenced.update(re.findall(r"vendor/([\w.-]+)", path.read_text(encoding="utf-8")))
    for name in referenced:
        assert (VENDOR / name).is_file(), f"templates reference vendor/{name}, which is not present"


def test_the_test_suite_does_not_write_to_the_real_log_files():
    """pytest must not append to `logs/fsm.log` — the app writes there live."""
    from museai.core import logging_setup

    assert logging_setup.LOG_DIR != Path("logs"), (
        "tests/conftest.py must redirect LOG_DIR before any museai import; "
        "otherwise test traffic interleaves with the running app's logs"
    )
    assert "museai-test-logs-" in str(logging_setup.LOG_DIR)
