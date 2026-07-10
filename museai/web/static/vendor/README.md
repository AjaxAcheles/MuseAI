# Vendored browser assets

MuseAI runs entirely on a local machine and **must work with the Internet disabled**. Every
third-party asset the browser loads lives in this directory and is served from
`/static/vendor/…`. Nothing is fetched from a CDN, a font host, or any other remote origin.

`tests/test_frontend_assets.py` enforces this. It fails the build if a template, stylesheet, or
first-party script gains a remote reference.

## Contents

| File | Library | Version | License | SHA-256 |
| --- | --- | --- | --- | --- |
| `bootstrap.min.css` | [Bootstrap](https://github.com/twbs/bootstrap) | 5.3.3 | MIT | `3c8f27e6009ccfd710a905e6dcf12d0ee3c6f2ac7da05b0572d3e0d12e736fc8` |
| `marked.min.js` | [marked](https://github.com/markedjs/marked) | 12.0.2 | MIT | `15fabce5b65898b32b03f5ed25e9f891a729ad4c0d6d877110a7744aa847a894` |
| `purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) | 3.1.6 | Apache-2.0 / MPL-2.0 | `c0845096a7c4a6741f362ac506c94c1c7d27dc603bcc1bf64a587f76f2dbe3a1` |

The URLs above are provenance for a human reader. They are never requested at runtime, and the
asset test ignores this file for exactly that reason.

## Why only a stylesheet from Bootstrap

We vendor Bootstrap's CSS but **not** its JavaScript bundle. The interactive components we use —
nav tabs, the offcanvas seed drawer, and toasts — are driven by roughly eighty lines of vanilla JS
in `static/js/ui.js` that toggle Bootstrap's own class names (`.show`, `.active`). This avoids an
additional ~80 KB dependency and keeps the JS surface small enough to read.

`marked` renders committed beat prose as Markdown; `DOMPurify` sanitizes it before it reaches the
DOM. Model output is untrusted input, so the sanitizer is not optional.

## Adding an asset

1. Download it once, by hand, and commit the file here.
2. Record the version, license, and `sha256sum` in the table above.
3. Reference it as `{{ url_for('static', filename='vendor/<file>') }}` — never a bare path.

Prefer vanilla CSS and JS. A new dependency needs to earn its bytes.
