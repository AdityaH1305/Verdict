# Vendored front-end libraries

Committed rather than loaded from a CDN, deliberately.

`tests/test_dashboard_api.py::test_dashboard_html_has_no_external_dependencies`
forbids any external reference in the dashboard, because a demo that breaks on a
bad connection is worse than a plainer one. Inlining these into `index.html` was
not an option either: GSAP's minified files carry `https://gsap.com` inside their
required licence banner, and that notice must not be stripped.

So they are served from the same origin as everything else, mounted at `/vendor`
by `src/api/main.py`, and referenced from `index.html` by relative path.

| File | Version | Retrieved | Source |
|---|---|---|---|
| `gsap.min.js` | 3.13.0 | 2026-09-05 | `cdnjs.cloudflare.com/ajax/libs/gsap/3.13.0/gsap.min.js` |
| `ScrollTrigger.min.js` | 3.13.0 | 2026-09-05 | `cdnjs.cloudflare.com/ajax/libs/gsap/3.13.0/ScrollTrigger.min.js` |
| `lenis.min.js` | 1.3.11 | 2026-09-05 | `cdn.jsdelivr.net/npm/lenis@1.3.11/dist/lenis.min.js` |

Byte-for-byte as published, licence banners intact.

## Licences

- **GSAP and ScrollTrigger** — GreenSock. Free for the great majority of uses
  under the standard "no charge" licence; see the banner at the top of each file
  and <https://gsap.com/community/standard-license>.
- **Lenis** — MIT, Studio Freight / Darkroom Engineering.

## Note for whoever extends the offline test

The offline-safety test scans **`index.html` only**, and that is on purpose.
These vendored files legitimately contain `https://gsap.com` in their licence
banners. They are never fetched over the network — they are read from disk and
served locally — so a URL in a comment is not an external dependency. Widening
that test to the whole `src/dashboard/` tree would fail for the wrong reason.

`tests/test_dashboard_api.py::TestVendoredLibraries` covers these instead: that
each one is served, and that each is large enough to be the real library rather
than an error page saved under a `.js` name.

## Updating

Re-download from the same URLs, keep the banners, bump the version and date in
the table above, and re-run `pytest tests/test_dashboard_api.py`.
