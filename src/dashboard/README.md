# Dashboard

One self-contained file: [`index.html`](index.html). Vanilla JS + CSS, **no build
step, no `npm install`, no CDN**. FastAPI serves it at `GET /`.

```bash
uvicorn src.api.main:app
# then open http://localhost:8000
```

## Why no framework

The brief asked for the approach fastest to build and simplest to run locally:

- **One command, one server, one port.** Same origin as the API, so no CORS, no
  proxy config, and no second dev server to keep alive during a demo.
- **Works offline.** A CDN-hosted React would break the demo on a bad connection.
  `tests/test_dashboard_api.py` asserts the page contains no external references,
  so that stays true.
- Four sections of bar charts don't need a component framework, and hand-rolled
  CSS bars were less work than wiring a chart library into a no-build page.

Node 24 is available if this ever outgrows a single file — but it hasn't.

## Sections

| Section | Source |
|---|---|
| Recovered revenue (headline) | `GET /stats/recovered-revenue` |
| Failure categories, split card vs UPI | `GET /stats/breakdown` → `category_by_method` |
| Action mix | `GET /stats/breakdown` → `by_action` |
| Retry storm before/after | `GET /reports/retry-storm` |
| Live decision feed + expandable "why" | `POST /simulate/seed` then `GET /decisions` |

The page computes no decisions of its own — every number comes from the backend.

## Quota safety

Normal dashboard use makes **zero** live-provider calls. Everything that loops
over transactions is served by the API's bulk agent, which is pinned to the
offline `TemplateAdapter`; explanations are still grounded in the real error-code
taxonomy, just generated deterministically.

The single exception is the **"Explain live with Gemini"** button inside an
expanded row, which posts that one transaction to `POST /decide`. It is opt-in,
one call per click, and absent entirely on fraud-blocked rows — those never reach
a language model at all.

`tests/test_dashboard_api.py::TestQuotaSafety` enforces this with a spy adapter
that fails the build if a looping endpoint ever touches the live provider.

## Implementation notes

- The feed is the **audit log** (`/decisions`), which carries the explanation and
  the grounded code meaning but not the model inputs. The live-explain button
  needs those inputs, so the feature columns from the `/simulate/seed` response
  are joined by `transaction_id` client-side rather than widening the audit
  record with fields no auditor would want.
- The feed renders the most recent 60 decisions. The stats panels already
  summarise the whole batch; rendering ~1,000 rows produced a 20,000px page.
- Fraud-blocked rows show "not scored" rather than a probability, because none is
  ever computed for them. That makes the architectural hard rule visible in the
  UI instead of merely asserted in the docs.
