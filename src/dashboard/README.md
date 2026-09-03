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

## Design

"Settlement" — the page is laid out as a ledger sheet rather than a grid of cards.
Elevation is spent exactly once, on the settlement summary at the top; every band
below it is flat on paper, separated by hairline rules. Merchant language leads and
the model layer sits behind explicit "show the numbers" disclosures, so a shop owner
can read the page end to end without meeting the word *calibration*, and an engineer
can still reach every figure the API returns.

Palette: cool paper `#F5F6F9` with an indigo accent `#2B3F8C`. Green `#0F7A5A` is
money recovered, amber `#A96400` means a human has to act, and fraud-blocked is plum
`#6B3E86` rather than red — a blocked fraud attempt is the system working, not
failing. Both themes are token swaps; the toggle persists to `localStorage` and the
un-stamped system-dark case is handled too.

Type is three system stacks, not a web font. That is forced by
`test_dashboard_html_has_no_external_dependencies`, which fails the build on any
`https://` in the file — including the `xmlns` attribute on an inline `<svg>`, which
is why there isn't one. The numerals carry the display weight (tabular figures
throughout); Georgia appears once per band at most, for the single plain-language
sentence that says what the numbers mean.

## Sections

| Section | Source |
|---|---|
| The verdict (headline recovered revenue, where the failed money went) | `GET /stats/recovered-revenue` |
| What needs you (escalations, pattern shift, fraud blocks) | the batch response, plus `GET /stats/drift` |
| Pattern-shift detail, and the monitor firing vs not firing | `GET /stats/drift`, `GET /reports/drift-comparison` |
| **Where your money went** — the money flow | the batch response (`amount`, `predicted_category`, `action`, `retry_success`) |
| Cards and UPI fail differently | `GET /stats/breakdown` → `category_by_method`, `by_payment_method` |
| We stopped hammering the banks (retry storm) | `GET /reports/retry-storm` |
| Every decision, on the record (the ledger) | `POST /simulate/*` then `GET /decisions` |
| Uncertainty range per row | `recovery_interval` on each decision |
| How this works (architecture, fraud guarantee, measured quality) | `GET /reports/metrics`, `GET /health` |

The page computes no decisions of its own — every number comes from the backend.
Where it sums or divides (the escalated value in "what needs you", the card-vs-UPI
skew), it is summing fields the backend already returned for that same batch.

## The money flow

The centrepiece. A hand-rolled SVG Sankey on a deep ink stage — the one place the
page goes dark — showing failed money on the left, why it failed, what Verdict did,
and where it ended up. **Ribbon thickness is rupees throughout.** Hover a ribbon and
it writes a sentence; select a reason or action block and it drives the ledger's
existing filters, so the diagram and the table can never disagree.

Switching batches tweens the underlying values over 700ms and recomputes the paths
each frame, so picking "a day when patterns shifted" visibly swells the fraud and UPI
currents. That is the page's one orchestrated motion, and `prefers-reduced-motion`
skips straight to the final state.

Two properties are worth knowing:

- **It is arithmetically closed.** Every stage sums to the same total, and the
  Recovered node equals `actual_recovered_value` from `/stats/recovered-revenue`
  exactly. Both are checked in the browser during verification.
- **It makes the fraud rule geometric.** Money blocked as fraud has exactly one
  outgoing ribbon, to "Left alone". There is no path from fraud to a recovery action
  because the agent cannot produce one — the diagram cannot draw what does not exist.
  The stage footer states the same thing as a number, counted from the batch on every
  run.

No charting library: the CSP/offline test forbids external references, and this is a
fixed 1-4-5-4 graph, so the layout is arithmetic rather than a dependency. Below 720px
it becomes three stacked steps with the same labels, colours, rupees and click-to-filter.

## Progressive disclosure

Every band ends with one consistent affordance — a text button that opens a ruled
panel holding the technical layer for that band: the predicted-vs-realised revenue
comparison, the flow's action table (counts, shares and value) with the 0.50 / 0.25
cutoffs and the hedging rule, the PSI table with its threshold scale, the model class
names beside their merchant labels, the full retry simulation, and the measured model
metrics. Nothing was removed from the old
page; the model vocabulary moved one click deeper.

## Quota safety

Normal dashboard use makes **zero** live-provider calls. Everything that loops
over transactions is served by the API's bulk agent, which is pinned to the
offline `TemplateAdapter`; explanations are still grounded in the real error-code
taxonomy, just generated deterministically.

The single exception is the **"Ask the live model to explain this"** button inside
an expanded ledger entry, which posts that one transaction to `POST /decide`. It is opt-in,
one call per click, and absent entirely on fraud-blocked rows — those never reach
a language model at all.

`tests/test_dashboard_api.py::TestQuotaSafety` enforces this with a spy adapter
that fails the build if a looping endpoint ever touches the live provider.

## Randomness

**"Run simulation" draws a fresh random sample from the held-out set on every
click** — different transactions, different revenue, different action mix. It
originally used `.head(n)`, which replayed one byte-identical batch forever.

Pass `?seed=N` to `/simulate/seed` to pin a batch for a scripted demo or a test.
The pipeline scripts (`train.py`, `prepare_data.py`, the generator) keep their
fixed seeds — regression numbers have to be reproducible, and
`scripts/evaluate_agent.py` runs on the full test set so it is unaffected either
way.

The stats panels read the *same* sample the feed shows, so the headline revenue
always describes the rows underneath it.

## Implementation notes

- The feed is the **audit log** (`/decisions`), which carries the explanation and
  the grounded code meaning but not the model inputs. The live-explain button
  needs those inputs, so the feature columns from the `/simulate/seed` response
  are joined by `transaction_id` client-side rather than widening the audit
  record with fields no auditor would want.
- The ledger renders the most recent 60 decisions. The stats panels already
  summarise the whole batch; rendering ~1,000 rows produced a 20,000px page.
- Fraud-blocked rows show "not scored" rather than a probability, because none is
  ever computed for them. That makes the architectural hard rule visible in the
  UI instead of merely asserted in the docs.
- The P(recovery) cell shows the point estimate with its uncertainty interval
  beneath, plus a small bar placing that range on 0..1 with a tick at the
  threshold the row's action turned on. The bar goes amber when uncertainty
  actually changed the action, and the expanded "why" then says which rule fired
  and why. Rows with no interval (fraud-blocked, or an older model artifact)
  render exactly as they did before.
