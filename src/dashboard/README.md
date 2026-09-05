# Dashboard

One page: [`index.html`](index.html), plus three vendored libraries in
[`vendor/`](vendor/README.md). Vanilla JS + CSS, **no build step, no `npm install`,
no CDN**. FastAPI serves the page at `GET /` and the libraries at `/vendor/*`.

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
| Was this the right call? (four-policy comparison + sensitivity chart) | `GET /reports/policy-eval` |
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

## Was this the right call?

An offline comparison of four retry policies over the held-out test set, computed by
`scripts/policy_eval.py` and served as a committed report. It answers the question the
rest of the page cannot: would a merchant who simply retried everything have done
better?

**Sometimes, yes, and the page says so in body text rather than in a footnote.** Net
value is `revenue − attempts × cost`, so the ranking depends entirely on what a retry
costs. Below ₹46.84 per attempt retrying everything nets more; above ₹365.20 a plain
decline-code rule engine does; Verdict wins the range between. The claim that holds at
every cost — and therefore the headline — is that Verdict captures 92.5% of all
recoverable revenue using 46% fewer attempts.

The cost per retry is an **assumption, not a measurement**, and the band says so without
needing a click. The sensitivity chart is what actually answers the question: both
crossovers are labelled directly on the lines, so a reader who believes a different cost
can read off their own answer.

No policy is allowed to see whether a retry would have worked: the outcome column is
dropped before any policy is asked to decide, and read only afterwards to score what each
one chose. `tests/test_policy_eval.py` proves it by handing every policy the raw frame
with every outcome flipped and asserting that not one decision moves.

The chart is hand-rolled inline SVG for the same reason as the money flow — the page may
not reference anything external.

## Progressive disclosure

Every band ends with one consistent affordance — a text button that opens a ruled
panel holding the technical layer for that band: the predicted-vs-realised revenue
comparison, the flow's action table (counts, shares and value) with the 0.50 / 0.25
cutoffs and the hedging rule, the PSI table with its threshold scale, the model class
names beside their merchant labels, the full retry simulation, and the measured model
metrics. Nothing was removed from the old
page; the model vocabulary moved one click deeper.

## Motion

GSAP + ScrollTrigger and Lenis, **vendored rather than loaded from a CDN** so the
offline guarantee survives — see [`vendor/README.md`](vendor/README.md) for provenance
and licences.

They replace hand-rolled motion rather than adding a second system on top:

- **Lenis** drives smooth scrolling, running off GSAP's ticker rather than its own
  `requestAnimationFrame`, so there is one animation loop rather than two. The three
  inner scroll containers (`.ledger-scroll`, `.policy-scroll`, `.chart-wrap`) carry
  `data-lenis-prevent` and keep native scrolling — without it Lenis swallows the wheel
  exactly where a reader needs to scroll a wide table sideways. `smoothTouch` is off, so
  touch devices keep the OS behaviour.
- **GSAP** drives the money-flow value tween, which previously ran its own rAF loop, with
  `lagSmoothing(0)` so a stalled frame cannot make the figures jump.
- **ScrollTrigger** plays the flow's draw-in when the diagram enters view rather than on
  load, and gives each section one short entrance. Every trigger is `once: true` and kills
  itself after firing.

Three things worth knowing if you change this:

1. **Nothing depends on the libraries loading.** `HAS_GSAP` / `HAS_LENIS` guard every
   integration point; remove the files and the page still renders, decides and scrolls.
   A test covers it.
2. **The entrance animates the section heading, not the section.** Fading a whole band
   promotes its entire subtree to a composite layer, and the ledger band is 4,689 px of
   table — measured, that alone took the worst frame from 12.8 ms to 35 ms.
3. **`ScrollTrigger.refresh()` runs after data lands**, debounced. Page height changes
   enormously once the ledger renders, and stale trigger positions are what strand a
   section at `opacity: 0`.

A `@media print` rule forces the finished state, so a print or a full-page capture can
never catch a section mid-reveal — the screenshots in `assets/` are taken with
`captureBeyondViewport`.

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
