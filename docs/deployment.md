# Deployment

**Recommendation: Render (free tier), via the committed [`render.yaml`](../render.yaml).**

## Why Render, not Vercel

Vercel is built for frontends and short-lived serverless functions. This is a
stateful Python process that:

- loads two XGBoost models (~7.6 MB) into memory once at startup and keeps them
  there — a per-invocation cold start would reload them on every request,
- reads files from disk at request time (`data/demo/demo_batch.csv`,
  `reports/retry_storm.json`, `data/processed/test.csv`),
- holds in-process state between requests (the audit log the decision feed reads,
  and the current batch the stats panels describe).

None of that fits a serverless model. Render (or Railway — equivalent for this
purpose) runs it as an ordinary long-lived web service, which is what it is.

Render was chosen over Railway only because its blueprint file lets the whole
configuration live in the repo and be reviewed.

## The thing that would have broken it

**A git-based deploy receives neither the trained models nor the dataset.**
Both are gitignored:

```
*.pkl                    -> models/failure_classifier.pkl, recovery_success_model.pkl
data/raw/*.csv           -> transactions.csv
data/processed/*.csv     -> train.csv, test.csv
```

Deployed as-is, the service boots but is inert: `/health` reports
`"model artifacts not found"` and every decision endpoint returns 503. Nothing
would have surfaced this until a judge opened the URL.

**What IS committed and does ship:** `data/demo/demo_batch.csv` (the pinned pitch
batch) and `reports/*.json` + `reports/*.png` — so the demo batch and the
retry-storm panel are present on the host without any extra step. Verified, not
assumed: see below.

The fix is the build command, which regenerates the missing artifacts on the
host:

```bash
pip install -r requirements.txt
python src/data_generation/generate_transactions.py
python scripts/prepare_data.py
python scripts/train.py
python scripts/retry_storm_demo.py
```

Every stage is seeded, so the host reproduces this machine's numbers exactly —
the same `+0.1458` macro-F1 lift and the same action counts. Committing the
pickles instead would bloat the repo and pin the artifacts to one exact
xgboost/scikit-learn build, which is a well-known way to get
"unpickling failed on the server".

`scripts/make_demo_batch.py` is deliberately **not** in the build command. The
demo batch is committed data; regenerating it on the host would defeat its
entire purpose.

## Portability audit

Checked before deploying, all clean:

| Risk | Status |
|---|---|
| Hardcoded `localhost` / ports | None in `src/` or `scripts/` |
| Dashboard API calls | All **relative** (`/simulate/demo`, `/stats/...`) — work on any host |
| Current-working-directory dependence | None; every path derives from `ROOT` in `src/paths.py`, which is `__file__`-relative |
| Path separators | `os.path.join` throughout |
| Binding | `--host 0.0.0.0 --port $PORT` in the start command; Render injects `$PORT` |

Verified by running the built app **from a different working directory**, bound
to `0.0.0.0` on a non-default port: dashboard, demo batch, stats, and retry-storm
report all served correctly.

## The Gemini key

Set `GEMINI_API_KEY` in **Render's dashboard** (Environment → Environment
Variables). `render.yaml` declares it with `sync: false`, which tells Render to
prompt for the value rather than read it from the repo. It is never committed —
`.gitignore` covers `.env`, and `.env.example` holds only an empty placeholder.

### What happens when the free-tier quota runs out mid-demo

It will: measured free-tier behaviour is ~11 live calls in a burst, after which
the limit persists well beyond a minute.

**Decisions are completely unaffected.** The LLM narrates a decision the
threshold policy has already made, so a quota failure costs prose, not
correctness. `/decide` still returns 200 with a complete, grounded explanation
built from the verified error-code taxonomy.

The *presentation* of that fallback needed fixing, and did not survive contact
with the question "what does a judge actually see?". The adapter marks a degraded
explanation inline, so the panel rendered:

> **live response**
> The bank returned code U67 — Debit timed out on the remitter side … *(LLM
> unavailable: 400, used template)*

— a heading saying "live response" above text saying "used template". That reads
as broken to anyone who doesn't know the backstory. The dashboard now detects the
marker, strips it, and relabels the panel:

> **grounded explanation · live model unavailable (free-tier quota) — the
> decision is unchanged**
> The bank returned code ZM — Invalid or incorrect UPI PIN entered — raised by
> the remitting bank. …

Honest rather than hidden, and it makes the architectural point instead of
undermining it. A test pins the marker format the dashboard depends on, so the
relabelling cannot silently stop working.

**If no key is set at all**, the service runs on `TemplateAdapter` from the
start: every explanation is grounded, complete, and carries no fallback notice at
all. That is a perfectly presentable demo — the live button is the only thing
that changes.

## Deploy steps

1. Push to GitHub (ensure `data/demo/` is committed — it is not covered by
   `.gitignore`, but it is newer than the last commit).
2. Render → **New +** → **Blueprint** → select the repo. It reads `render.yaml`.
3. When prompted for `GEMINI_API_KEY`, paste the key (or skip — see above).
4. First build takes a few minutes: dependency install plus model training.
5. Open the URL and run the post-deploy checks below.

## Post-deploy verification

Repeat the local checks against the public URL:

- [ ] `/health` → `status: ok`, `models_loaded: true`
- [ ] Dashboard loads on the **pinned demo batch**, headline reads **₹1,00,744**
- [ ] Action mix shows all five actions (42 / 43 / 76 / 4 / 135)
- [ ] Expand a normal row → grounded explanation naming the real error code
- [ ] Expand a **fraud** row → shows "not scored", no "Explain live" button
- [ ] Retry-storm panel renders 3,025 → 1,062 from the committed report
- [ ] "Explain live with Gemini" → either a live response or the calm degraded
      label; never a raw diagnostic

**Free-tier caveat worth knowing before a judging session:** Render's free web
services sleep after ~15 minutes idle, and the next request pays a cold start —
which here means reloading 7.6 MB of models. Open the URL a minute before
presenting so it is warm.
