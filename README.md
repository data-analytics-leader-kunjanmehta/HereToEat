# HeretoEat — Local App (Live Google Data)

Runs on your laptop. Pulls real restaurant data from Google Places API
(New) once your key is set up — falls back automatically to a small local
sample dataset if no key is found, so it never just breaks.

## 1. Google Cloud setup (one-time, ~10 minutes)

Follow these in your browser (console.cloud.google.com):

1. Create/open a Google Cloud project.
2. Enable billing (required by Google even for free-tier use — nothing is
   charged just from this step).
3. Enable **Places API (New)** under APIs & Services → Library.
4. Create an API key under APIs & Services → Credentials, then click
   **Restrict key** and limit it to Places API (New) only.
5. Set a **hard quota cap** under Places API (New) → Quotas & System Limits
   (e.g. 100 requests/day). This is what makes it genuinely zero-cost —
   once hit, Google errors out instead of billing you.
6. (Optional) Set a ₹1 budget alert under Billing → Budgets & Alerts as a
   second tripwire.

## 2. Add your key

```bash
cd rf_app
cp .env.example .env
```

Open `.env` and paste your real key in place of `your_key_here`. `.env` is
already in `.gitignore` — it will never get committed or shared.

## 3. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. If `.env` has a valid key, you'll see
"Live mode" at the top and results come from real, current Google data.
If something's off with the key, the app tells you and automatically falls
back to the local sample dataset rather than crashing.

## Important: this is your first real test of the live call

This app was built and unit-tested against a *simulated* Google response —
the sandbox used to write it can't reach Google's servers directly. Your
laptop can. The first time you click "Find restaurants" in live mode is
the actual first real test of the API call. If it errors, the message
shown will tell you what's wrong (bad key, API not enabled, quota hit,
etc.) — paste that error back and it's a quick fix.

## What's inside

```
rf_app/
├─ app.py                       # UI + reasoning, all in one process
├─ google_places.py             # Layer 1 (Data) — live Google Places call
├─ data/sample_restaurants.json # local fallback dataset
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```

`app.py` keeps the same four layers as before — only Layer 1 (Data) changed
to call `google_places.search_restaurants()` instead of reading the local
JSON file. Layers 2–4 (context, reasoning, presentation) are untouched.

## Next: sharing it with others

Not today's problem — this still runs locally, on your machine only.
Once it's working the way you want, the next step is deploying the same
app to free hosting (e.g. Streamlit Community Cloud) to get a shareable
link, with your key stored as a hosting "secret" instead of a local `.env`.
