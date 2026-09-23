# HeretoEat — Local App (Live Google Data + GenAI + Agentic AI)

Runs on your laptop or Colab. Pulls real restaurant data from Google Places
API, understands free-text requests via Claude (GenAI), and adaptively
refines its own search when results fall short (Agentic AI). Falls back
automatically to a small local sample dataset if the Google key is
missing, so it never just breaks.

## What's new: GenAI + Agentic AI

- **GenAI — free-text understanding** (`llm_agent.extract_context_from_text`):
  type what you want in plain language instead of using dropdowns; Claude
  converts it into the same structured context the app already used.
- **Agentic AI — adaptive search-refinement** (`llm_agent.agentic_search`):
  a Reason → Act → Observe loop. If the first search comes back with too
  few usable results, the agent decides how to adjust (broaden cuisine,
  drop a filter, rephrase) and retries — up to 3 attempts — instead of
  just returning "no matches." Every step is shown in an "Agent Reasoning"
  panel in the UI.
- **Real geocoding + midpoint search** (`google_places.geocode_place_name`,
  `llm_agent.geographic_midpoint`, `resolve_search_center`): if a request
  mentions two different areas (e.g. two people's home neighborhoods) and
  wants something equidistant, both are geocoded and the actual geographic
  midpoint is computed and used as the real search center — not just
  whichever area happened to be mentioned first in the sentence.
- **Exclusion handling** (`_hard_filter`'s `exclude_places`): if the request
  names a specific place to avoid (e.g. "we've been to X plenty, suggest
  something similar but different"), that place is filtered out of results.

Both features are the ONLY flow now — the earlier dropdown-based rule-based
flow has been removed entirely. This is a single-path, AI-only app.

## 1. Google Cloud setup (one-time)

See the earlier setup notes — Places API (New) enabled, key restricted,
quota capped. Unchanged from before.

## 2. Anthropic (Claude) API setup (new)

1. Go to console.anthropic.com and sign in / create an account.
2. Go to Settings → API Keys → Create Key. Copy it.
3. This is a separate billing relationship from Google — usage for this
   app (short extraction + refinement calls) is very cheap, but there's
   no free-tier quota cap like Google's, so keep an eye on usage under
   Settings → Usage if you're cost-conscious.

## 3. Add your keys

```bash
cd heretoeat_app
cp .env.example .env
```

Open `.env` and paste both real keys in. `.env` is already in
`.gitignore` — it will never get committed or shared.

**Running in Colab instead?** Don't put the key in a file — use a Colab
Secret (🔑 icon in the sidebar) named `ANTHROPIC_API_KEY`, same pattern as
`GOOGLE_PLACES_API_KEY`, then write both into `.env` inside the session
exactly like before.

## 4. Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Type a request in plain language, click
"Understand my request" — the AI writes back a single plain-English
paraphrase of what it thinks you want (not a form, no fixed categories for
mood/occasion/group). Click "Yes, search for that" if it's right, or "No,
let me rephrase" to edit your original sentence directly. Both API keys
are required.

## What's inside

```
heretoeat_app/
├─ app.py                       # UI + both flows, all in one process
├─ google_places.py             # Layer 1 (Data) — live Google Places call
├─ llm_agent.py                 # GenAI (extraction) + Agentic AI (refinement)
├─ data/sample_restaurants.json # local fallback dataset
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ README.md
```

`llm_agent.py` plugs into the *existing* Layer 2 (Context) and Layer 3
(Reasoning) seams — `build_context()` and `rank_restaurants()` in `app.py`
are completely unchanged. The GenAI step only changes how the context gets
filled in; the agentic loop only changes what gets searched for. Layer 4
(Presentation) gained one new panel (Agent Reasoning) but nothing else
about it changed either.

## Important: first real test, same as before

Both `llm_agent.py` functions were built and unit-tested against
*simulated* Claude responses — this sandbox can't reach `api.anthropic.com`
directly. Your first real click of "Understand my request" is the actual
first live test. If it errors, the message shown will say what's wrong —
paste it back for a quick fix.
