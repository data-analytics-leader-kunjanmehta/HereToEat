"""
GenAI + Agentic AI layer for HeretoEat
-----------------------------------------
Two capabilities, kept deliberately separate so each can be explained,
tested, and talked about on its own:

1. extract_context_from_text()  -- GENAI USE CASE
   Free-text understanding. Takes whatever the person actually typed
   ("somewhere quiet for a work call over coffee") and uses an LLM to
   turn it into the exact same structured shape build_context() in
   app.py already expects. This plugs into Layer 2 (Context & Intent) -
   nothing about Layer 2 itself changes, only how it gets filled in.

2. agentic_search()              -- AGENTIC AI USE CASE
   Adaptive search-refinement. A Reason -> Act -> Observe loop: search,
   look at whether the results are actually usable, and if not, decide
   HOW to adjust and try again - autonomously, up to a few attempts -
   instead of just returning "no matches." This plugs into Layer 3
   (Reasoning), sitting in front of the existing rank_restaurants().

Both call the Claude API (Anthropic). Needs ANTHROPIC_API_KEY set
(same pattern as GOOGLE_PLACES_API_KEY - via .env locally, or a Colab
Secret when running in Colab). Uses Claude Haiku 4.5 - fast and cheap,
appropriate for short structured-extraction and decision-making calls
like these rather than long-form generation.
"""

import json
import os

import anthropic

import google_places

MODEL = "claude-haiku-4-5-20251001"


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("No ANTHROPIC_API_KEY found (set it in .env or as a Colab Secret)")
    return anthropic.Anthropic(api_key=api_key)


def _parse_json_response(raw: str) -> dict:
    """Claude sometimes wraps JSON in ```...``` fences even when asked not to - strip if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# GENAI USE CASE: free-text query understanding
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """You convert a free-text restaurant request into structured JSON \
for a restaurant search app. Read what the person wrote and infer the closest matching \
values below - use your judgement for implied meaning, not just keyword matching.

Return ONLY valid JSON, no other text, no markdown fences, matching exactly this shape:
{
  "mood": one of ["Any","Romantic","Casual","Celebratory","Cozy / Quiet","Lively / Energetic"],
  "occasion": one of ["Any","Casual meal","Date","Birthday / Celebration","Family gathering","Business meal","Anniversary"],
  "group": one of ["Any","Solo","Couple / Date","Friends","Family","Large group"],
  "cuisine_pref": array of cuisine name strings explicitly or clearly implied, else [],
  "max_price": integer 1-3 (1=budget conscious, 3=no limit mentioned), default 3 if not mentioned,
  "open_now_only": boolean, true only if urgency is implied ("right now","tonight","currently open"), else false,
  "area": area/neighborhood mentioned as plain text, else null
}"""


def extract_context_from_text(free_text: str) -> dict:
    """
    GenAI step: natural language -> structured context.
    Raises on API failure or unparseable response - caller should catch
    and fall back to the manual dropdowns rather than crash the app.
    """
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": free_text}],
    )
    return _parse_json_response(resp.content[0].text)


# ---------------------------------------------------------------------------
# AGENTIC AI USE CASE: adaptive search-refinement
# ---------------------------------------------------------------------------
REFINE_SYSTEM = """You are refining a restaurant search that returned too few usable results.
Given the original search query, the area, and why it fell short, decide ONE concrete
change to make and write a new search query text.

Return ONLY valid JSON, no other text, no markdown fences:
{"change": "one short sentence describing what you're relaxing and why", "new_query": "the new search text to run"}"""


def _decide_refinement(query: str, area: str, reason: str) -> dict:
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=REFINE_SYSTEM,
        messages=[{"role": "user", "content": f"Original query: {query}\nArea: {area}\nProblem: {reason}"}],
    )
    return _parse_json_response(resp.content[0].text)


def agentic_search(area: str, cuisine_pref: list[str], ctx: dict, rank_fn, max_attempts: int = 3):
    """
    Reason -> Act -> Observe loop over Google Places + the existing ranking layer.

    area, cuisine_pref: used to build the first search query.
    ctx: the context dict from build_context() - scored against by rank_fn.
    rank_fn: pass app.py's rank_restaurants directly - keeps Layer 3's scoring
             logic completely untouched; the agent only decides WHAT to search,
             never how results get scored once they're in hand.
    Returns (results, trace) - trace is a list of plain-English steps for the
    "Agent Reasoning" panel in the UI.
    """
    trace = []
    local_ctx = dict(ctx)
    cuisine_str = f"{' '.join(cuisine_pref)} " if cuisine_pref else ""
    query = f"{cuisine_str}restaurants near {area}".strip()
    trace.append(f'Searching Google Places for: "{query}"')

    results = []
    for attempt in range(1, max_attempts + 1):
        try:
            restaurants = google_places.search_restaurants(query_text=query)
        except Exception as e:
            trace.append(f"Search failed: {e}")
            return [], trace

        results = rank_fn(restaurants, local_ctx)
        trace.append(f"Attempt {attempt}: {len(restaurants)} raw results \u2192 {len(results)} passed filters")

        if len(results) >= 3:
            trace.append("Enough good matches found \u2014 stopping here")
            return results, trace

        if attempt == max_attempts:
            trace.append("Reached max attempts \u2014 returning best available")
            return results, trace

        # Not enough usable results yet - let the agent decide how to adjust.
        reason = (
            "Too few results passed budget/open-now filters"
            if local_ctx.get("open_now_only") or local_ctx.get("max_price", 3) < 3
            else "Too few results matched \u2014 query may be too narrow"
        )
        try:
            refinement = _decide_refinement(query, area, reason)
            trace.append(f"Agent decision: {refinement['change']}")
            query = refinement["new_query"]
        except Exception as e:
            trace.append(f"Refinement call failed ({e}) \u2014 relaxing open-now filter as a fallback")
            local_ctx["open_now_only"] = False

    return results, trace
