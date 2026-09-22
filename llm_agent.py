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
for a restaurant search app. Read what the person wrote and capture it in your own words - \
do not force it into a fixed category if nothing fits well. Use your judgement for implied \
meaning, not just keyword matching.

CRITICAL RULE ON CONFLICTING SIGNALS: a message can mention other people (a friend, a partner, \
a family member) as BACKGROUND to why the person feels a certain way, while explicitly stating \
a DIFFERENT plan for who is actually going. When this happens, the person's explicit, direct \
statement of their plan always wins over an incidental mention - never assume someone \
mentioned in passing is joining unless the person actually says so.
Example: "I had a huge fight with my sister and I just want to eat alone tonight" -> \
the sister is NOT going. group: "solo". Do not output "with sister" or "two people" here - \
that would be ignoring their explicit statement in favor of an incidental mention.

Return ONLY valid JSON, no other text, no markdown fences, matching exactly this shape:
{
  "mood": a short free-text phrase capturing the person's mood or emotional state, in your \
own words (e.g. "wanting quiet solitude", "celebratory", "processing a breakup"), or "Any" \
if nothing is expressed,
  "occasion": a short free-text phrase for the occasion or purpose, in your own words \
(e.g. "a first date", "comfort after a bad day", "a business lunch"), or "Any" if nothing \
is expressed,
  "group": a short free-text phrase describing who is ACTUALLY going, in your own words \
(e.g. "solo", "a couple", "a big family group") - apply the CRITICAL RULE above if the \
message mentions other people ambiguously, or "Any" if nothing is expressed,
  "cuisine_pref": array of cuisine name strings explicitly or clearly implied, else [],
  "max_price": integer 1-3 (1=budget conscious, 3=no limit mentioned), default 3 if not \
mentioned - this MUST stay a plain number, it is used for real price filtering downstream,
  "open_now_only": boolean, true only if urgency is implied ("right now","tonight","currently \
open"), else false - this MUST stay true/false, it is used for a real open-status filter,
  "area": area/neighborhood mentioned as plain text, else null,
  "keywords": array of SPECIFIC requirements or amenities mentioned that don't fit the fields \
above (e.g. "big screen tv", "live sports", "outdoor seating", "pet friendly", "rooftop", \
"parking"), else [],
  "summary": ONE natural, warm sentence in plain conversational English paraphrasing your \
understanding of what they want and why - written for the person to read and confirm, not a \
list of field values. Reflect the genuine nuance of what they wrote, including their emotional \
state if relevant. This is the single most important field - it is the ONLY thing the person \
sees before deciding whether your understanding is correct.
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
# AGENTIC AI USE CASE: adaptive search-refinement + LLM-based re-ranking
# ---------------------------------------------------------------------------
REFINE_SYSTEM = """You are refining a restaurant search that returned too few usable results.
Given the original search query, the area, and why it fell short, decide ONE concrete
change to make and write a new search query text.

Return ONLY valid JSON, no other text, no markdown fences:
{"change": "one short sentence describing what you're relaxing and why", "new_query": "the new search text to run"}"""

RERANK_SYSTEM = """You are ranking real restaurant candidates for a specific person's request.
You'll get their mood/occasion/group, and a numbered list of candidates with whatever real
data is available (name, cuisine, type, description, rating). Restaurant data comes from
Google and does NOT include a "mood" field - you must judge fit yourself from the name,
description, and type, the way a person would.

Return ONLY valid JSON, no other text, no markdown fences: an array covering EVERY input
candidate, ordered best-fit first, each item:
{"index": <int, the 0-based index from the input list>, "reason": "one short specific sentence on why this does or doesn't fit well, in your own words"}
Do not invent restaurants that aren't in the input list. Include all of them, just reordered."""


def _decide_refinement(query: str, area: str, reason: str) -> dict:
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=REFINE_SYSTEM,
        messages=[{"role": "user", "content": f"Original query: {query}\nArea: {area}\nProblem: {reason}"}],
    )
    return _parse_json_response(resp.content[0].text)


def _hard_filter(restaurants: list[dict], ctx: dict) -> list[dict]:
    """Objective filters only (budget, open-now) - no tag-matching here at all,
    since live Google data has no tags to match. This is intentionally dumb."""
    out = []
    for r in restaurants:
        if ctx.get("open_now_only") and not r["open_now"]:
            continue
        if r["price_level"] > ctx.get("max_price", 3):
            continue
        out.append(r)
    return out


def llm_rerank(restaurants: list[dict], mood: str, occasion: str, group: str, keywords: list[str] = None) -> list[dict]:
    """
    The actual fix for live-data mood matching: ask an LLM to judge fit from
    real restaurant text (name/description/type), since there's no mood_tags
    field to pattern-match against on live Google results. keywords carries
    anything specific the person asked for that didn't fit mood/occasion/group
    (e.g. "big screen tv", "live sports") so the model can weigh it explicitly.
    """
    if not restaurants:
        return []

    client = _client()
    listing = "\n".join(
        f"{i}. {r['name']} \u2014 {', '.join(r['cuisine'])} \u2014 {r['restaurant_type']} \u2014 "
        f"{r['description'] or 'no description available'} \u2014 rated {r['rating']}\u2605"
        for i, r in enumerate(restaurants)
    )
    keywords_line = f"Specific requirements: {', '.join(keywords)}\n" if keywords else ""
    user_msg = f"Mood: {mood}\nOccasion: {occasion}\nGroup: {group}\n{keywords_line}\nCandidates:\n{listing}"

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=RERANK_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    ranking = _parse_json_response(resp.content[0].text)

    reordered = []
    for item in ranking:
        idx = item.get("index")
        if idx is not None and 0 <= idx < len(restaurants):
            r = dict(restaurants[idx])
            r["_reasons"] = [item.get("reason", "")]
            reordered.append(r)

    # Safety net: if the model dropped any candidates, tack them on unranked
    # rather than silently losing them.
    seen = {r["name"] for r in reordered}
    for r in restaurants:
        if r["name"] not in seen:
            r2 = dict(r)
            r2["_reasons"] = ["Not explicitly ranked by the AI \u2014 included for completeness"]
            reordered.append(r2)
    return reordered


def agentic_search(area: str, cuisine_pref: list[str], ctx: dict,
                    mood: str = "Any", occasion: str = "Any", group: str = "Any",
                    keywords: list[str] = None, max_attempts: int = 3):
    """
    Reason -> Act -> Observe loop over Google Places, followed by LLM re-ranking.

    Step A (Act):      search Google with a query that includes mood/occasion
                        AND keywords (e.g. "big screen tv", "live sports") -
                        anything specific the person asked for that doesn't
                        fit the structured fields still reaches the search.
    Step B (Observe):  apply objective hard filters only (budget, open-now).
    Step C (Reason):   if too few survive, ask the LLM how to adjust and retry.
    Step D (Reason):   once enough candidates exist, ask the LLM to judge and
                        rank them by genuine fit, keywords included - this is
                        what replaces the old empty-tag scoring for live data.

    Returns (results, trace) - trace is a list of plain-English steps for the
    "Agent Reasoning" panel in the UI.
    """
    trace = []
    local_ctx = dict(ctx)
    keywords = keywords or []
    cuisine_str = f"{' '.join(cuisine_pref)} " if cuisine_pref else ""
    mood_str = f"{mood} " if mood and mood.strip().lower() != "any" else ""
    occasion_str = f" for {occasion}" if occasion and occasion.strip().lower() != "any" else ""
    keywords_str = f" with {', '.join(keywords)}" if keywords else ""
    query = f"{mood_str}{cuisine_str}restaurants near {area}{occasion_str}{keywords_str}".strip()
    trace.append(f'Searching Google Places for: "{query}"')

    filtered = []
    for attempt in range(1, max_attempts + 1):
        try:
            restaurants = google_places.search_restaurants(query_text=query)
        except Exception as e:
            trace.append(f"Search failed: {e}")
            return [], trace

        filtered = _hard_filter(restaurants, local_ctx)
        trace.append(f"Attempt {attempt}: {len(restaurants)} raw results \u2192 {len(filtered)} passed budget/open-now filters")

        if len(filtered) >= 3 or attempt == max_attempts:
            break

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

    if not filtered:
        trace.append("No candidates left after filtering \u2014 nothing to rank")
        return [], trace

    trace.append(f"Asking the LLM to judge {len(filtered)} candidates for genuine mood/occasion fit")
    try:
        ranked = llm_rerank(filtered, mood, occasion, group, keywords)
        trace.append("LLM ranking complete")
    except Exception as e:
        trace.append(f"LLM ranking failed ({e}) \u2014 falling back to rating-sorted order")
        ranked = sorted(filtered, key=lambda r: r["rating"], reverse=True)
        for r in ranked:
            r["_reasons"] = [f"Rated {r['rating']}\u2605 (fallback sort \u2014 LLM ranking unavailable)"]

    return ranked, trace
