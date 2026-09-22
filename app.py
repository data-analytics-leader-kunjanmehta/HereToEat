"""
HeretoEat - Phase 1 (Local Prototype)
-----------------------------------------------
Runs entirely on your laptop. No API key, no internet dependency, no cost.
Data source: data/sample_restaurants.json (a small, manually curated list).

This file intentionally keeps all four architecture layers visible and
separated as plain Python functions, so swapping the Data Layer for a live
Google Places API call later touches ONLY load_restaurants() below -
nothing else in this file needs to change.

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import google_places
import llm_agent

load_dotenv()  # reads .env for GOOGLE_PLACES_API_KEY and ANTHROPIC_API_KEY

DATA_PATH = Path(__file__).parent / "data" / "sample_restaurants.json"

MOOD_OPTIONS = ["Any", "Romantic", "Casual", "Celebratory", "Cozy / Quiet", "Lively / Energetic"]
OCCASION_OPTIONS = ["Any", "Casual meal", "Date", "Birthday / Celebration", "Family gathering", "Business meal", "Anniversary"]
GROUP_OPTIONS = ["Any", "Solo", "Couple / Date", "Friends", "Family", "Large group"]

MOOD_TO_TAGS = {
    "Romantic": ["romantic", "elegant", "quiet", "cozy"],
    "Casual": ["casual", "comfort", "quick"],
    "Celebratory": ["celebratory", "lively", "playful"],
    "Cozy / Quiet": ["cozy", "quiet", "nostalgic"],
    "Lively / Energetic": ["lively", "playful", "celebratory"],
}

OCCASION_TO_SUITABLE = {
    "Casual meal": ["casual", "solo"],
    "Date": ["date"],
    "Birthday / Celebration": ["celebration", "special_event"],
    "Family gathering": ["family"],
    "Business meal": ["business"],
    "Anniversary": ["anniversary", "date"],
}

GROUP_TO_SUITABLE = {
    "Solo": ["solo", "casual"],
    "Couple / Date": ["date", "anniversary"],
    "Friends": ["friends", "large_group"],
    "Family": ["family"],
    "Large group": ["large_group", "friends"],
}


# ---------------------------------------------------------------------------
# LAYER 1 - DATA LAYER
# Today: reads the local JSON file below.
# Later: replace the body of this one function with a Google Places API call
# that returns the same shape (list of dicts) - nothing downstream changes.
# ---------------------------------------------------------------------------
def load_restaurants() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_restaurants_live(area: str, cuisine_pref: list[str]) -> list[dict]:
    cuisine_str = f"{' '.join(cuisine_pref)} " if cuisine_pref else ""
    query = f"{cuisine_str}restaurants near {area}".strip()
    return google_places.search_restaurants(query_text=query)


# ---------------------------------------------------------------------------
# LAYER 2 - CONTEXT & INTENT LAYER
# Turns the person's picks in the UI into a plain context dict the
# reasoning layer can score against.
# ---------------------------------------------------------------------------
def build_context(mood: str, occasion: str, group: str, cuisine_pref: list[str],
                   max_price: int, open_now_only: bool) -> dict:
    return {
        "mood_tags": MOOD_TO_TAGS.get(mood, []),
        "suitable_for": OCCASION_TO_SUITABLE.get(occasion, []) + GROUP_TO_SUITABLE.get(group, []),
        "cuisine_pref": cuisine_pref,
        "max_price": max_price,
        "open_now_only": open_now_only,
    }


# ---------------------------------------------------------------------------
# LAYER 3 - AGENTIC REASONING LAYER
# MVP version: transparent rule-based scoring, so every recommendation can
# state exactly *why* it was ranked where it was. This is the seam where
# LLM-based reasoning (Claude API) can be layered in later without touching
# Layers 1, 2, or 4.
# ---------------------------------------------------------------------------
def score_restaurant(r: dict, ctx: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []

    # Hard filters first
    if ctx["open_now_only"] and not r["open_now"]:
        return -1, ["Currently closed"]
    if r["price_level"] > ctx["max_price"]:
        return -1, ["Above selected budget"]

    # Soft scoring
    if ctx["cuisine_pref"]:
        cuisine_overlap = set(c.lower() for c in r["cuisine"]) & set(c.lower() for c in ctx["cuisine_pref"])
        if cuisine_overlap:
            score += 2
            reasons.append(f"Matches cuisine: {', '.join(sorted(cuisine_overlap))}")
    mood_overlap = set(r["mood_tags"]) & set(ctx["mood_tags"])
    if mood_overlap:
        score += 2 * len(mood_overlap)
        reasons.append(f"Matches mood: {', '.join(sorted(mood_overlap))}")

    suitable_overlap = set(r["suitable_for"]) & set(ctx["suitable_for"])
    if suitable_overlap:
        score += 3 * len(suitable_overlap)
        reasons.append(f"Good fit for: {', '.join(sorted(suitable_overlap))}")

    score += r["rating"]  # baseline quality signal
    reasons.append(f"Rated {r['rating']}\u2605 ({r['reviews']:,} reviews)")

    if r["reviews"] > 5000:
        score += 0.5  # confidence bump for well-reviewed places

    return score, reasons


def rank_restaurants(restaurants: list[dict], ctx: dict) -> list[dict]:
    ranked = []
    for r in restaurants:
        score, reasons = score_restaurant(r, ctx)
        if score < 0:
            continue
        ranked.append({**r, "_score": score, "_reasons": reasons})
    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# LAYER 4 - PRESENTATION LAYER
# ---------------------------------------------------------------------------
def render_result_card(r: dict, rank: int):
    price_label = "\u20b9" * r["price_level"]
    status = "\U0001F7E2 Open" if r["open_now"] else "\U0001F534 Closed"
    with st.container(border=True):
        st.markdown(f"#### {rank}. {r['name']}")
        st.caption(f"{r['area']}  \u00b7  {', '.join(r['cuisine'])}  \u00b7  {price_label}  \u00b7  {status}")
        st.write(f"\u2b50 **{r['rating']}** ({r['reviews']:,} reviews)  \u00b7  {r['restaurant_type']}")
        st.write(r["description"])
        with st.expander("Why this was recommended"):
            for reason in r["_reasons"]:
                st.markdown(f"- {reason}")


def render_agent_trace(trace: list[str]):
    with st.container(border=True):
        st.markdown("**\U0001F916 Agent Reasoning**")
        for i, step in enumerate(trace, start=1):
            st.markdown(f"{i}. {step}")


COMMON_CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Continental",
                    "Cafe", "Thai", "Japanese", "Andhra", "Biryani", "Desserts", "Fast Food"]


def main():
    st.set_page_config(page_title="HeretoEat", page_icon="\U0001F37D\uFE0F", layout="centered")
    st.title("\U0001F37D\uFE0F HeretoEat")

    api_key_present = bool(os.environ.get("GOOGLE_PLACES_API_KEY"))
    llm_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if api_key_present:
        st.caption("Live mode \u2014 pulling real results from Google Places.")
        data_mode = "live"
    else:
        st.warning(
            "No GOOGLE_PLACES_API_KEY found in .env \u2014 falling back to the local sample "
            "dataset for now. Add your key to go live (see README).",
            icon="\u26a0\ufe0f",
        )
        data_mode = "local"

    # -----------------------------------------------------------------
    # GenAI + Agentic entry point: describe it in your own words
    # -----------------------------------------------------------------
    st.markdown("##### Or just describe what you're looking for")
    free_text = st.text_area(
        "Free text",
        placeholder="e.g. \"Somewhere quiet in Indiranagar for a work call over coffee\"",
        label_visibility="collapsed",
    )
    ask_ai_clicked = st.button("\U0001F916 Let AI figure it out", type="primary", disabled=not llm_key_present)
    if not llm_key_present:
        st.caption("Needs ANTHROPIC_API_KEY in .env to enable this \u2014 the dropdowns below still work without it.")

    if ask_ai_clicked and free_text.strip():
        with st.spinner("Reading your request..."):
            try:
                parsed = llm_agent.extract_context_from_text(free_text)
            except Exception as e:
                st.error(f"Couldn't understand that with AI: {e}\n\nTry the dropdowns below instead.")
                parsed = None

        if parsed:
            with st.expander("What the AI understood from your text", expanded=True):
                st.json(parsed)

            area = parsed.get("area") or "Bangalore"
            cuisine_pref = parsed.get("cuisine_pref") or []
            ctx = build_context(
                parsed.get("mood", "Any"), parsed.get("occasion", "Any"), parsed.get("group", "Any"),
                cuisine_pref, parsed.get("max_price", 3), parsed.get("open_now_only", False),
            )

            with st.spinner("Searching and refining..."):
                results, trace = llm_agent.agentic_search(
                    area, cuisine_pref, ctx,
                    mood=parsed.get("mood", "Any"),
                    occasion=parsed.get("occasion", "Any"),
                    group=parsed.get("group", "Any"),
                    keywords=parsed.get("keywords", []),
                )

            st.markdown("---")
            render_agent_trace(trace)
            st.markdown("---")
            if not results:
                st.warning("The agent couldn't find a good match even after refining \u2014 try the dropdowns below.")
            else:
                st.markdown(f"##### Top {min(5, len(results))} matches")
                for i, r in enumerate(results[:5], start=1):
                    render_result_card(r, i)
            return  # don't also render the manual flow below on this run

    st.markdown("---")
    st.markdown("##### Or use the filters directly")

    if data_mode == "live":
        area = st.text_input("Area / neighborhood", value="Bangalore")
    col1, col2, col3 = st.columns(3)
    with col1:
        mood = st.selectbox("Mood", MOOD_OPTIONS)
    with col2:
        occasion = st.selectbox("Occasion", OCCASION_OPTIONS)
    with col3:
        group = st.selectbox("Who's going", GROUP_OPTIONS)

    cuisine_choices = COMMON_CUISINES if data_mode == "live" else sorted(
        {c for r in load_restaurants() for c in r["cuisine"]}
    )
    cuisine_pref = st.multiselect("Cuisine preference (optional)", cuisine_choices)
    max_price = st.select_slider("Max budget", options=[1, 2, 3], value=3,
                                  format_func=lambda x: "\u20b9" * x)
    open_now_only = st.checkbox("Only show places open right now", value=False)

    if st.button("Find restaurants", type="primary"):
        with st.spinner("Searching..."):
            try:
                if data_mode == "live":
                    restaurants = load_restaurants_live(area, cuisine_pref)
                else:
                    restaurants = load_restaurants()
            except Exception as e:
                st.error(f"Couldn't reach Google Places API: {e}\n\nFalling back to local sample data.")
                restaurants = load_restaurants()

        ctx = build_context(mood, occasion, group, cuisine_pref, max_price, open_now_only)
        results = rank_restaurants(restaurants, ctx)

        st.markdown("---")
        if not results:
            st.warning("No matches with these filters \u2014 try loosening budget, cuisine, or open-now.")
        else:
            st.markdown(f"##### Top {min(5, len(results))} matches")
            for i, r in enumerate(results[:5], start=1):
                render_result_card(r, i)


if __name__ == "__main__":
    main()
