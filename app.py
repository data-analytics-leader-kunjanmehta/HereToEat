"""
HeretoEat - AI-only flow
-----------------------------------------------
Single input path: describe what you want in plain text, an LLM extracts
context, an agent searches Google Places and refines/re-ranks the results.
The earlier dropdown-based manual flow (rule-based scoring against a local
sample dataset) has been removed - this file now has exactly one path.

Layers, still kept separate on purpose:
  Layer 1 (Data):             google_places.py - live Google Places call
  Layer 2 (Context & Intent): build_context() - just the two fields that
                              still need to be structured (budget, open-now)
  Layer 3 (Reasoning):        llm_agent.py - extraction, adaptive search
                              refinement, and LLM-based re-ranking
  Layer 4 (Presentation):     render_result_card() / render_agent_trace()

Run it with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

import llm_agent

load_dotenv()  # reads .env for GOOGLE_PLACES_API_KEY and ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# LAYER 2 - CONTEXT & INTENT LAYER
# Only max_price and open_now_only stay structured here - they're used for
# real numeric/boolean filtering downstream. mood/occasion/group are now
# free text and get passed straight through to llm_agent, not through this.
# ---------------------------------------------------------------------------
def build_context(max_price: int, open_now_only: bool) -> dict:
    return {"max_price": max_price, "open_now_only": open_now_only}


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


def main():
    st.set_page_config(page_title="HeretoEat", page_icon="\U0001F37D\uFE0F", layout="centered")
    st.title("\U0001F37D\uFE0F HeretoEat")

    google_key_present = bool(os.environ.get("GOOGLE_PLACES_API_KEY"))
    llm_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if google_key_present and llm_key_present:
        st.caption("Live mode \u2014 Google Places + Claude reasoning, both active.")
    else:
        missing = []
        if not google_key_present:
            missing.append("GOOGLE_PLACES_API_KEY")
        if not llm_key_present:
            missing.append("ANTHROPIC_API_KEY")
        st.error(
            f"Missing: {', '.join(missing)} in .env \u2014 the app can't run without both keys "
            f"(see README).",
            icon="\u26a0\ufe0f",
        )

    st.markdown("##### Describe what you're looking for")
    free_text = st.text_area(
        "Free text",
        placeholder="e.g. \"Somewhere quiet in Indiranagar for a work call over coffee\"",
        label_visibility="collapsed",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        max_price = st.select_slider("Max budget", options=[1, 2, 3], value=3,
                                      format_func=lambda x: "\u20b9" * x)
    with col2:
        open_now_only = st.checkbox("Open right now only", value=False)

    ask_ai_clicked = st.button(
        "\U0001F916 Find restaurants",
        type="primary",
        disabled=not (google_key_present and llm_key_present),
    )

    if ask_ai_clicked and free_text.strip():
        with st.spinner("Reading your request..."):
            try:
                parsed = llm_agent.extract_context_from_text(free_text)
            except Exception as e:
                st.error(f"Couldn't understand that with AI: {e}")
                parsed = None

        if parsed:
            with st.expander("What the AI understood from your text", expanded=True):
                st.json(parsed)

            area = parsed.get("area") or "Bangalore"
            cuisine_pref = parsed.get("cuisine_pref") or []
            # Free-text max_price/open_now_only from the UI controls above take
            # precedence if the person set them; otherwise fall back to whatever
            # the AI inferred from the text itself.
            ctx = build_context(
                max_price if max_price != 3 else parsed.get("max_price", 3),
                open_now_only or parsed.get("open_now_only", False),
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
                st.warning("The agent couldn't find a good match even after refining \u2014 try rephrasing.")
            else:
                st.markdown(f"##### Top {min(5, len(results))} matches")
                for i, r in enumerate(results[:5], start=1):
                    render_result_card(r, i)
    elif ask_ai_clicked:
        st.warning("Type something in the box above first.")


if __name__ == "__main__":
    main()
