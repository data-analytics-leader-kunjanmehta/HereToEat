"""
LAYER 1 - DATA LAYER (live version)
------------------------------------
Calls Google Places API (New) - Text Search and normalizes the response
into the exact same shape app.py already expects:

    {name, area, cuisine, rating, reviews, price_level, restaurant_type,
     description, dine_in, takeaway, open_now, mood_tags, suitable_for}

mood_tags / suitable_for don't exist in Google's data (that's Layer 2's job,
built from the user's own picks in the UI) - they come back as empty lists
here and get combined with context at scoring time.

Docs this follows: https://developers.google.com/maps/documentation/places/web-service/text-search
"""

import os
import requests

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.currentOpeningHours.openNow",
    "places.types",
    "places.primaryTypeDisplayName",
    "places.editorialSummary",
    "places.dineIn",
    "places.takeout",
    "places.businessStatus",
])

# Rough Bangalore center - used as a location bias only (biases results
# toward this area, doesn't restrict them). Adjust if you're elsewhere.
DEFAULT_LAT = 12.9716
DEFAULT_LNG = 77.5946
DEFAULT_RADIUS_M = 15000

PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE": 1,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 3,
}

# Google 'types' entries that look like cuisine (ends in _restaurant) get
# turned into readable cuisine tags, e.g. "italian_restaurant" -> "Italian"
def _extract_cuisine(types: list[str], fallback: str) -> list[str]:
    cuisines = []
    for t in types or []:
        if t.endswith("_restaurant") and t != "restaurant":
            cuisines.append(t.replace("_restaurant", "").replace("_", " ").title())
    if not cuisines and fallback:
        cuisines = [fallback]
    return cuisines or ["Restaurant"]


def _normalize(place: dict) -> dict:
    display_name = place.get("displayName", {}).get("text", "Unknown")
    primary_type = place.get("primaryTypeDisplayName", {}).get("text", "Restaurant")
    return {
        "name": display_name,
        "area": place.get("formattedAddress", ""),
        "cuisine": _extract_cuisine(place.get("types"), primary_type),
        "rating": place.get("rating", 0.0),
        "reviews": place.get("userRatingCount", 0),
        "price_level": PRICE_LEVEL_MAP.get(place.get("priceLevel"), 2),
        "restaurant_type": primary_type,
        "description": place.get("editorialSummary", {}).get("text", ""),
        "dine_in": place.get("dineIn", True),
        "takeaway": place.get("takeout", False),
        "open_now": place.get("currentOpeningHours", {}).get("openNow", True),
        "mood_tags": [],       # Layer 2 territory - filled in from user context, not Google
        "suitable_for": [],    # same
    }


def search_restaurants(query_text: str, api_key: str | None = None,
                        lat: float = DEFAULT_LAT, lng: float = DEFAULT_LNG,
                        radius_m: float = DEFAULT_RADIUS_M, max_results: int = 15) -> list[dict]:
    """
    query_text: free-text search, e.g. "restaurants near Indiranagar, Bangalore"
    Returns a list of dicts in the same shape as data/sample_restaurants.json.
    Raises RuntimeError with a readable message on any API failure.
    """
    api_key = api_key or os.environ.get("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("No Google Places API key found (set GOOGLE_PLACES_API_KEY in .env)")

    body = {
        "textQuery": query_text,
        "maxResultCount": max_results,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_m,
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    resp = requests.post(PLACES_URL, json=body, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Google Places API error {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    places = data.get("places", [])
    return [_normalize(p) for p in places]
