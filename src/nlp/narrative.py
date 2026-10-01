"""Deterministic synthetic claim narratives and lightweight text extraction."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

RISK_TERMS = ("total loss", "injury", "stolen", "fire", "multiple vehicles", "attorney")
VEHICLES = ("sedan", "suv", "pickup", "motorcycle", "vehicle")
WEATHER = ("wind", "storm", "hail", "rain", "flood")


def build_narratives(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Create reproducible synthetic narratives reflecting existing claim fields."""
    rng = np.random.default_rng(seed)
    locations = ["Austin, TX", "Denver, CO", "Miami, FL", "Phoenix, AZ", "Seattle, WA"]
    parties = ["the insured driver", "another driver", "a neighbor", "the policyholder"]
    vehicle = rng.choice(["sedan", "SUV", "pickup"], size=len(df))
    narratives: list[str] = []
    for index, row in df.reset_index(drop=True).iterrows():
        event = {
            "Collision": f"{vehicle[index]} collided with another vehicle",
            "Comprehensive": f"{vehicle[index]} sustained non-collision damage",
            "Liability": "multiple vehicles were involved in an incident",
            "Glass": f"{vehicle[index]} windshield was damaged",
            "Water": "sudden water discharge damaged the property",
            "Wind": "wind and storm conditions damaged the property",
            "Fire": "fire damaged the insured property",
            "Theft": f"{vehicle[index]} was reported stolen",
        }[row.claim_type]
        severity_text = {"Minor": "minor damage", "Moderate": "significant damage", "Major": "total loss with urgent escalation"}[row.incident_severity]
        injury = " An injury was reported and an attorney may be involved." if row.claim_type == "Liability" and row.incident_severity != "Minor" else ""
        police = " Police report is available." if row.police_report else ""
        narratives.append(
            f"On 2026-0{(index % 9) + 1}-{(index % 27) + 1:02d}, {parties[index % len(parties)]} reported that {event} in "
            f"{locations[index % len(locations)]}. Initial estimate is ${row.reported_amount:,.0f}; {severity_text}.{injury}{police}"
        )
    return pd.Series(narratives, index=df.index, name="claim_narrative")


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract deterministic, demonstrable entities without external PII services."""
    lower = text.lower()
    return {
        "vehicle_type": [term for term in VEHICLES if re.search(rf"\b{term}\b", lower)],
        "location": re.findall(r"\b[A-Z][a-z]+,\s[A-Z]{2}\b", text),
        "injury_mentions": re.findall(r"\b(?:injury|injured|attorney)\b", lower),
        "dates": re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text),
        "dollar_amounts": re.findall(r"\$[\d,]+(?:\.\d{2})?", text),
        "weather_terms": [term for term in WEATHER if re.search(rf"\b{term}\b", lower)],
        "parties": [term for term in ("insured driver", "another driver", "neighbor", "policyholder") if term in lower],
    }


def analyze_narrative(text: str) -> dict[str, Any]:
    """Return high-risk phrases and simple urgency/distress signal."""
    lower = text.lower()
    matched = [term for term in RISK_TERMS if term in lower]
    urgency_terms = ("urgent", "total loss", "attorney", "injury", "immediately")
    urgency = min(1.0, sum(term in lower for term in urgency_terms) / 3)
    sentiment = "escalated" if urgency >= 0.34 else "neutral"
    return {"entities": extract_entities(text), "risk_terms": matched, "urgency_score": round(urgency, 2), "sentiment": sentiment}
