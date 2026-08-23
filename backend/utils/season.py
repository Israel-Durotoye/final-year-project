"""
season.py — Nigerian Climate Season Detection

Determines the agricultural season from a timestamp based on Nigeria's
bimodal rainfall distribution:

    Month       Season
    ─────────   ──────────────────────
    Nov – Feb   Dry (Harmattan)
    Mar         Late Dry / Transition
    Apr – Jun   Early Rainy
    Jul – Sep   Peak Rainy
    Oct         Late Rainy / Transition

This applies broadly to the Guinea Savanna / Middle Belt zone (where
the project's nodes are deployed around Ilorin at ~8.48°N). Northern
and southern zones shift by ±1 month but the core pattern holds.
"""

from __future__ import annotations

from datetime import datetime, timezone


def get_nigerian_season(timestamp: datetime | str | None = None) -> str:
    """
    Return the current Nigerian agricultural season label.

    Parameters
    ----------
    timestamp : datetime, ISO-format string, or None
        The point in time to classify. If None, uses the current UTC time.

    Returns
    -------
    str
        One of: "Dry (Harmattan)", "Late Dry", "Early Rainy",
        "Peak Rainy", or "Late Rainy".
    """
    if timestamp is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, str):
        # Accept common ISO formats with or without timezone
        cleaned = timestamp.strip().replace("T", " ").split("+")[0].split("Z")[0]
        try:
            dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(cleaned, "%Y-%m-%d")
            except ValueError:
                dt = datetime.now(timezone.utc)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        dt = datetime.now(timezone.utc)

    month = dt.month

    if month in (11, 12, 1, 2):
        return "Dry (Harmattan)"
    elif month == 3:
        return "Late Dry"
    elif month in (4, 5, 6):
        return "Early Rainy"
    elif month in (7, 8, 9):
        return "Peak Rainy"
    elif month == 10:
        return "Late Rainy"
    else:
        return "Unknown"
