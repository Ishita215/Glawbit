"""
core/analytics/market.py
=========================
Market intelligence derived from Analytics Jobs.csv.

Key design decisions
--------------------
1. Skill prevalence is calculated per-skill, not per-capability.
   This avoids the apples-to-oranges problem of comparing JD capability scores
   directly with market capability percentages.

2. A "usable" job is one that has at least key_skills OR job_description.

3. Demand tiers are calculated dynamically from the actual distribution
   (25th / 75th percentile), NOT hardcoded thresholds.

4. Market profile (cached) is computed once per process.

All public functions return plain dicts/lists (no DataFrames) so they are
safe to serialise to JSON.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.data.loaders import load_analytics_jobs
from core.nlp.taxonomy import SKILL_TAXONOMY, CAPABILITY_DIMENSIONS

logger = logging.getLogger("dsxray.analytics.market")


# ── Internal helpers ──────────────────────────────────────────

def _get_usable_jobs() -> pd.DataFrame:
    """
    Return rows from Analytics Jobs that have at least some text
    in key_skills OR job_description (case-normalised).
    """
    df = load_analytics_jobs()
    if df.empty:
        return df

    has_text = (
        (df["key_skills"].str.len() > 2) |
        (df["job_description"].str.len() > 2)
    )
    usable = df[has_text].copy()
    logger.debug("Usable Analytics Jobs rows: %d / %d", len(usable), len(df))
    return usable


def _combined_text_series(df: pd.DataFrame) -> pd.Series:
    """Concatenate key_skills + job_description into a single lowercase series."""
    return (
        df["key_skills"].fillna("").astype(str)
        + " "
        + df["job_description"].fillna("").astype(str)
    ).str.lower()


def _count_jobs_with_skill(
    keyword: str,
    text_series: pd.Series,
) -> int:
    """Count rows in text_series that contain keyword as a whole word."""
    try:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        return int(text_series.str.contains(pattern, regex=True, na=False).sum())
    except re.error:
        return 0


# ── Core market statistics ────────────────────────────────────

_market_stats_cache: Optional[dict] = None
_capability_profile_cache: Optional[dict[str, dict]] = None


def compute_market_statistics(force_reload: bool = False) -> dict:
    """
    Compute per-skill prevalence across ALL Analytics Jobs (not just DS).
    Returns a dict keyed by normalised skill name with prevalence info.

    Result is cached after first computation.
    """
    global _market_stats_cache
    if _market_stats_cache is not None and not force_reload:
        return _market_stats_cache

    # Try loading precomputed file from ml/
    stats_file = Path(__file__).resolve().parent.parent.parent / "ml" / "market_stats.json"
    if stats_file.exists() and not force_reload:
        try:
            import json
            with open(stats_file, "r", encoding="utf-8") as fh:
                _market_stats_cache = json.load(fh)
            logger.info("Loaded precomputed market statistics from %s", stats_file)
            return _market_stats_cache
        except Exception as exc:
            logger.warning("Could not load precomputed stats: %s", exc)

    df = _get_usable_jobs()
    if df.empty:
        logger.error("No usable Analytics Jobs rows for market statistics.")
        return {}

    total_jobs = len(df)
    text_series = _combined_text_series(df)

    logger.info("Computing market statistics across %d jobs…", total_jobs)

    stats: dict[str, dict] = {}
    seen_normalized: set[str] = set()

    for entry in SKILL_TAXONOMY:
        norm = entry["normalized"]
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)

        # Count jobs matching any keyword for this skill in a single regex
        kws = [re.escape(k.lower()) for k in entry["keywords"]]
        matched = 0
        if kws:
            pattern = r"\b(?:" + "|".join(sorted(kws, key=len, reverse=True)) + r")\b"
            try:
                matched = int(text_series.str.contains(pattern, regex=True, na=False).sum())
            except re.error:
                pass

        prevalence_pct = round(matched / total_jobs * 100, 2) if total_jobs > 0 else 0.0

        stats[norm] = {
            "normalized_skill": norm,
            "capability":       entry["capability"],
            "job_count":        matched,
            "total_jobs":       total_jobs,
            "prevalence_pct":   prevalence_pct,
        }

    # ── Compute demand tiers from distribution ────────────────
    prevalence_values = np.array([v["prevalence_pct"] for v in stats.values()])
    p25 = float(np.percentile(prevalence_values, 25))
    p75 = float(np.percentile(prevalence_values, 75))

    for skill_data in stats.values():
        pct = skill_data["prevalence_pct"]
        if pct >= p75:
            tier = "High"
        elif pct >= p25:
            tier = "Medium"
        else:
            tier = "Low"
        skill_data["demand_tier"] = tier
        skill_data["p25_threshold"] = round(p25, 2)
        skill_data["p75_threshold"] = round(p75, 2)

    stats["_meta"] = {
        "total_usable_jobs": total_jobs,
        "p25_threshold": round(p25, 2),
        "p75_threshold": round(p75, 2),
        "skills_tracked": len(stats) - 1,
        "source": "Analytics Jobs.csv",
    }

    logger.info(
        "Market stats computed: %d skills, p25=%.1f%%, p75=%.1f%%",
        len(stats) - 1, p25, p75,
    )
    _market_stats_cache = stats

    try:
        import json
        stats_file.parent.mkdir(exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not persist market stats cache: %s", exc)

    return stats


def get_skill_market_data(normalized_skill: str) -> Optional[dict]:
    """Return market data for a single normalised skill name."""
    stats = compute_market_statistics()
    return stats.get(normalized_skill)


def get_top_skills(n: int = 20) -> list[dict]:
    """Return the n most prevalent skills in the market, sorted descending."""
    stats = compute_market_statistics()
    skills = [v for k, v in stats.items() if k != "_meta"]
    return sorted(skills, key=lambda x: x["prevalence_pct"], reverse=True)[:n]


def get_skill_prevalence_for_extracted(extracted_skills: list[dict]) -> list[dict]:
    """
    For a list of extracted skill dicts (from skill_extractor),
    attach market prevalence data to each.
    Returns a list of enriched dicts.
    """
    stats = compute_market_statistics()
    enriched = []
    for skill in extracted_skills:
        norm = skill["normalized_skill"]
        market = stats.get(norm, {})
        enriched.append({
            **skill,
            "job_count":    market.get("job_count", 0),
            "total_jobs":   market.get("total_jobs", 0),
            "prevalence_pct": market.get("prevalence_pct", 0.0),
            "demand_tier":  market.get("demand_tier", "Unknown"),
        })
    return enriched


def market_health_summary() -> dict:
    """Summary dict for the Command Centre card."""
    stats = compute_market_statistics()
    meta = stats.get("_meta", {})
    return {
        "total_jobs_in_dataset": meta.get("total_usable_jobs", 0),
        "skills_tracked":        meta.get("skills_tracked", 0),
        "source":                meta.get("source", "Analytics Jobs.csv"),
        "p25_threshold_pct":     meta.get("p25_threshold", 0),
        "p75_threshold_pct":     meta.get("p75_threshold", 0),
    }


# ── Capability-level market profile ───────────────────────────
# (for charts/visualisation purposes only – not for direct comparison)

def capability_market_profile(force_reload: bool = False) -> dict[str, dict]:
    """
    For each capability dimension: what fraction of jobs mention ANY
    skill in that dimension.
    Use for visualisation only (not for JD vs market comparison).
    Cached in-memory and to disk for fast repeated calls.
    """
    global _capability_profile_cache
    if _capability_profile_cache is not None and not force_reload:
        return _capability_profile_cache

    profile_file = Path(__file__).resolve().parent.parent.parent / "ml" / "capability_profile.json"
    if profile_file.exists() and not force_reload:
        try:
            import json
            with open(profile_file, "r", encoding="utf-8") as fh:
                _capability_profile_cache = json.load(fh)
            return _capability_profile_cache
        except Exception:
            pass

    df = _get_usable_jobs()
    if df.empty:
        return {cap: {"pct": 0, "jobs": 0} for cap in CAPABILITY_DIMENSIONS}

    total = len(df)
    text_series = _combined_text_series(df)
    profile: dict[str, dict] = {}

    for cap in CAPABILITY_DIMENSIONS:
        cap_entries = [e for e in SKILL_TAXONOMY if e["capability"] == cap]
        keywords = set()
        for entry in cap_entries:
            for kw in entry["keywords"]:
                keywords.add(re.escape(kw.lower()))

        if keywords:
            pattern = r"\b(?:" + "|".join(sorted(keywords, key=len, reverse=True)) + r")\b"
            try:
                matched_count = int(text_series.str.contains(pattern, regex=True, na=False).sum())
            except re.error:
                matched_count = 0
        else:
            matched_count = 0

        pct = round(matched_count / total * 100, 1) if total > 0 else 0.0
        profile[cap] = {"pct": pct, "jobs": matched_count, "total": total}

    _capability_profile_cache = profile

    try:
        import json
        profile_file.parent.mkdir(exist_ok=True)
        with open(profile_file, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2)
    except Exception:
        pass

    return profile
