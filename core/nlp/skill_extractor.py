"""
core/nlp/skill_extractor.py
============================
Explainable, rule-based skill extraction from a job description.

Approach
--------
1. Lowercase and normalise whitespace.
2. Iterate over every entry in SKILL_TAXONOMY.
3. For each entry test all keyword variants using whole-word regex.
4. If any keyword matches, record one extracted skill (the first match).
5. De-duplicate by normalised skill name so each canonical skill appears once.

Output
------
list[dict] where each element has:
  - original_text   : the matched keyword substring as it appeared in the JD
  - normalized_skill: canonical skill name (from taxonomy)
  - capability      : the capability dimension
  - match_keyword   : the taxonomy keyword that triggered the match
  - confidence      : "rule_match" (always – this is a deterministic rule-based system)
"""
from __future__ import annotations

import re
import logging
from typing import Optional

from .taxonomy import SKILL_TAXONOMY

logger = logging.getLogger("dsxray.nlp")

# Pre-compile patterns once at import time for performance
_COMPILED_TAXONOMY: list[dict] = []

def _build_compiled_taxonomy() -> None:
    for entry in SKILL_TAXONOMY:
        compiled_patterns = []
        for kw in entry["keywords"]:
            # Escape special regex chars, then wrap in word boundary
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            try:
                compiled_patterns.append((kw, re.compile(pattern)))
            except re.error as exc:
                logger.warning("Failed to compile pattern for '%s': %s", kw, exc)
        _COMPILED_TAXONOMY.append({
            "compiled": compiled_patterns,
            "normalized": entry["normalized"],
            "capability": entry["capability"],
        })

_build_compiled_taxonomy()


def extract_skills(job_description: str) -> list[dict]:
    """
    Extract skills from a job description string.

    Parameters
    ----------
    job_description : str
        Raw job description text (any case).

    Returns
    -------
    list[dict]
        Deduplicated list of extracted skill records, ordered by capability
        then normalized skill name.
    """
    if not job_description or not job_description.strip():
        return []

    # Normalise: lowercase, collapse whitespace
    text = job_description.lower()
    text = re.sub(r"\s+", " ", text)

    seen_normalized: set[str] = set()
    extracted: list[dict] = []

    for entry in _COMPILED_TAXONOMY:
        if entry["normalized"] in seen_normalized:
            continue

        for kw, pattern in entry["compiled"]:
            m = pattern.search(text)
            if m:
                # Find the actual text in the original (preserve case)
                # by searching the same span in the lowercased text
                matched_text = m.group(0)
                extracted.append({
                    "original_text":    matched_text,
                    "normalized_skill": entry["normalized"],
                    "capability":       entry["capability"],
                    "match_keyword":    kw,
                    "confidence":       "rule_match",
                })
                seen_normalized.add(entry["normalized"])
                break  # one match per taxonomy entry is enough

    # Sort for deterministic output
    extracted.sort(key=lambda s: (s["capability"], s["normalized_skill"]))
    logger.debug("Extracted %d skills from JD (%d chars)", len(extracted), len(job_description))
    return extracted


def group_by_capability(extracted_skills: list[dict]) -> dict[str, list[str]]:
    """
    Group normalised skill names by capability dimension.

    Returns
    -------
    dict[str, list[str]]
        Keys are capability names; values are sorted lists of skill names.
    """
    from .taxonomy import CAPABILITY_DIMENSIONS
    groups: dict[str, list[str]] = {cap: [] for cap in CAPABILITY_DIMENSIONS}
    for s in extracted_skills:
        cap = s["capability"]
        if cap in groups:
            groups[cap].append(s["normalized_skill"])
    # Sort each group
    for cap in groups:
        groups[cap] = sorted(set(groups[cap]))
    return groups


def build_capability_vector(extracted_skills: list[dict]) -> dict[str, int]:
    """
    Count distinct skills per capability dimension.
    Returns raw integer counts (not normalised).
    """
    from .taxonomy import CAPABILITY_DIMENSIONS
    groups = group_by_capability(extracted_skills)
    return {cap: len(skills) for cap, skills in groups.items()}
