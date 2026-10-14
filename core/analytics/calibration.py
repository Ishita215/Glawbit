"""
core/analytics/calibration.py
==============================
Requirement Stress Test, Frankenstein Role Detector, Structural Balance,
Market x Outcome Matrix, and Calibration Engine.

All diagnostics are rule-based and explainable.
No magic numbers are hidden – thresholds are declared as named constants
at the top of each section with docstrings explaining their rationale.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from core.nlp.taxonomy import CAPABILITY_DIMENSIONS, ROLE_FAMILY_PATTERNS

logger = logging.getLogger("dsxray.analytics.calibration")


# ────────────────────────────────────────────────────────────────
# CONSTANTS  (all tunable from one place)
# ────────────────────────────────────────────────────────────────

# Requirement breadth: if total extracted skills exceed this, signal HIGH breadth
BREADTH_HIGH_THRESHOLD = 14
BREADTH_MEDIUM_THRESHOLD = 8

# Dimensions covered: if JD covers this many or more capability dimensions,
# consider it "full-spectrum"
FULL_SPECTRUM_DIMENSIONS = 5

# Rare-skill concentration: if more than this fraction of skills are in the
# "Low" demand tier, flag rare-skill concentration
RARE_FRACTION_HIGH = 0.40
RARE_FRACTION_MEDIUM = 0.20

# Experience parsing: regex to find "N yrs" / "N years" patterns
_EXP_PATTERN = re.compile(
    r"(\d+)\s*(?:to|-|–)\s*(\d+)\s*(?:yrs?|years?)"
    r"|(\d+)\+?\s*(?:yrs?|years?)",
    re.IGNORECASE,
)

# Frankenstein detector: if JD matches skills from this many distinct role
# families, flag as possible multi-role
MULTI_ROLE_FAMILY_THRESHOLD = 3
# Minimum number of signals from a family to count it as "present"
ROLE_FAMILY_MIN_SIGNALS = 2


# ────────────────────────────────────────────────────────────────
# EXPERIENCE PARSING
# ────────────────────────────────────────────────────────────────

def parse_experience_years(text: str) -> Optional[tuple[int, int]]:
    """
    Return (min_years, max_years) from JD text, or None if not found.
    Handles patterns like:
      "5+ years", "3-5 years", "3 to 5 yrs", "3-8 yrs"
    """
    for m in _EXP_PATTERN.finditer(text):
        if m.group(1) and m.group(2):      # range pattern
            return int(m.group(1)), int(m.group(2))
        elif m.group(3):                    # single/+ pattern
            val = int(m.group(3))
            return val, val
    return None


# ────────────────────────────────────────────────────────────────
# REQUIREMENT STRESS TEST
# ────────────────────────────────────────────────────────────────

def requirement_stress_test(
    extracted_skills: list[dict],
    capability_vector: dict[str, int],
    job_description: str,
) -> dict:
    """
    Detect whether the JD places excessive demands on candidates.

    Signals checked
    ---------------
    1. Total skill count (breadth)
    2. Number of capability dimensions covered
    3. Rare-skill concentration (Low-demand-tier skills)
    4. Experience requirement
    5. Mixed-role signals (delegated to frankenstein_role_detector)

    Returns
    -------
    dict with:
        breadth_level     : "High" | "Medium" | "Low"
        total_skills      : int
        dimensions_covered: int
        rare_skill_count  : int
        rare_skill_fraction: float
        rare_skills       : list[str]
        experience_range  : tuple | None
        experience_stress : bool
        overall_stress    : "High" | "Medium" | "Low"
        signals           : list[str]   – human-readable signal messages
        evidence          : list[str]   – evidence statements
    """
    total = len(extracted_skills)
    dims_covered = sum(1 for v in capability_vector.values() if v > 0)

    # Breadth
    if total >= BREADTH_HIGH_THRESHOLD:
        breadth_level = "High"
    elif total >= BREADTH_MEDIUM_THRESHOLD:
        breadth_level = "Medium"
    else:
        breadth_level = "Low"

    # Rare skill concentration
    rare_skills = [
        s["normalized_skill"]
        for s in extracted_skills
        if s.get("demand_tier") == "Low"
    ]
    rare_count = len(rare_skills)
    rare_fraction = round(rare_count / total, 2) if total > 0 else 0.0
    rare_stress = rare_fraction >= RARE_FRACTION_HIGH

    # Experience
    exp_range = parse_experience_years(job_description)
    experience_stress = False
    if exp_range:
        min_exp, max_exp = exp_range
        # Flag if min experience >= 7 AND skill breadth is high
        if min_exp >= 7 and breadth_level == "High":
            experience_stress = True

    # Overall stress
    stress_factors = sum([
        breadth_level == "High",
        dims_covered >= FULL_SPECTRUM_DIMENSIONS,
        rare_fraction >= RARE_FRACTION_HIGH,
        experience_stress,
    ])
    if stress_factors >= 3:
        overall_stress = "High"
    elif stress_factors >= 1:
        overall_stress = "Medium"
    else:
        overall_stress = "Low"

    # Build human-readable signals and evidence
    signals: list[str] = []
    evidence: list[str] = []

    if breadth_level == "High":
        signals.append(
            f"High requirement breadth: {total} distinct skills detected "
            f"(threshold ≥ {BREADTH_HIGH_THRESHOLD})."
        )
        evidence.append(
            f"The JD contains {total} distinct skills across "
            f"{dims_covered} capability dimensions."
        )
    elif breadth_level == "Medium":
        signals.append(
            f"Moderate requirement breadth: {total} skills detected."
        )

    if dims_covered >= FULL_SPECTRUM_DIMENSIONS:
        signals.append(
            f"Full-spectrum coverage: skills span all {dims_covered} "
            f"capability dimensions."
        )
        evidence.append(
            "A role covering all five capability dimensions simultaneously may "
            "be difficult to fill and may indicate a composite-role definition."
        )

    if rare_fraction >= RARE_FRACTION_HIGH:
        signals.append(
            f"Rare-skill concentration: {rare_count} of {total} skills "
            f"({round(rare_fraction*100)}%) are in the Low market-demand tier."
        )
        evidence.append(
            f"Rare skills detected: {', '.join(rare_skills[:8])}."
        )
    elif rare_fraction >= RARE_FRACTION_MEDIUM:
        signals.append(
            f"Some rare skills: {rare_count} skill(s) have low market prevalence."
        )

    if experience_stress and exp_range:
        signals.append(
            f"Experience requirement ({exp_range[0]}+ years) combined with "
            f"high skill breadth may significantly constrain the candidate pool."
        )

    return {
        "breadth_level":      breadth_level,
        "total_skills":       total,
        "dimensions_covered": dims_covered,
        "rare_skill_count":   rare_count,
        "rare_skill_fraction": rare_fraction,
        "rare_skills":        rare_skills,
        "experience_range":   list(exp_range) if exp_range else None,
        "experience_stress":  experience_stress,
        "overall_stress":     overall_stress,
        "signals":            signals,
        "evidence":           evidence,
        "thresholds": {
            "breadth_high":        BREADTH_HIGH_THRESHOLD,
            "breadth_medium":      BREADTH_MEDIUM_THRESHOLD,
            "rare_fraction_high":  RARE_FRACTION_HIGH,
            "rare_fraction_medium": RARE_FRACTION_MEDIUM,
            "full_spectrum_dims":  FULL_SPECTRUM_DIMENSIONS,
        },
    }


# ────────────────────────────────────────────────────────────────
# FRANKENSTEIN ROLE DETECTOR
# ────────────────────────────────────────────────────────────────

def detect_role_families(extracted_skills: list[dict]) -> dict:
    """
    Detect which role families are represented in the JD.

    Returns
    -------
    dict with:
        detected_families  : list of {family, signals, signal_count}
        is_multi_role      : bool
        multi_role_message : str
        contributing_skills: list[str]
    """
    skill_names = {s["normalized_skill"] for s in extracted_skills}

    detected: list[dict] = []
    for family, signature_skills in ROLE_FAMILY_PATTERNS.items():
        matched = [s for s in signature_skills if s in skill_names]
        if len(matched) >= ROLE_FAMILY_MIN_SIGNALS:
            detected.append({
                "family":       family,
                "signals":      matched,
                "signal_count": len(matched),
            })

    detected.sort(key=lambda x: x["signal_count"], reverse=True)

    is_multi = len(detected) >= MULTI_ROLE_FAMILY_THRESHOLD

    if is_multi:
        family_names = [d["family"] for d in detected]
        msg = (
            f"Possible multi-role requirement detected. "
            f"The JD contains signals from {len(detected)} role families: "
            f"{', '.join(family_names)}.  "
            "Consider whether this is intentional or whether requirements "
            "should be split into separate roles or preferred/required tiers."
        )
    elif len(detected) == 2:
        msg = (
            f"The JD spans two role families "
            f"({detected[0]['family']} and {detected[1]['family']}). "
            "This may be intentional for hybrid roles."
        )
    elif len(detected) == 1:
        msg = f"The JD is primarily aligned with: {detected[0]['family']}."
    else:
        msg = "Insufficient signals to classify this JD into a standard role family."

    # All skills contributing to any family match
    contributing = sorted({
        s
        for d in detected
        for s in d["signals"]
    })

    return {
        "detected_families":  detected,
        "family_count":       len(detected),
        "is_multi_role":      is_multi,
        "multi_role_message": msg,
        "contributing_skills": contributing,
        "disclaimer": (
            "Role-family classification uses a rule-based skill pattern matcher. "
            "It is indicative only and should be validated against the actual "
            "business context of the role."
        ),
    }


# ────────────────────────────────────────────────────────────────
# STRUCTURAL BALANCE
# ────────────────────────────────────────────────────────────────

def structural_balance(capability_vector: dict[str, int]) -> dict:
    """
    Measure how evenly skill requirements are distributed across the
    five capability dimensions.

    Method
    ------
    - Coefficient of Variation (std / mean) of the raw skill counts.
    - High CV → highly concentrated; Low CV → balanced.
    - We invert and scale to 0-100 so that 100 = perfectly balanced.

    Returns
    -------
    dict with:
        balance_score   : float 0-100 (100 = perfectly balanced)
        cv              : float (coefficient of variation)
        dimension_pct   : dict – each dimension's share of total skills
        concentration   : "High" | "Medium" | "Low"
        dominant_dimension: str | None
        deficit_dimensions: list[str]
        interpretation  : str
    """
    values = np.array(
        [capability_vector.get(cap, 0) for cap in CAPABILITY_DIMENSIONS],
        dtype=float,
    )

    total = float(values.sum())
    if total == 0:
        return {
            "balance_score": 0.0,
            "cv": 0.0,
            "dimension_pct": {cap: 0.0 for cap in CAPABILITY_DIMENSIONS},
            "concentration": "Low",
            "dominant_dimension": None,
            "deficit_dimensions": CAPABILITY_DIMENSIONS[:],
            "interpretation": "No skills detected; structural balance cannot be assessed.",
        }

    mean_val = float(values.mean())
    std_val = float(values.std())
    cv = std_val / mean_val if mean_val > 0 else 0.0

    # Balance score: 100 = zero variance, 0 = all skills in one dimension
    balance_score = round(max(0.0, min(100.0, (1 - cv) * 100)), 1)

    # Dimension percentages
    dim_pct = {
        cap: round(float(values[i]) / total * 100, 1)
        for i, cap in enumerate(CAPABILITY_DIMENSIONS)
    }

    # Concentration level
    if cv >= 0.8:
        concentration = "High"
    elif cv >= 0.4:
        concentration = "Medium"
    else:
        concentration = "Low"

    # Dominant dimension
    max_idx = int(np.argmax(values))
    dominant = CAPABILITY_DIMENSIONS[max_idx] if values[max_idx] > mean_val * 1.5 else None

    # Deficit dimensions (below 10% of total)
    deficit = [cap for i, cap in enumerate(CAPABILITY_DIMENSIONS) if dim_pct[cap] < 10.0]

    if balance_score >= 80:
        interp = "Requirements are relatively balanced across capability dimensions."
    elif balance_score >= 60:
        interp = (
            f"Requirements are moderately concentrated. "
            f"{dominant + ' dominates. ' if dominant else ''}"
            "Consider whether other dimensions need strengthening."
        )
    else:
        interp = (
            f"Requirements are highly concentrated. "
            f"{'Dominant: ' + dominant + '. ' if dominant else ''}"
            "Deficit dimensions: " + ", ".join(deficit) + "."
        )

    return {
        "balance_score":      balance_score,
        "cv":                 round(cv, 4),
        "dimension_pct":      dim_pct,
        "raw_counts":         {cap: int(values[i]) for i, cap in enumerate(CAPABILITY_DIMENSIONS)},
        "total_skills":       int(total),
        "concentration":      concentration,
        "dominant_dimension": dominant,
        "deficit_dimensions": deficit,
        "interpretation":     interp,
        "methodology": (
            "Balance score = (1 - Coefficient of Variation) × 100. "
            "Coefficient of Variation = std(skill_counts) / mean(skill_counts). "
            "100 = equal skills across all dimensions; 0 = all skills in one dimension."
        ),
    }


# ────────────────────────────────────────────────────────────────
# MARKET × OUTCOME INTELLIGENCE MATRIX
# ────────────────────────────────────────────────────────────────

# Outcome correlation data from JDS Skill Traits (from spec)
# These are from the provided dataset analysis and will be recomputed
# when train_models is run.  Until then, we use the spec values.
_OUTCOME_CORRELATION_DEFAULTS: dict[str, float] = {
    "Big Data":                  0.112,
    "Maths / Statistics":        0.524,
    "Coding":                    0.444,
    "AI / ML":                   0.405,
    "Dashboard & Storytelling":  0.554,
}

_OUTCOME_CORR_HIGH_THRESHOLD = 0.40   # r >= 0.40 → High outcome association
_OUTCOME_CORR_MEDIUM_THRESHOLD = 0.20  # r >= 0.20 → Medium


def market_outcome_matrix(
    extracted_skills: list[dict],
    capability_vector: dict[str, int],
    outcome_correlations: Optional[dict[str, float]] = None,
) -> dict:
    """
    Build the Market × Outcome matrix for each capability dimension.

    Quadrants
    ---------
    A: High Market Demand + Low Outcome Association  → market-required, weaker outcome link
    B: High Market Demand + High Outcome Association → strategic core
    C: Low Market Demand  + High Outcome Association → niche / high-value
    D: Low Market Demand  + Low Outcome Association  → optional / role-specific

    Returns
    -------
    dict with:
        matrix: list of {capability, market_demand, outcome_assoc, quadrant, interpretation}
        methodology: str
        disclaimer: str
    """
    from core.analytics.market import capability_market_profile
    if outcome_correlations is None:
        outcome_correlations = _OUTCOME_CORRELATION_DEFAULTS

    market_profile = capability_market_profile()

    # Determine market demand tier per capability using 50th percentile split
    market_pcts = [market_profile.get(cap, {}).get("pct", 0) for cap in CAPABILITY_DIMENSIONS]
    market_median = float(np.median(market_pcts)) if market_pcts else 50.0
    total_jd_skills = sum(capability_vector.get(cap, 0) for cap in CAPABILITY_DIMENSIONS)

    matrix_rows: list[dict] = []

    for cap in CAPABILITY_DIMENSIONS:
        mkt_pct = market_profile.get(cap, {}).get("pct", 0.0)
        corr = outcome_correlations.get(cap, 0.0)
        is_in_jd = capability_vector.get(cap, 0) > 0

        # Market tier
        market_tier = "High" if mkt_pct >= market_median else "Low"

        # Outcome tier
        if corr >= _OUTCOME_CORR_HIGH_THRESHOLD:
            outcome_tier = "High"
        elif corr >= _OUTCOME_CORR_MEDIUM_THRESHOLD:
            outcome_tier = "Medium"
        else:
            outcome_tier = "Low"

        # Quadrant
        if market_tier == "High" and outcome_tier == "High":
            quadrant = "B"
            label    = "Strategic Core"
            action   = "Keep as core requirement if relevant to the role."
        elif market_tier == "High" and outcome_tier in ("Low", "Medium"):
            quadrant = "A"
            label    = "Market-Demanded"
            action   = "Market-required; weaker outcome association in the provided dataset."
        elif market_tier == "Low" and outcome_tier == "High":
            quadrant = "C"
            label    = "Niche / High-Value"
            action   = (
                "Less common in the market but associated with better outcomes. "
                "Consider as a valued differentiator."
            )
        else:
            quadrant = "D"
            label    = "Optional / Context-Specific"
            action   = "Low market demand and weaker outcome association; make optional if included."

        matrix_rows.append({
            "capability":        cap,
            "market_pct":        round(mkt_pct, 1),
            "market_tier":       market_tier,
            "outcome_corr":      round(corr, 3),
            "outcome_tier":      outcome_tier,
            "quadrant":          quadrant,
            "quadrant_label":    label,
            "action":            action,
            "present_in_jd":     is_in_jd,
            "skill_count":       capability_vector.get(cap, 0),
            "jd_pct":            round(
                capability_vector.get(cap, 0) / total_jd_skills * 100, 1
            ) if total_jd_skills else 0.0,
        })

    return {
        "matrix":    matrix_rows,
        "market_median_pct": round(market_median, 1),
        "outcome_thresholds": {
            "high":   _OUTCOME_CORR_HIGH_THRESHOLD,
            "medium": _OUTCOME_CORR_MEDIUM_THRESHOLD,
        },
        "methodology": (
            "Market tier is based on whether the capability's market prevalence "
            "is above or below the median across all five dimensions. "
            "Outcome association uses Pearson correlation with salary_hike_high_or_low "
            "from the JDS Skill Traits dataset."
        ),
        "disclaimer": (
            "These are associations in the provided JDS dataset. "
            "They do not establish causal relationships between capabilities and outcomes. "
            "Market prevalence is computed from Analytics Jobs.csv."
        ),
    }


# ────────────────────────────────────────────────────────────────
# CALIBRATION ENGINE
# ────────────────────────────────────────────────────────────────

def generate_calibration_recommendations(
    extracted_skills: list[dict],
    stress_test: dict,
    role_families: dict,
    balance: dict,
    matrix: dict,
) -> list[dict]:
    """
    Generate actionable, evidence-driven recommendations.

    Each recommendation has:
        signal      – what was detected
        evidence    – data supporting the signal
        recommendation – the action to take
        reason      – why this recommendation makes sense
        severity    – "High" | "Medium" | "Low"
        type        – "remove" | "relax" | "keep" | "prefer" | "structure"
    """
    recs: list[dict] = []

    # ── R1: High breadth → move rare skills to preferred ──────
    if stress_test["breadth_level"] == "High" and stress_test["rare_skill_count"] > 0:
        rare_list = stress_test["rare_skills"][:6]
        recs.append({
            "signal":         f"High requirement breadth ({stress_test['total_skills']} skills) with rare requirements.",
            "evidence":       f"{stress_test['rare_skill_count']} skill(s) have low market prevalence: {', '.join(rare_list)}.",
            "recommendation": f"Move these skills to 'Preferred' or 'Nice-to-Have': {', '.join(rare_list)}.",
            "reason":         (
                "These skills are rare in the market, which constrains your candidate pool. "
                "Making them preferred rather than required widens the talent pool "
                "without lowering the bar for high-demand skills."
            ),
            "severity":       "High",
            "type":           "prefer",
            "skills_affected": rare_list,
        })

    # ── R2: Frankenstein role → consider splitting ─────────────
    if role_families["is_multi_role"]:
        families = [d["family"] for d in role_families["detected_families"]]
        recs.append({
            "signal":         f"Multi-role signals detected across {len(families)} role families.",
            "evidence":       f"Role families: {', '.join(families[:4])}. Contributing skills: {', '.join(role_families['contributing_skills'][:6])}.",
            "recommendation": (
                "Review whether this role should be split into two more focused roles, "
                "or whether requirements from secondary families should be moved to 'Preferred'."
            ),
            "reason":         (
                "Roles spanning multiple families are difficult to fill, may attract mismatched candidates, "
                "and can reflect conflicting expectations. Clarity about the primary role identity "
                "improves both hiring quality and candidate experience."
            ),
            "severity":       "High",
            "type":           "structure",
            "skills_affected": role_families["contributing_skills"][:8],
        })

    # ── R3: Strategic core skills → keep ──────────────────────
    strategic = [
        r for r in matrix["matrix"]
        if r["quadrant"] == "B" and r["present_in_jd"]
    ]
    if strategic:
        caps = [r["capability"] for r in strategic]
        recs.append({
            "signal":         "Strategic core capabilities are present in the JD.",
            "evidence":       f"High market demand + high outcome association: {', '.join(caps)}.",
            "recommendation": f"Keep these as core requirements: {', '.join(caps)}.",
            "reason":         (
                "These capabilities are both in high market demand and associated with "
                "positive outcomes in the JDS dataset. They are the strongest justification "
                "for requirements in this JD."
            ),
            "severity":       "Low",
            "type":           "keep",
            "skills_affected": caps,
        })

    # ── R4: Niche/high-value capabilities present ─────────────
    niche = [
        r for r in matrix["matrix"]
        if r["quadrant"] == "C" and r["present_in_jd"]
    ]
    if niche:
        caps = [r["capability"] for r in niche]
        recs.append({
            "signal":         "Niche capabilities with high outcome association detected.",
            "evidence":       f"Low market prevalence but high outcome association: {', '.join(caps)}.",
            "recommendation": (
                f"Consider labelling {', '.join(caps)} as a valued differentiator "
                "rather than a strict requirement to avoid narrowing the candidate pool."
            ),
            "reason":         (
                "These capabilities are rare in the market, which may constrain supply. "
                "However, they are associated with better outcomes in the dataset, "
                "so consider them a strong 'nice-to-have' rather than eliminating them."
            ),
            "severity":       "Medium",
            "type":           "prefer",
            "skills_affected": caps,
        })

    # ── R5: Full-spectrum coverage ─────────────────────────────
    if stress_test["dimensions_covered"] >= 5:
        recs.append({
            "signal":         "The JD covers all five capability dimensions.",
            "evidence":       "Skills detected in: Big Data, Maths/Stats, Coding, AI/ML, Dashboard & Storytelling.",
            "recommendation": (
                "Review whether full-spectrum coverage is intentional. "
                "Consider identifying the two or three core dimensions for this specific role."
            ),
            "reason":         (
                "Covering all five dimensions simultaneously is rare in market data and "
                "may indicate a 'hybrid role' that is difficult to hire for. "
                "Identifying primary and secondary dimensions clarifies expectations."
            ),
            "severity":       "Medium",
            "type":           "structure",
            "skills_affected": [],
        })

    # ── R6: Experience stress ──────────────────────────────────
    if stress_test["experience_stress"] and stress_test["experience_range"]:
        min_exp = stress_test["experience_range"][0]
        recs.append({
            "signal":         f"High experience requirement ({min_exp}+ years) combined with high skill breadth.",
            "evidence":       (
                f"Minimum {min_exp} years of experience + "
                f"{stress_test['total_skills']} distinct skills detected."
            ),
            "recommendation": (
                "Consider whether a lower experience floor (e.g., 5 years) combined with "
                "a strong skills assessment would be more effective than the current requirement."
            ),
            "reason":         (
                "High experience requirements combined with high skill breadth significantly "
                "constrain the candidate pool. The two constraints together may make the "
                "role very difficult to fill."
            ),
            "severity":       "Medium",
            "type":           "relax",
            "skills_affected": [],
        })

    # ── R7: Structural imbalance ───────────────────────────────
    if balance["concentration"] == "High" and balance["dominant_dimension"]:
        dom = balance["dominant_dimension"]
        deficit = balance["deficit_dimensions"]
        recs.append({
            "signal":         f"High skill concentration in {dom}.",
            "evidence":       f"Balance score: {balance['balance_score']}/100. Deficit dimensions: {', '.join(deficit) if deficit else 'none'}.",
            "recommendation": (
                f"Review whether the concentration in {dom} is intentional. "
                + (f"Consider adding requirements in: {', '.join(deficit[:3])}." if deficit else "")
            ),
            "reason":         (
                "Highly concentrated requirements may leave critical adjacent capabilities "
                "unaddressed. A more balanced profile often reflects a clearer role definition."
            ),
            "severity":       "Low",
            "type":           "structure",
            "skills_affected": [],
        })

    # Sort: High → Medium → Low severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: severity_order.get(r["severity"], 3))

    return recs


# ────────────────────────────────────────────────────────────────
# SCORECARD
# ────────────────────────────────────────────────────────────────

def build_scorecard(
    outcome_fit: dict,
    stress_test: dict,
    balance: dict,
    extracted_skills: list[dict],
) -> dict:
    """
    Build the four-dimension Role X-Ray Scorecard.

    Dimensions
    ----------
    1. Outcome Fit        – from outcome model (0-100, or None)
    2. Demand Alignment   – fraction of JD skills in High/Medium demand tier
    3. Structural Balance – from balance diagnostic (0-100)
    4. Requirement Stress – inverse of overall_stress level

    No composite score is emitted by default.
    If a composite is needed for display, it is clearly labelled as a
    configurable product heuristic, NOT a model probability.
    """
    # Outcome Fit
    outcome_score = outcome_fit.get("probability_high")

    # Demand Alignment
    total_skills = len(extracted_skills)
    if total_skills > 0:
        high_medium = sum(
            1 for s in extracted_skills
            if s.get("demand_tier") in ("High", "Medium")
        )
        demand_alignment_score = round(high_medium / total_skills * 100, 1)
    else:
        demand_alignment_score = None

    # Structural Balance
    balance_score = balance.get("balance_score")

    # Requirement Stress (inverted)
    stress_map = {"Low": 90.0, "Medium": 55.0, "High": 20.0}
    stress_score = stress_map.get(stress_test.get("overall_stress", "Medium"), 55.0)

    # Count available (non-None) scores
    available_scores = [
        s for s in [outcome_score, demand_alignment_score, balance_score, stress_score]
        if s is not None
    ]

    if len(available_scores) == 4:
        # Simple equal-weight average – clearly labelled as a heuristic
        composite = round(
            0.30 * (outcome_score or 50)
            + 0.30 * (demand_alignment_score or 50)
            + 0.20 * balance_score
            + 0.20 * stress_score,
            1,
        )
        composite_label = "Product heuristic (30% Outcome Fit + 30% Demand Alignment + 20% Balance + 20% Stress Inverse)"
    else:
        composite = None
        composite_label = "Composite not available (model not trained)"

    return {
        "outcome_fit": {
            "score":       outcome_score,
            "label":       "Outcome Fit",
            "description": (
                "Probability that this capability profile is associated with a high "
                "salary-hike outcome in the JDS Skill Traits dataset. "
                "Association measure only – not a prediction of employee success."
            ),
        },
        "demand_alignment": {
            "score":       demand_alignment_score,
            "label":       "Demand Alignment",
            "description": (
                f"{high_medium if total_skills > 0 else 0} of {total_skills} extracted skills "
                "are in the High or Medium market-demand tier."
            ),
        },
        "structural_balance": {
            "score":       balance_score,
            "label":       "Structural Balance",
            "description": balance.get("interpretation", ""),
        },
        "requirement_stress": {
            "score":       stress_score,
            "level":       stress_test.get("overall_stress", "Unknown"),
            "label":       "Requirement Stress (inverted)",
            "description": (
                "Inverted stress score: higher = less stressed JD. "
                f"Overall stress: {stress_test.get('overall_stress', 'Unknown')}."
            ),
        },
        "composite": {
            "score":       composite,
            "methodology": composite_label,
            "warning":     (
                "This composite is a configurable product heuristic, NOT a model output. "
                "Individual dimension scores are more informative."
            ),
        },
    }
