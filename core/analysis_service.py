"""
core/analysis_service.py
=========================
Main orchestration layer.  Django views call this; they do NOT call
individual analytics modules directly.

analyze_job_description()
  – the single entry point for the /analyze/ API.

All heavy computation is delegated to:
  core.nlp.skill_extractor
  core.ml.outcome_model
  core.analytics.market
  core.analytics.calibration
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.nlp.skill_extractor import (
    build_capability_vector,
    extract_skills,
    group_by_capability,
)
from core.nlp.taxonomy import CAPABILITY_DIMENSIONS
from core.ml.outcome_model import predict_outcome_fit
from core.analytics.market import (
    compute_market_statistics,
    get_skill_prevalence_for_extracted,
)
from core.analytics.calibration import (
    build_scorecard,
    detect_role_families,
    generate_calibration_recommendations,
    market_outcome_matrix,
    requirement_stress_test,
    structural_balance,
)

logger = logging.getLogger("dsxray.analysis")


def analyze_job_description(job_description: str) -> dict[str, Any]:
    """
    Full DS-XRAY analysis pipeline.

    Parameters
    ----------
    job_description : str  Raw JD text.

    Returns
    -------
    dict – structured analysis result (JSON-serialisable).
    """
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    jd = job_description.strip()
    t_start = time.monotonic()

    logger.info("Starting analysis (JD length=%d chars)", len(jd))

    # ── P0.2  Skill extraction ──────────────────────────────
    extracted_skills = extract_skills(jd)
    logger.info("Extracted %d skills", len(extracted_skills))

    # ── P0.4  Capability vector (raw counts) ────────────────
    capability_vector = build_capability_vector(extracted_skills)
    skills_by_capability = group_by_capability(extracted_skills)

    # ── P0.7  Market intelligence ───────────────────────────
    enriched_skills = get_skill_prevalence_for_extracted(extracted_skills)

    # ── P0.5  Outcome model ─────────────────────────────────
    outcome_fit = predict_outcome_fit(capability_vector)

    # ── P0.10 Requirement stress test ──────────────────────
    stress = requirement_stress_test(enriched_skills, capability_vector, jd)

    # ── P0.11 Frankenstein role detector ───────────────────
    role_families = detect_role_families(extracted_skills)

    # ── P0.12 Structural balance ────────────────────────────
    balance = structural_balance(capability_vector)

    # ── P0.9  Market × Outcome matrix ──────────────────────
    mo_matrix = market_outcome_matrix(extracted_skills, capability_vector)

    # ── P0.13 Scorecard ─────────────────────────────────────
    scorecard = build_scorecard(outcome_fit, stress, balance, enriched_skills)

    # ── P0.14 Calibration recommendations ──────────────────
    recommendations = generate_calibration_recommendations(
        enriched_skills, stress, role_families, balance, mo_matrix
    )

    # ── Demand alignment summary ────────────────────────────
    total_skills = len(enriched_skills)
    high_demand  = [s for s in enriched_skills if s.get("demand_tier") == "High"]
    med_demand   = [s for s in enriched_skills if s.get("demand_tier") == "Medium"]
    low_demand   = [s for s in enriched_skills if s.get("demand_tier") == "Low"]

    demand_summary = {
        "total_skills":     total_skills,
        "high_demand":      len(high_demand),
        "medium_demand":    len(med_demand),
        "low_demand":       len(low_demand),
        "high_skills":      [s["normalized_skill"] for s in high_demand],
        "medium_skills":    [s["normalized_skill"] for s in med_demand],
        "low_skills":       [s["normalized_skill"] for s in low_demand],
    }

    elapsed_ms = round((time.monotonic() - t_start) * 1000, 1)
    logger.info("Analysis complete in %.1f ms", elapsed_ms)

    return {
        "status":            "success",
        "processing_time_ms": elapsed_ms,

        # P0.2 / P0.3
        "extracted_skills":  enriched_skills,
        "skills_by_capability": skills_by_capability,

        # P0.4
        "capability_vector": capability_vector,

        # P0.5
        "outcome_fit":       outcome_fit,

        # P0.7 / P0.8
        "market_demand":     demand_summary,

        # P0.9
        "market_outcome_matrix": mo_matrix,

        # P0.10
        "requirement_stress": stress,

        # P0.11
        "role_family_signals": role_families,

        # P0.12
        "structural_balance": balance,

        # P0.13
        "scorecard":          scorecard,

        # P0.14
        "recommendations":    recommendations,

        # Methodology
        "methodology": {
            "skill_extraction": (
                "Explainable rule-based extraction using a controlled skill taxonomy. "
                "Each skill is matched via whole-word regex against the job description. "
                "No LLM or black-box model is used."
            ),
            "capability_mapping": (
                "Each extracted skill maps to exactly one of five capability dimensions "
                "defined in core/nlp/taxonomy.py.  The mapping is editable and transparent."
            ),
            "outcome_model": (
                "Logistic Regression with StandardScaler, validated via Stratified 5-Fold "
                "Cross-Validation on JDS Skill Traits.xlsx (n=139). "
                "Primary metric: ROC-AUC. "
                "Output is an association probability, NOT a prediction of employee success."
            ),
            "market_prevalence": (
                "Skill prevalence = (jobs mentioning skill) / (total usable jobs) "
                "from Analytics Jobs.csv. "
                "Demand tiers (High / Medium / Low) are set at the 75th and 25th percentiles "
                "of the observed prevalence distribution."
            ),
        },

        "limitations": [
            "Skill extraction is rule-based. Skills expressed in unusual phrasing may be missed.",
            "The outcome model is trained on n=139 records (JDS Skill Traits). "
            "Results should be interpreted as indicative associations, not causal predictions.",
            "Market prevalence covers Analytics Jobs.csv only. It may not represent all geographies or sectors.",
            "Role-family classification uses skill pattern matching and is indicative only.",
            "Personality trait data (SDS) is available for descriptive analysis only "
            "and must not be used for automated hiring decisions.",
        ],
    }
