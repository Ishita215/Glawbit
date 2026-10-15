"""
core/views.py
=============
Thin Django views.  Business logic lives in core.analysis_service.
"""
import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from core.analysis_service import analyze_job_description
from core.analytics.market import (
    compute_market_statistics,
    get_top_skills,
    market_health_summary,
    capability_market_profile,
)
from core.data.loaders import dataset_health_report, load_jds_traits
from core.ml.outcome_model import load_meta
from core.models import Analysis, ExtractedSkill, CapabilityScore, Recommendation

logger = logging.getLogger("dsxray.views")


# ── Page views ──────────────────────────────────────────────────

def home(request):
    return render(request, "core/index.html")

def xray_page(request):
    return render(request, "core/xray.html")

def market_page(request):
    return render(request, "core/market.html")

def evidence_page(request):
    return render(request, "core/evidence.html")


# ── API: Analyze ─────────────────────────────────────────────

@csrf_exempt
def analyze(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required."}, status=405)

    try:
        # Accept both form-encoded and JSON
        if request.content_type and "application/json" in request.content_type:
            body = json.loads(request.body)
            job_description = body.get("job_description", "")
        else:
            job_description = request.POST.get("job_description", "")

        result = analyze_job_description(job_description)

        # ── Persist to DB ─────────────────────────────────
        try:
            analysis = Analysis.objects.create(
                job_description=job_description,
                status="success",
                processing_time_ms=result.get("processing_time_ms"),
                total_skills_extracted=len(result.get("extracted_skills", [])),
                dimensions_covered=result.get("requirement_stress", {}).get("dimensions_covered", 0),
            )

            for skill in result.get("extracted_skills", []):
                ExtractedSkill.objects.create(
                    analysis=analysis,
                    original_text=skill.get("original_text", ""),
                    normalized_skill=skill.get("normalized_skill", ""),
                    capability=skill.get("capability", ""),
                    match_keyword=skill.get("match_keyword", ""),
                    prevalence_pct=skill.get("prevalence_pct"),
                    demand_tier=skill.get("demand_tier", ""),
                )

            for cap, count in result.get("capability_vector", {}).items():
                CapabilityScore.objects.create(
                    analysis=analysis,
                    capability=cap,
                    raw_count=count,
                )

            for rec in result.get("recommendations", []):
                Recommendation.objects.create(
                    analysis=analysis,
                    signal=rec.get("signal", ""),
                    evidence=rec.get("evidence", ""),
                    recommendation=rec.get("recommendation", ""),
                    reason=rec.get("reason", ""),
                    severity=rec.get("severity", "Medium"),
                    rec_type=rec.get("type", "structure"),
                )

            result["analysis_id"] = analysis.pk

        except Exception as db_err:
            logger.warning("DB persistence error (non-fatal): %s", db_err)

        return JsonResponse({"success": True, "result": result})

    except ValueError as err:
        return JsonResponse({"success": False, "error": str(err)}, status=400)

    except Exception as err:
        logger.exception("Analysis error: %s", err)
        return JsonResponse({"success": False, "error": f"Analysis error: {err}"}, status=500)


# ── API: Market intelligence ──────────────────────────────────

@require_GET
def api_market_overview(request):
    """
    Returns top skills, capability market profile, and dataset health.
    """
    try:
        top_skills      = get_top_skills(n=30)
        cap_profile     = capability_market_profile()
        health          = market_health_summary()
        return JsonResponse({
            "success": True,
            "top_skills": top_skills,
            "capability_profile": cap_profile,
            "market_health": health,
        })
    except Exception as err:
        logger.exception("Market overview error: %s", err)
        return JsonResponse({"success": False, "error": str(err)}, status=500)


# ── API: Dataset health ───────────────────────────────────────

@require_GET
def api_data_health(request):
    try:
        return JsonResponse({"success": True, "report": dataset_health_report()})
    except Exception as err:
        logger.exception("Data health error: %s", err)
        return JsonResponse({"success": False, "error": str(err)}, status=500)


# ── API: Model evidence ───────────────────────────────────────

@require_GET
def api_model_evidence(request):
    """Returns model metadata for the Evidence / Judge Mode page."""
    try:
        meta = load_meta()
        return JsonResponse({"success": True, "model_meta": meta})
    except Exception as err:
        logger.exception("Model evidence error: %s", err)
        return JsonResponse({"success": False, "error": str(err)}, status=500)


# ── API: Command Centre stats ─────────────────────────────────

@require_GET
def api_command_centre(request):
    try:
        analyses_count = Analysis.objects.count()
        skills_tracked = len(compute_market_statistics()) - 1  # subtract _meta
        jds_df         = load_jds_traits()
        outcome_records = len(jds_df) if not jds_df.empty else 0
        meta           = load_meta()
        mean_auc       = (
            meta.get("models", {})
                .get("logistic_regression", {})
                .get("mean_auc")
        )
        from core.data.loaders import load_datascience_jobs
        ds_jobs_df     = load_datascience_jobs()
        ds_jobs_count  = len(ds_jobs_df) if not ds_jobs_df.empty else 0

        return JsonResponse({
            "success": True,
            "stats": {
                "analyses_run":    analyses_count,
                "ds_jobs":         ds_jobs_count,
                "outcome_records": outcome_records,
                "skills_tracked":  skills_tracked,
                "model_auc":       round(mean_auc, 3) if mean_auc else None,
                "model_version":   meta.get("model_version", "unknown"),
            }
        })
    except Exception as err:
        logger.exception("Command centre error: %s", err)
        return JsonResponse({"success": False, "error": str(err)}, status=500)
