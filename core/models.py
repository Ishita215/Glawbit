"""
core/models.py
==============
Django database models for DS-XRAY.
"""
from django.db import models


class Analysis(models.Model):
    """Top-level record for each JD analysis request."""
    created_at      = models.DateTimeField(auto_now_add=True)
    job_description = models.TextField()
    status          = models.CharField(max_length=32, default="success")
    processing_time_ms = models.FloatField(null=True, blank=True)
    total_skills_extracted = models.IntegerField(default=0)
    dimensions_covered = models.IntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        snippet = self.job_description[:60].replace("\n", " ")
        return f"Analysis #{self.pk} – {snippet}…"


class ExtractedSkill(models.Model):
    """One normalised skill extracted from a JD."""
    analysis        = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="extracted_skills")
    original_text   = models.CharField(max_length=200)
    normalized_skill = models.CharField(max_length=200)
    capability      = models.CharField(max_length=100)
    match_keyword   = models.CharField(max_length=200, blank=True)
    prevalence_pct  = models.FloatField(null=True, blank=True)
    demand_tier     = models.CharField(max_length=20, blank=True)

    class Meta:
        ordering = ["capability", "normalized_skill"]


class CapabilityScore(models.Model):
    """Raw skill count per capability dimension for an analysis."""
    analysis   = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="capability_scores")
    capability = models.CharField(max_length=100)
    raw_count  = models.IntegerField(default=0)

    class Meta:
        ordering = ["capability"]


class Recommendation(models.Model):
    """A calibration recommendation generated for an analysis."""
    SEVERITY_CHOICES = [("High", "High"), ("Medium", "Medium"), ("Low", "Low")]
    TYPE_CHOICES     = [
        ("keep", "Keep"), ("prefer", "Prefer"), ("relax", "Relax"),
        ("remove", "Remove"), ("structure", "Structure"),
    ]
    analysis        = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="recommendations")
    signal          = models.TextField()
    evidence        = models.TextField(blank=True)
    recommendation  = models.TextField()
    reason          = models.TextField(blank=True)
    severity        = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="Medium")
    rec_type        = models.CharField(max_length=20, choices=TYPE_CHOICES, default="structure")

    class Meta:
        ordering = ["severity"]
