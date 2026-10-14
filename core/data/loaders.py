"""
core/data/loaders.py
====================
Centralised dataset loaders.  Each loader:
  - Returns a clean DataFrame (or empty DataFrame on failure).
  - Logs the reason when rows are dropped.
  - Caches the result at module level so the CSV/XLSX is read only once per
    process lifetime.  Django management commands that need a fresh load can
    call clear_caches().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("dsxray.data")

BASE_DIR = Path(__file__).resolve().parent.parent.parent   # project root

DATA_DIR = BASE_DIR / "data"

# ── file paths ────────────────────────────────────────────────
ANALYTICS_JOBS_PATH = DATA_DIR / "Analytics Jobs.csv"
DATASCIENCE_JOBS_PATH = DATA_DIR / "DataScience Jobs.csv"
JDS_TRAITS_PATH = DATA_DIR / "JDS Skill Traits.xlsx"
SDS_TRAITS_PATH = DATA_DIR / "SDS Personality Traits.xlsx"

# ── module-level caches ───────────────────────────────────────
_analytics_jobs_cache: Optional[pd.DataFrame] = None
_datascience_jobs_cache: Optional[pd.DataFrame] = None
_jds_traits_cache: Optional[pd.DataFrame] = None
_sds_traits_cache: Optional[pd.DataFrame] = None


def clear_caches() -> None:
    """Invalidate all cached DataFrames (e.g. after retraining)."""
    global _analytics_jobs_cache, _datascience_jobs_cache
    global _jds_traits_cache, _sds_traits_cache
    _analytics_jobs_cache = None
    _datascience_jobs_cache = None
    _jds_traits_cache = None
    _sds_traits_cache = None
    logger.info("Dataset caches cleared.")


def load_analytics_jobs(force_reload: bool = False) -> pd.DataFrame:
    """
    Load Analytics Jobs.csv (15,841 rows).
    Returns all rows; callers may filter by role.
    Preserves original values in *_raw companion columns.
    """
    global _analytics_jobs_cache
    if _analytics_jobs_cache is not None and not force_reload:
        return _analytics_jobs_cache

    if not ANALYTICS_JOBS_PATH.exists():
        logger.error("Analytics Jobs.csv not found at %s", ANALYTICS_JOBS_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_csv(ANALYTICS_JOBS_PATH, low_memory=False)
        logger.info("Loaded Analytics Jobs: %d rows", len(df))

        # Preserve originals
        df["key_skills_raw"] = df["key_skills"].copy()
        df["job_description_raw"] = df["job_description"].copy()

        # Normalise text columns
        for col in ["job_desig", "job_description", "key_skills", "job_type", "location"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Drop rows where both key_skills and job_description are empty
        before = len(df)
        empty_mask = (df["key_skills"].str.len() == 0) & (df["job_description"].str.len() == 0)
        if empty_mask.sum():
            logger.info(
                "Dropping %d rows with no key_skills AND no job_description",
                empty_mask.sum(),
            )
        df = df[~empty_mask].reset_index(drop=True)
        after = len(df)
        logger.info("Analytics Jobs usable rows: %d (dropped %d)", after, before - after)

        _analytics_jobs_cache = df
        return df

    except Exception as exc:
        logger.exception("Failed to load Analytics Jobs: %s", exc)
        return pd.DataFrame()


def load_datascience_jobs(force_reload: bool = False) -> pd.DataFrame:
    """
    Load DataScience Jobs.csv (1,602 rows).
    Columns: reference_no, company_name, job_title, min_experience,
             avg_salary, min_salary, max_salary, num_of_jobs
    """
    global _datascience_jobs_cache
    if _datascience_jobs_cache is not None and not force_reload:
        return _datascience_jobs_cache

    if not DATASCIENCE_JOBS_PATH.exists():
        logger.error("DataScience Jobs.csv not found at %s", DATASCIENCE_JOBS_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATASCIENCE_JOBS_PATH, low_memory=False)
        logger.info("Loaded DataScience Jobs: %d rows", len(df))

        for col in ["company_name", "job_title"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        _datascience_jobs_cache = df
        return df

    except Exception as exc:
        logger.exception("Failed to load DataScience Jobs: %s", exc)
        return pd.DataFrame()


def load_jds_traits(force_reload: bool = False) -> pd.DataFrame:
    """
    Load JDS Skill Traits.xlsx (139 rows).
    Columns: id, big_data_skills, maths-stats_skills, coding_skills,
             ai_and_ml_skills, dashboard_and_storytelling_skills,
             salary_hike_high_or_low
    Target: salary_hike_high_or_low (0/1)
    """
    global _jds_traits_cache
    if _jds_traits_cache is not None and not force_reload:
        return _jds_traits_cache

    if not JDS_TRAITS_PATH.exists():
        logger.error("JDS Skill Traits.xlsx not found at %s", JDS_TRAITS_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_excel(JDS_TRAITS_PATH, engine="openpyxl")
        logger.info("Loaded JDS Skill Traits: %d rows", len(df))

        # Validate expected columns
        required = [
            "big_data_skills", "maths-stats_skills", "coding_skills",
            "ai_and_ml_skills", "dashboard_and_storytelling_skills",
            "salary_hike_high_or_low",
        ]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            logger.error("JDS Skill Traits missing columns: %s", missing_cols)
            return pd.DataFrame()

        # Drop rows with missing target
        before = len(df)
        df = df.dropna(subset=["salary_hike_high_or_low"]).reset_index(drop=True)
        if len(df) < before:
            logger.info("Dropped %d rows with missing salary_hike_high_or_low", before - len(df))

        # Ensure target is int
        df["salary_hike_high_or_low"] = df["salary_hike_high_or_low"].astype(int)

        _jds_traits_cache = df
        return df

    except Exception as exc:
        logger.exception("Failed to load JDS Skill Traits: %s", exc)
        return pd.DataFrame()


def load_sds_traits(force_reload: bool = False) -> pd.DataFrame:
    """
    Load SDS Personality Traits.xlsx (161 rows).
    Columns: id, neuroticism, extraversion, openness_to_experience,
             agreeableness, conscientiousness, success_classification_high_low
    NOTE: This dataset must NOT be used for automated hiring decisions.
          It is available for descriptive / experimental analysis only.
    """
    global _sds_traits_cache
    if _sds_traits_cache is not None and not force_reload:
        return _sds_traits_cache

    if not SDS_TRAITS_PATH.exists():
        logger.warning("SDS Personality Traits.xlsx not found at %s", SDS_TRAITS_PATH)
        return pd.DataFrame()

    try:
        df = pd.read_excel(SDS_TRAITS_PATH, engine="openpyxl")
        logger.info("Loaded SDS Personality Traits: %d rows", len(df))

        # Standardise column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]

        _sds_traits_cache = df
        return df

    except Exception as exc:
        logger.exception("Failed to load SDS Personality Traits: %s", exc)
        return pd.DataFrame()


def dataset_health_report() -> dict:
    """
    Return a lightweight health-check dict for all four datasets.
    Does NOT use caches so every call re-inspects the files.
    """
    import time

    report: dict[str, dict] = {}

    datasets = [
        ("Analytics Jobs",       ANALYTICS_JOBS_PATH,     "csv"),
        ("DataScience Jobs",     DATASCIENCE_JOBS_PATH,   "csv"),
        ("JDS Skill Traits",     JDS_TRAITS_PATH,          "xlsx"),
        ("SDS Personality Traits", SDS_TRAITS_PATH,        "xlsx"),
    ]

    for name, path, fmt in datasets:
        entry: dict = {"file": str(path), "exists": path.exists()}
        if not path.exists():
            entry["rows"] = 0
            entry["columns"] = []
            entry["missing_values"] = {}
            entry["duplicate_count"] = 0
            entry["usable_rows"] = 0
            entry["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            report[name] = entry
            continue

        try:
            t0 = time.time()
            if fmt == "csv":
                df = pd.read_csv(path, low_memory=False)
            else:
                df = pd.read_excel(path, engine="openpyxl")
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            entry["rows"] = len(df)
            entry["columns"] = list(df.columns)
            entry["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}
            entry["duplicate_count"] = int(df.duplicated().sum())
            entry["usable_rows"] = len(df)   # callers may refine this
            entry["load_time_ms"] = elapsed_ms
            entry["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        except Exception as exc:
            entry["error"] = str(exc)

        report[name] = entry

    return report
