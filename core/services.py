import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml" / "success_model.joblib"
META_PATH = BASE_DIR / "ml" / "model_meta.json"
MARKET_DATA_PATH = BASE_DIR / "data" / "Analytics Jobs.csv"


CAPABILITIES = [
    "Big Data",
    "Maths / Statistics",
    "Coding",
    "AI / ML",
    "Dashboard & Storytelling",
]


MODEL_FEATURES = [
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
]


SKILL_RULES = {

    "Big Data": [
        "big data",
        "hadoop",
        "spark",
        "pyspark",
        "hive",
        "kafka",
        "data lake",
        "distributed computing",
        "distributed systems",
        "large datasets",
        "large data",
        "databricks",
    ],

    "Maths / Statistics": [
        "statistics",
        "statistical",
        "probability",
        "mathematics",
        "maths",
        "mathematical",
        "hypothesis testing",
        "regression analysis",
        "statistical modelling",
        "statistical modeling",
        "a/b testing",
        "experimentation",
        "time series",
        "anova",
        "bayesian",
    ],

    "Coding": [
        "python",
        "r programming",
        "r language",
        "java",
        "c++",
        "sql",
        "programming",
        "coding",
        "software development",
        "scikit-learn",
        "sklearn",
        "pandas",
        "numpy",
    ],

    "AI / ML": [
        "machine learning",
        "machine-learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "ml",
        "predictive modeling",
        "predictive modelling",
        "nlp",
        "natural language processing",
        "computer vision",
        "neural network",
        "neural networks",
        "tensorflow",
        "pytorch",
        "xgboost",
        "classification",
        "clustering",
    ],

    "Dashboard & Storytelling": [
        "data visualization",
        "data visualisation",
        "visualization",
        "visualisation",
        "dashboard",
        "dashboards",
        "tableau",
        "power bi",
        "powerbi",
        "business intelligence",
        "storytelling",
        "communicate insights",
        "communication",
        "stakeholder",
        "stakeholders",
        "reporting",
        "presentation",
    ],
}


def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def load_metadata():

    if not META_PATH.exists():
        return {}

    try:

        with open(
            META_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def extract_capabilities(job_description):

    text = job_description.lower()

    capability_scores = {}
    detected_skills = {}

    for capability, keywords in SKILL_RULES.items():

        matches = []

        for keyword in keywords:

            pattern = (
                r"\b"
                + re.escape(keyword.lower())
                + r"\b"
            )

            if re.search(pattern, text):
                matches.append(keyword)

        score = min(
            len(set(matches)),
            5
        )

        capability_scores[capability] = score

        detected_skills[capability] = sorted(
            set(matches)
        )

    return (
        capability_scores,
        detected_skills
    )


def normalize_capability_scores(
    capability_scores
):

    return {

        capability: round(
            (score / 5) * 100,
            1
        )

        for capability, score
        in capability_scores.items()
    }


def calculate_skill_balance(
    capability_scores
):

    values = np.array(
        list(
            capability_scores.values()
        ),
        dtype=float
    )

    if len(values) == 0:
        return 0.0

    if np.max(values) == 0:
        return 0.0

    mean_value = np.mean(values)
    std_value = np.std(values)

    balance = (
        100
        - (
            std_value
            / max(mean_value, 1)
            * 100
        )
    )

    return round(
        float(
            np.clip(
                balance,
                0,
                100
            )
        ),
        1
    )


def calculate_success_fit(
    capability_scores
):

    model = load_model()

    if model is None:
        return 0.0

    feature_values = {

        "big_data_skills":
            capability_scores.get(
                "Big Data",
                0
            ),

        "maths-stats_skills":
            capability_scores.get(
                "Maths / Statistics",
                0
            ),

        "coding_skills":
            capability_scores.get(
                "Coding",
                0
            ),

        "ai_and_ml_skills":
            capability_scores.get(
                "AI / ML",
                0
            ),

        "dashboard_and_storytelling_skills":
            capability_scores.get(
                "Dashboard & Storytelling",
                0
            ),
    }

    X = pd.DataFrame(
        [[
            feature_values["big_data_skills"],
            feature_values["maths-stats_skills"],
            feature_values["coding_skills"],
            feature_values["ai_and_ml_skills"],
            feature_values[
                "dashboard_and_storytelling_skills"
            ],
        ]],
        columns=MODEL_FEATURES
    )

    try:

        probability = (
            model
            .predict_proba(X)[0][1]
        )

        return round(
            float(
                probability * 100
            ),
            1
        )

    except Exception as error:

        print(
            "SUCCESS MODEL ERROR:",
            error
        )

        return 0.0


def load_market_data():

    if not MARKET_DATA_PATH.exists():

        print(
            "MARKET DATA NOT FOUND:",
            MARKET_DATA_PATH
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            MARKET_DATA_PATH
        )

        ds_mask = (
            df["job_desig"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "data scientist",
                regex=False
            )
        )

        return df[ds_mask].copy()

    except Exception as error:

        print(
            "MARKET DATA ERROR:",
            error
        )

        return pd.DataFrame()


def calculate_market_profile():

    df = load_market_data()

    if df.empty:

        return {
            capability: 0.0
            for capability in CAPABILITIES
        }

    combined_text = (
        df["key_skills"]
        .fillna("")
        .astype(str)
        + " "
        + df["job_description"]
        .fillna("")
        .astype(str)
    ).str.lower()

    market_profile = {}

    for capability, keywords in SKILL_RULES.items():

        matched_jobs = 0

        for text in combined_text:

            if any(
                keyword.lower() in text
                for keyword in keywords
            ):
                matched_jobs += 1

        percentage = (
            matched_jobs
            / len(df)
            * 100
        )

        market_profile[capability] = round(
            float(percentage),
            1
        )

    return market_profile


def calculate_market_fit(
    capability_scores
):

    market_profile = calculate_market_profile()

    if not market_profile:
        return 0.0

    jd_profile = normalize_capability_scores(
        capability_scores
    )

    differences = []

    for capability in CAPABILITIES:

        jd_value = jd_profile.get(
            capability,
            0
        )

        market_value = market_profile.get(
            capability,
            0
        )

        differences.append(
            abs(
                jd_value - market_value
            )
        )

    average_difference = np.mean(
        differences
    )

    market_fit = (
        100 - average_difference
    )

    return round(
        float(
            np.clip(
                market_fit,
                0,
                100
            )
        ),
        1
    )


def generate_recommendation(
    capability,
    status
):

    recommendations = {

        "Big Data": {

            "Overweighted":
                "Reduce broad Big Data requirements unless distributed data processing is essential to this specific role.",

            "Above Market":
                "Keep Big Data requirements focused on the datasets and infrastructure actually needed by the role.",

            "Underweighted":
                "Consider adding relevant large-scale data processing requirements if the role works with substantial datasets.",

            "Missing":
                "Consider adding large-scale data processing requirements if the role handles substantial datasets.",
        },

        "Maths / Statistics": {

            "Overweighted":
                "Avoid making advanced statistical knowledge broader than the actual analytical work requires.",

            "Above Market":
                "Keep statistical requirements focused on the modelling, experimentation and analysis needed for the role.",

            "Underweighted":
                "Consider adding statistical modelling, experimentation or hypothesis testing where relevant.",

            "Missing":
                "Consider adding statistical reasoning or experimental analysis requirements.",
        },

        "Coding": {

            "Overweighted":
                "Avoid turning the Data Scientist role into a broad software-engineering requirement.",

            "Above Market":
                "Keep programming requirements focused on data analysis, modelling and production needs.",

            "Underweighted":
                "Consider adding explicit programming or SQL requirements needed for analytical work.",

            "Missing":
                "Consider adding core programming requirements such as Python, R or SQL.",
        },

        "AI / ML": {

            "Overweighted":
                "Reduce unnecessary AI/ML framework requirements unless advanced modelling is central to the role.",

            "Above Market":
                "Keep AI/ML as a core requirement, but avoid adding unnecessary framework-specific requirements.",

            "Underweighted":
                "Consider strengthening machine-learning requirements if predictive modelling is central to the role.",

            "Missing":
                "Consider adding relevant machine-learning or predictive modelling requirements.",
        },

        "Dashboard & Storytelling": {

            "Overweighted":
                "Keep stakeholder communication, but avoid making dashboard or BI tooling a core Data Scientist requirement.",

            "Above Market":
                "Keep visualization and communication skills focused on explaining analytical results.",

            "Underweighted":
                "Consider adding requirements around communicating model results and presenting analytical insights.",

            "Missing":
                "Consider adding a requirement for communicating analytical findings to stakeholders.",
        },
    }

    return recommendations.get(
        capability,
        {}
    ).get(
        status,
        "Review this capability against the specific responsibilities of the role."
    )


def generate_market_insights(
    capability_scores
):

    market_profile = calculate_market_profile()

    jd_profile = normalize_capability_scores(
        capability_scores
    )

    insights = []
    calibration = {}

    for capability in CAPABILITIES:

        jd_value = jd_profile.get(
            capability,
            0
        )

        market_value = market_profile.get(
            capability,
            0
        )

        difference = round(
            jd_value - market_value,
            1
        )

        if jd_value == 0:

            status = "Missing"
            severity = "High"

        elif difference >= 40:

            status = "Overweighted"
            severity = "High"

        elif difference >= 20:

            status = "Above Market"
            severity = "Medium"

        elif difference <= -20:

            status = "Underweighted"
            severity = "Medium"

        else:

            status = "Aligned"
            severity = "Low"

        recommendation = generate_recommendation(
            capability,
            status
        )

        calibration[capability] = {

            "jd_score": jd_value,

            "market_score": market_value,

            "difference": difference,

            "status": status,

            "severity": severity,

            "recommendation":
                recommendation,
        }

        if status != "Aligned":

            insights.append({

                "capability":
                    capability,

                "status":
                    status,

                "severity":
                    severity,

                "jd_score":
                    jd_value,

                "market_score":
                    market_value,

                "difference":
                    difference,

                "message":
                    (
                        f"{capability} is "
                        f"{status.lower()} compared "
                        f"with the Data Scientist "
                        f"market benchmark."
                    ),

                "recommendation":
                    recommendation,
            })

    severity_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    insights.sort(
        key=lambda item:
        severity_order.get(
            item["severity"],
            3
        )
    )

    return {

        "market_profile":
            market_profile,

        "calibration":
            calibration,

        "insights":
            insights,
    }


def generate_gaps(
    capability_scores
):

    gaps = []

    for capability, score in (
        capability_scores.items()
    ):

        if score == 0:

            gaps.append({

                "capability":
                    capability,

                "severity":
                    "High",

                "message":
                    (
                        f"No clear "
                        f"{capability} requirement "
                        f"was detected."
                    ),
            })

        elif score == 1:

            gaps.append({

                "capability":
                    capability,

                "severity":
                    "Medium",

                "message":
                    (
                        f"{capability} appears "
                        f"weakly represented "
                        f"in the job description."
                    ),
            })

    return gaps


def calculate_overall_score(
    success_fit,
    market_fit,
    skill_balance
):

    score = (
        success_fit * 0.50
        + market_fit * 0.30
        + skill_balance * 0.20
    )

    return round(
        float(score),
        1
    )


# ============================================================
# CALIBRATED JD GENERATOR
# ============================================================

def calibrate_job_description(
    job_description
):

    if not job_description or not job_description.strip():

        raise ValueError(
            "Job description cannot be empty."
        )

    original = job_description.strip()

    capability_scores, detected_skills = (
        extract_capabilities(original)
    )

    market_insights = generate_market_insights(
        capability_scores
    )

    calibration = market_insights[
        "calibration"
    ]

    # --------------------------------------------------------
    # Detect likely title
    # --------------------------------------------------------

    title = "Data Scientist"

    title_match = re.search(
        r"(senior\s+|junior\s+|lead\s+|principal\s+)?"
        r"data scientist",
        original,
        re.IGNORECASE
    )

    if title_match:

        title = title_match.group(0).strip().title()

    # --------------------------------------------------------
    # Core skills
    # --------------------------------------------------------

    core_skills = []

    for capability in [
        "Coding",
        "AI / ML",
        "Maths / Statistics",
    ]:

        skills = detected_skills.get(
            capability,
            []
        )

        for skill in skills:

            if skill not in core_skills:

                core_skills.append(skill)

    # --------------------------------------------------------
    # Keep a reasonable number of skills
    # --------------------------------------------------------

    core_skills = core_skills[:10]

    if not core_skills:

        core_skills = [
            "Python",
            "SQL",
            "Machine Learning",
            "Statistical Analysis",
        ]

    # --------------------------------------------------------
    # Big Data
    # --------------------------------------------------------

    big_data_status = calibration[
        "Big Data"
    ]["status"]

    big_data_skills = detected_skills.get(
        "Big Data",
        []
    )

    if big_data_status in [
        "Overweighted",
        "Above Market"
    ]:

        big_data_section = (
            "Experience working with large-scale "
            "or distributed data is valuable where "
            "required by the business problem."
        )

    elif big_data_skills:

        big_data_section = (
            "Experience with large-scale data "
            "processing and distributed data tools "
            "is preferred."
        )

    else:

        big_data_section = (
            "Ability to work with large datasets "
            "where relevant to the role."
        )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    dashboard_status = calibration[
        "Dashboard & Storytelling"
    ]["status"]

    if dashboard_status == "Overweighted":

        communication_section = (
            "Communicate analytical findings clearly "
            "to technical and business stakeholders. "
            "Visualization should support analytical "
            "storytelling rather than serve as the "
            "primary responsibility of the role."
        )

    else:

        communication_section = (
            "Communicate analytical findings clearly "
            "through appropriate visualizations, "
            "presentations and stakeholder discussions."
        )

    # --------------------------------------------------------
    # Build calibrated JD
    # --------------------------------------------------------

    calibrated_jd = f"""JOB TITLE
{title}

ABOUT THE ROLE
We are looking for a Data Scientist to solve business problems using data, statistical analysis and machine learning. The role will involve translating analytical problems into measurable solutions and communicating findings to stakeholders.

KEY RESPONSIBILITIES
• Analyze structured and unstructured data to identify meaningful patterns and insights.
• Develop, evaluate and improve statistical and machine-learning models.
• Apply appropriate statistical reasoning, experimentation and validation techniques.
• Work with engineering and business teams to translate analytical findings into practical outcomes.
• Communicate model results, assumptions and recommendations clearly to technical and non-technical stakeholders.
• Work with large datasets and production data environments where required by the business problem.

REQUIRED SKILLS
• Strong programming ability in Python, SQL or relevant analytical programming languages.
• Practical experience with machine learning and predictive modelling.
• Strong understanding of statistics, probability and analytical methods.
• Ability to evaluate model performance and communicate analytical trade-offs.
• Strong problem-solving and analytical thinking.

TECHNICAL SKILLS IDENTIFIED FROM THE ORIGINAL JD
{", ".join(core_skills)}

DATA & SCALE
{big_data_section}

COMMUNICATION & STORYTELLING
{communication_section}

PREFERRED EXPERIENCE
• Experience solving real-world business problems using data.
• Experience collaborating with cross-functional stakeholders.
• Experience taking analytical solutions from exploration through evaluation and delivery.

NOTE
This calibrated version prioritizes capabilities using the DS-XRAY market benchmark. Specific requirements should still be validated against the actual responsibilities, data environment and business context of the role.
"""

    return {

        "original_jd":
            original,

        "calibrated_jd":
            calibrated_jd.strip(),

        "market_insights":
            market_insights,

        "changes": [

            item
            for item in market_insights[
                "insights"
            ]
        ],

        "capability_scores":
            normalize_capability_scores(
                capability_scores
            ),
    }


# ============================================================
# MAIN ANALYSIS ENGINE
# ============================================================

def analyze_job_description(
    job_description
):

    if not job_description:

        raise ValueError(
            "Job description cannot be empty."
        )

    job_description = (
        job_description.strip()
    )

    if not job_description:

        raise ValueError(
            "Job description cannot be empty."
        )

    capability_scores, detected_skills = (
        extract_capabilities(
            job_description
        )
    )

    normalized_scores = (
        normalize_capability_scores(
            capability_scores
        )
    )

    success_fit = calculate_success_fit(
        capability_scores
    )

    market_fit = calculate_market_fit(
        capability_scores
    )

    skill_balance = calculate_skill_balance(
        capability_scores
    )

    overall_score = calculate_overall_score(
        success_fit,
        market_fit,
        skill_balance
    )

    gaps = generate_gaps(
        capability_scores
    )

    market_insights = (
        generate_market_insights(
            capability_scores
        )
    )

    return {

        "overall_score":
            overall_score,

        "success_fit":
            success_fit,

        "market_fit":
            market_fit,

        "skill_balance":
            skill_balance,

        "capability_scores":
            normalized_scores,

        "raw_capability_scores":
            capability_scores,

        "detected_skills":
            detected_skills,

        "gaps":
            gaps,

        "market_insights":
            market_insights,

        "model": {

            "name":
                "DS-XRAY Success Alignment Model",

            "type":
                "Logistic Regression",

            "features":
                MODEL_FEATURES,
        },
    }