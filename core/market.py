import pandas as pd
import os


CAPABILITIES = [
    "Big Data",
    "Maths / Statistics",
    "Coding",
    "AI / ML",
    "Dashboard & Storytelling",
]


SKILL_MAP = {
    "Big Data": [
        "hadoop",
        "spark",
        "hive",
        "hbase",
        "kafka",
        "big data",
        "databricks",
        "distributed computing",
        "data lake",
        "pyspark",
    ],

    "Maths / Statistics": [
        "statistics",
        "statistical",
        "probability",
        "hypothesis testing",
        "regression",
        "time series",
        "mathematical",
        "mathematics",
        "anova",
        "bayesian",
        "experimental design",
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
        "scikit-learn",
        "pandas",
        "numpy",
    ],

    "AI / ML": [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "ml",
        "nlp",
        "natural language processing",
        "computer vision",
        "predictive modeling",
        "tensorflow",
        "pytorch",
        "keras",
    ],

    "Dashboard & Storytelling": [
        "tableau",
        "power bi",
        "dashboard",
        "data visualization",
        "visualization",
        "storytelling",
        "business intelligence",
        "bi",
        "reporting",
        "presentation",
        "stakeholder",
    ],
}


def load_market_data():

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    csv_path = os.path.join(
        base_dir,
        "data",
        "Analytics Jobs.csv"
    )

    df = pd.read_csv(csv_path)

    # Data Scientist jobs
    ds_mask = (
        df["job_desig"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("data scientist", regex=False)
    )

    ds_jobs = df[ds_mask].copy()

    return ds_jobs


def calculate_market_profile():

    df = load_market_data()

    total_jobs = len(df)

    profile = {}

    combined_text = (
        df["key_skills"].fillna("").astype(str)
        + " "
        + df["job_description"].fillna("").astype(str)
    ).str.lower()

    for capability, keywords in SKILL_MAP.items():

        matched_jobs = 0

        for text in combined_text:

            if any(
                keyword.lower() in text
                for keyword in keywords
            ):
                matched_jobs += 1

        percentage = (
            matched_jobs / total_jobs * 100
            if total_jobs > 0
            else 0
        )

        profile[capability] = {
            "jobs": matched_jobs,
            "percentage": round(percentage, 1),
        }

    return {
        "total_data_scientist_jobs": total_jobs,
        "profile": profile,
    }


if __name__ == "__main__":

    result = calculate_market_profile()

    print("\nDS-XRAY MARKET BENCHMARK")
    print("=" * 40)

    print(
        "Data Scientist jobs:",
        result["total_data_scientist_jobs"]
    )

    for capability, data in result["profile"].items():

        print(
            f"{capability}: "
            f"{data['percentage']}% "
            f"({data['jobs']} jobs)"
        )


 