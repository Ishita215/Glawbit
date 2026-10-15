"""
core/nlp/taxonomy.py
====================
Central, single-source-of-truth skill taxonomy for DS-XRAY.
"""
from __future__ import annotations

CAPABILITY_DIMENSIONS: list[str] = [
    "Big Data",
    "Maths / Statistics",
    "Coding",
    "AI / ML",
    "Dashboard & Storytelling",
]

CAPABILITY_FEATURE_MAP: dict[str, str] = {
    "Big Data":                   "big_data_skills",
    "Maths / Statistics":         "maths-stats_skills",
    "Coding":                     "coding_skills",
    "AI / ML":                    "ai_and_ml_skills",
    "Dashboard & Storytelling":   "dashboard_and_storytelling_skills",
}

SKILL_TAXONOMY: list[dict] = [
    # ── Big Data ──────────────────────────────────────────────
    {"keywords": ["hadoop"],                        "normalized": "Hadoop",                    "capability": "Big Data"},
    {"keywords": ["spark", "apache spark"],         "normalized": "Apache Spark",              "capability": "Big Data"},
    {"keywords": ["pyspark"],                       "normalized": "PySpark",                   "capability": "Big Data"},
    {"keywords": ["hive"],                          "normalized": "Hive",                      "capability": "Big Data"},
    {"keywords": ["hbase"],                         "normalized": "HBase",                     "capability": "Big Data"},
    {"keywords": ["kafka", "apache kafka"],         "normalized": "Kafka",                     "capability": "Big Data"},
    {"keywords": ["flink", "apache flink"],         "normalized": "Flink",                     "capability": "Big Data"},
    {"keywords": ["data lake", "datalake"],         "normalized": "Data Lake",                 "capability": "Big Data"},
    {"keywords": ["databricks"],                    "normalized": "Databricks",                "capability": "Big Data"},
    {"keywords": ["distributed computing", "distributed systems", "distributed data"], "normalized": "Distributed Computing", "capability": "Big Data"},
    {"keywords": ["big data"],                      "normalized": "Big Data",                  "capability": "Big Data"},
    {"keywords": ["airflow", "apache airflow"],     "normalized": "Airflow",                   "capability": "Big Data"},
    {"keywords": ["data pipeline", "data pipelines", "etl pipeline", "elt pipeline", "etl"], "normalized": "Data Pipelines / ETL", "capability": "Big Data"},
    {"keywords": ["aws", "amazon web services", "s3", "ec2", "emr", "glue", "redshift"], "normalized": "AWS",                 "capability": "Big Data"},
    {"keywords": ["gcp", "google cloud", "bigquery", "dataflow"], "normalized": "GCP / BigQuery",       "capability": "Big Data"},
    {"keywords": ["azure", "azure databricks", "azure ml", "azure data factory"], "normalized": "Azure",               "capability": "Big Data"},
    {"keywords": ["snowflake"],                     "normalized": "Snowflake",                 "capability": "Big Data"},
    {"keywords": ["docker"],                        "normalized": "Docker",                    "capability": "Big Data"},
    {"keywords": ["kubernetes", "k8s"],             "normalized": "Kubernetes",                "capability": "Big Data"},
    # ── Maths / Statistics ────────────────────────────────────
    {"keywords": ["statistics", "statistical analysis", "statistical modelling", "statistical modeling", "statistical methods"], "normalized": "Statistics", "capability": "Maths / Statistics"},
    {"keywords": ["probability", "probabilistic"],  "normalized": "Probability",               "capability": "Maths / Statistics"},
    {"keywords": ["mathematics", "maths", "mathematical"], "normalized": "Mathematics",        "capability": "Maths / Statistics"},
    {"keywords": ["hypothesis testing", "a/b testing", "ab testing", "a-b testing", "experimentation", "significance testing"], "normalized": "Hypothesis Testing / A-B Testing", "capability": "Maths / Statistics"},
    {"keywords": ["regression", "regression analysis", "linear regression", "logistic regression", "regression modelling"], "normalized": "Regression Analysis", "capability": "Maths / Statistics"},
    {"keywords": ["time series", "time-series", "forecasting"], "normalized": "Time Series / Forecasting", "capability": "Maths / Statistics"},
    {"keywords": ["bayesian", "bayes", "bayesian inference"], "normalized": "Bayesian Methods", "capability": "Maths / Statistics"},
    {"keywords": ["anova"],                         "normalized": "ANOVA",                     "capability": "Maths / Statistics"},
    {"keywords": ["experimental design", "designed experiment"], "normalized": "Experimental Design", "capability": "Maths / Statistics"},
    {"keywords": ["optimization", "optimisation", "mathematical optimization"], "normalized": "Mathematical Optimization", "capability": "Maths / Statistics"},
    {"keywords": ["linear algebra", "calculus"],    "normalized": "Linear Algebra / Calculus", "capability": "Maths / Statistics"},
    {"keywords": ["causal inference", "causality"], "normalized": "Causal Inference",          "capability": "Maths / Statistics"},
    # ── Coding ────────────────────────────────────────────────
    {"keywords": ["python", "python programming", "python developer", "python scripting"], "normalized": "Python",          "capability": "Coding"},
    {"keywords": ["r programming", "r language", "r studio", "rstudio"],                  "normalized": "R",               "capability": "Coding"},
    {"keywords": ["sql", "mysql", "postgresql", "postgres", "t-sql", "tsql", "pl/sql", "plsql", "oracle sql"], "normalized": "SQL", "capability": "Coding"},
    {"keywords": ["pandas"],                        "normalized": "Pandas",                    "capability": "Coding"},
    {"keywords": ["numpy"],                         "normalized": "NumPy",                     "capability": "Coding"},
    {"keywords": ["scikit-learn", "sklearn", "scikit learn"],  "normalized": "Scikit-Learn",   "capability": "Coding"},
    {"keywords": ["java"],                          "normalized": "Java",                      "capability": "Coding"},
    {"keywords": ["scala"],                         "normalized": "Scala",                     "capability": "Coding"},
    {"keywords": ["c++", "cpp"],                    "normalized": "C++",                       "capability": "Coding"},
    {"keywords": ["javascript", "js"],              "normalized": "JavaScript",                "capability": "Coding"},
    {"keywords": ["programming", "coding", "software development", "software engineering"], "normalized": "Programming", "capability": "Coding"},
    {"keywords": ["git", "version control", "github", "gitlab"], "normalized": "Git / Version Control", "capability": "Coding"},
    {"keywords": ["api", "rest api", "restful", "flask", "fastapi"],                      "normalized": "API / Web Framework", "capability": "Coding"},
    {"keywords": ["data wrangling", "data manipulation", "data munging"],                 "normalized": "Data Wrangling",  "capability": "Coding"},
    {"keywords": ["matlab"],                        "normalized": "MATLAB",                    "capability": "Coding"},
    {"keywords": ["sas", "sas programming"],        "normalized": "SAS",                       "capability": "Coding"},
    {"keywords": ["excel", "ms excel", "microsoft excel", "vba"],                         "normalized": "Excel / VBA",     "capability": "Coding"},
    # ── AI / ML ───────────────────────────────────────────────
    {"keywords": ["machine learning", "machine-learning", "ml model", "ml models", "ml pipeline", "ml algorithms"], "normalized": "Machine Learning", "capability": "AI / ML"},
    {"keywords": ["deep learning", "deep neural", "dnn"],                                 "normalized": "Deep Learning",   "capability": "AI / ML"},
    {"keywords": ["artificial intelligence", "ai"],                                       "normalized": "Artificial Intelligence", "capability": "AI / ML"},
    {"keywords": ["nlp", "natural language processing", "text mining", "text analytics", "language model"], "normalized": "NLP", "capability": "AI / ML"},
    {"keywords": ["computer vision", "image recognition", "image processing", "object detection"], "normalized": "Computer Vision", "capability": "AI / ML"},
    {"keywords": ["neural network", "neural networks", "ann", "cnn", "rnn", "lstm", "transformer"], "normalized": "Neural Networks", "capability": "AI / ML"},
    {"keywords": ["tensorflow", "tf"],              "normalized": "TensorFlow",               "capability": "AI / ML"},
    {"keywords": ["pytorch", "torch"],              "normalized": "PyTorch",                   "capability": "AI / ML"},
    {"keywords": ["keras"],                         "normalized": "Keras",                     "capability": "AI / ML"},
    {"keywords": ["xgboost", "xgb", "gradient boosting", "catboost", "lightgbm", "lgbm", "boosting"], "normalized": "Gradient Boosting / XGBoost", "capability": "AI / ML"},
    {"keywords": ["random forest", "decision tree", "decision trees"],                    "normalized": "Tree-Based Models", "capability": "AI / ML"},
    {"keywords": ["predictive modeling", "predictive modelling", "predictive analytics"], "normalized": "Predictive Modelling", "capability": "AI / ML"},
    {"keywords": ["classification", "binary classification", "multiclass"],               "normalized": "Classification",  "capability": "AI / ML"},
    {"keywords": ["clustering", "unsupervised learning", "k-means", "kmeans"],            "normalized": "Clustering / Unsupervised", "capability": "AI / ML"},
    {"keywords": ["recommendation", "recommender system", "collaborative filtering"],     "normalized": "Recommendation Systems", "capability": "AI / ML"},
    {"keywords": ["feature engineering", "feature selection", "feature extraction"],      "normalized": "Feature Engineering", "capability": "AI / ML"},
    {"keywords": ["model evaluation", "model performance", "model validation", "cross-validation", "cross validation", "hyperparameter tuning"], "normalized": "Model Evaluation / Tuning", "capability": "AI / ML"},
    {"keywords": ["generative ai", "gen ai", "genai", "llm", "large language model", "chatgpt", "gpt", "prompt engineering"], "normalized": "Generative AI / LLMs", "capability": "AI / ML"},
    {"keywords": ["mlops", "ml ops", "model deployment", "model serving", "production ml", "mlflow"], "normalized": "MLOps / Model Deployment", "capability": "AI / ML"},
    # ── Dashboard & Storytelling ──────────────────────────────
    {"keywords": ["tableau"],                       "normalized": "Tableau",                   "capability": "Dashboard & Storytelling"},
    {"keywords": ["power bi", "powerbi", "power-bi"], "normalized": "Power BI",               "capability": "Dashboard & Storytelling"},
    {"keywords": ["data visualization", "data visualisation", "visualization", "visualisation", "data viz"], "normalized": "Data Visualization", "capability": "Dashboard & Storytelling"},
    {"keywords": ["dashboard", "dashboards", "interactive dashboard"],                    "normalized": "Dashboards",      "capability": "Dashboard & Storytelling"},
    {"keywords": ["business intelligence", "bi tools", "bi reporting"],                   "normalized": "Business Intelligence", "capability": "Dashboard & Storytelling"},
    {"keywords": ["storytelling", "data storytelling", "narrative"],                      "normalized": "Data Storytelling", "capability": "Dashboard & Storytelling"},
    {"keywords": ["communication", "communicate", "presenting", "presentation", "communicating insights"], "normalized": "Communication / Presentation", "capability": "Dashboard & Storytelling"},
    {"keywords": ["stakeholder", "stakeholders", "stakeholder management", "executive reporting"], "normalized": "Stakeholder Management", "capability": "Dashboard & Storytelling"},
    {"keywords": ["reporting", "report", "analytical reporting"],                         "normalized": "Reporting",       "capability": "Dashboard & Storytelling"},
    {"keywords": ["looker", "google data studio", "metabase", "qlik", "qlikview", "qlik sense", "microstrategy"], "normalized": "BI Tools (Looker / Qlik)", "capability": "Dashboard & Storytelling"},
    {"keywords": ["matplotlib", "seaborn", "plotly", "ggplot", "bokeh", "altair"],       "normalized": "Visualisation Libraries", "capability": "Dashboard & Storytelling"},
]

ROLE_FAMILY_PATTERNS: dict[str, list[str]] = {
    "Data Scientist": [
        "Machine Learning", "Statistics", "Predictive Modelling", "Python",
        "Feature Engineering", "Model Evaluation / Tuning", "Deep Learning", "NLP",
    ],
    "Data Analyst": [
        "SQL", "Reporting", "Data Visualization", "Dashboards", "Excel / VBA",
        "Business Intelligence", "Tableau", "Power BI",
    ],
    "ML Engineer": [
        "MLOps / Model Deployment", "Docker", "Kubernetes", "API / Web Framework",
        "Scikit-Learn", "TensorFlow", "PyTorch", "Git / Version Control",
    ],
    "Data Engineer": [
        "Apache Spark", "Hadoop", "Airflow", "Data Pipelines / ETL", "AWS",
        "GCP / BigQuery", "Azure", "Snowflake", "Kafka", "Databricks",
    ],
    "BI Analyst": [
        "Power BI", "Tableau", "Business Intelligence", "Dashboards", "Reporting",
        "Stakeholder Management", "Data Visualization", "BI Tools (Looker / Qlik)",
    ],
    "DevOps / Cloud": [
        "Docker", "Kubernetes", "AWS", "Azure", "GCP / BigQuery",
        "Git / Version Control", "API / Web Framework",
    ],
}
