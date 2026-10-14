# DS-XRAY — Data Scientist Hiring Calibration Engine

A Django-based hackathon prototype that audits a Data Scientist job description against two evidence layers from the supplied SAS hackathon data:

1. **Success evidence** — the five JDS capability dimensions and the `salary_hike_high_or_low` outcome.
2. **Market evidence** — Data Scientist postings identified from `Analytics Jobs.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000/

## API

`POST /analyze/` with form field `job_description` returns JSON containing:
- overall score
- success fit
- market fit
- skill balance
- five capability scores
- matched skill terms
- calibration gaps

## Important analytical note

The success model is a prototype trained on only 139 labeled observations. The reported cross-validated ROC-AUC is useful as a hackathon signal, not proof of causality or production validity. The app must not be used to screen or reject candidates based on personality or inferred characteristics.
