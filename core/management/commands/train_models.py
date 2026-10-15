"""
management/commands/train_models.py
=====================================
Usage:  python manage.py train_models

1. Loads JDS Skill Traits.xlsx
2. Cleans data
3. Trains Logistic Regression + comparison baselines
4. Stratified 5-Fold Cross-Validation
5. Compares models
6. Saves selected model (LR) + metadata to ml/
7. Prints a full report to stdout
"""
from django.core.management.base import BaseCommand

import logging
logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = "Train the DS-XRAY outcome-association model from JDS Skill Traits.xlsx"

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\n" + "=" * 60))
        self.stdout.write(self.style.MIGRATE_HEADING("DS-XRAY  |  Model Training"))
        self.stdout.write(self.style.MIGRATE_HEADING("=" * 60 + "\n"))

        try:
            from core.ml.outcome_model import train_and_evaluate
            report = train_and_evaluate()
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f"Training failed: {exc}"))
            raise

        # ── Print model comparison ────────────────────────
        self.stdout.write(self.style.SUCCESS("\nMODEL COMPARISON (Stratified 5-Fold ROC-AUC)"))
        self.stdout.write("-" * 60)

        for name in ["logistic_regression", "random_forest", "dummy"]:
            r = report[name]
            folds_str = "  ".join(f"{s:.4f}" for s in r["fold_scores"])
            self.stdout.write(
                f"  {name:<25}  mean={r['mean_auc']:.4f}  sd={r['std_auc']:.4f}"
            )
            self.stdout.write(f"    folds: {folds_str}")

        # ── Print selection reason ────────────────────────
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("SELECTED MODEL: Logistic Regression"))
        self.stdout.write(f"  Reason: {report['comparison_reason']}")

        # ── Print coefficients ────────────────────────────
        meta = report["model_meta"]
        self.stdout.write(self.style.SUCCESS("\nLOGISTIC REGRESSION COEFFICIENTS"))
        for feature, coef in meta.get("lr_coefficients", {}).items():
            self.stdout.write(f"  {feature:<40} {coef:+.4f}")

        # ── Class distribution ────────────────────────────
        dist = meta.get("class_distribution", {})
        self.stdout.write(self.style.SUCCESS("\nDATASET"))
        self.stdout.write(f"  n samples:         {meta.get('n_samples')}")
        self.stdout.write(f"  class high (1):    {dist.get('high')}")
        self.stdout.write(f"  class low  (0):    {dist.get('low')}")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Model saved to ml/success_model.joblib"))
        self.stdout.write(self.style.SUCCESS("Metadata saved to ml/model_meta.json"))
        self.stdout.write("")
