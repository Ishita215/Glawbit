from django.test import TestCase


class AnalyzeApiTests(TestCase):
    job_description = (
        "Data Scientist with Python, SQL, pandas, scikit-learn, "
        "machine learning, regression analysis, Tableau and communication skills."
    )

    def test_empty_job_description_is_rejected(self):
        response = self.client.post("/analyze/", {"job_description": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_analysis_returns_nested_scorecard_values(self):
        response = self.client.post(
            "/analyze/",
            {"job_description": self.job_description},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        scorecard = payload["result"]["scorecard"]
        self.assertIn("score", scorecard["composite"])
        self.assertIn("score", scorecard["outcome_fit"])
        self.assertIn("score", scorecard["demand_alignment"])
        self.assertIn("score", scorecard["structural_balance"])

    def test_market_matrix_contains_capability_rows_and_jd_shares(self):
        response = self.client.post(
            "/analyze/",
            {"job_description": self.job_description},
        )

        matrix = response.json()["result"]["market_outcome_matrix"]["matrix"]
        self.assertEqual(len(matrix), 5)
        self.assertTrue(all("capability" in row for row in matrix))
        self.assertAlmostEqual(sum(row["jd_pct"] for row in matrix), 100.0)
