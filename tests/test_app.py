"""Smoke tests for inference helpers and the Streamlit UI; synthetic data only."""

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from inference import load_model, predict_review, validate_review
from ml_baseline import CLASS_LABELS, create_nb_pipeline

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["great tasty love", "awful stale bad", "okay fine average",
         "love great yum", "bad gross awful", "fine okay meh"]
LABELS = ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral"]


def fitted_model():
    return create_nb_pipeline("tfidf").fit(TEXTS, LABELS)


class InferenceTests(unittest.TestCase):
    def test_predict_returns_label_and_ordered_probabilities(self):
        result = predict_review(fitted_model(), "  great love  ")
        self.assertEqual(result["label"], "Positive")
        self.assertEqual(list(result["probabilities"]), list(CLASS_LABELS))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_validate_rejects_blank_and_long(self):
        for bad in ("", "   ", None, "x" * 5001):
            with self.subTest(bad=bad if bad is None else len(bad)), self.assertRaises(ValueError):
                validate_review(bad)

    def test_missing_model_returns_none(self):
        self.assertIsNone(load_model(Path(tempfile.gettempdir()) / "no_such_model.joblib"))


class AppTests(unittest.TestCase):
    def run_app(self, model_file):
        patcher = mock.patch.dict(os.environ, {"SENTIMENT_MODEL_PATH": str(model_file)})
        patcher.start()
        self.addCleanup(patcher.stop)  # keep the path set across reruns
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_without_model_shows_notice(self):
        at = self.run_app(Path(tempfile.gettempdir()) / "no_such_model.joblib")
        self.assertFalse(at.exception)
        self.assertIn("No trained model found", at.info[0].value)

    def test_with_model_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted_model(), path)
            at = self.run_app(path)
            at.text_area(key="review").input("awful stale bad")
            at.button[-1].click().run()  # form submit button is last
            self.assertFalse(at.exception)
            self.assertIn("Negative", [s.value for s in at.subheader])


if __name__ == "__main__":
    unittest.main()
