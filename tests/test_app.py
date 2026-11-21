"""Smoke tests for inference helpers and the Streamlit UI; synthetic data only."""

import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import joblib
import streamlit as st
from streamlit.testing.v1 import AppTest

from inference import load_model, model_paths, predict_review, validate_review
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

    def test_model_paths_lists_joblib_files_default_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("b.joblib", "sentiment_pipeline.joblib", "a.joblib", ".gitkeep", "notes.txt"):
                (root / name).touch()
            (root / "dir.joblib").mkdir()
            env = {"SENTIMENT_MODEL_DIR": tmp}
            with mock.patch.dict(os.environ, env):
                os.environ.pop("SENTIMENT_MODEL_PATH", None)
                names = [p.name for p in model_paths()]
            self.assertEqual(names, ["sentiment_pipeline.joblib", "a.joblib", "b.joblib"])

    def test_model_path_override_limits_to_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.joblib").touch()
            chosen = Path(tmp) / "b.joblib"
            chosen.touch()
            env = {"SENTIMENT_MODEL_DIR": tmp, "SENTIMENT_MODEL_PATH": str(chosen)}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(model_paths(), [chosen])
                chosen.unlink()
                self.assertEqual(model_paths(), [])


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

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

    def test_model_created_after_first_run_enables_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            at = self.run_app(path)
            self.assertTrue(at.button[-1].disabled)
            joblib.dump(fitted_model(), path)
            at.run()
            self.assertFalse(at.exception)
            self.assertFalse(at.button[-1].disabled)
            self.assertEqual(len(at.info), 0)
            # A second session also sees the newly available artifact.
            fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(fresh.exception)
            self.assertFalse(fresh.button[-1].disabled)
            at.text_area(key="review").input("great tasty love")
            at.button[-1].click().run()
            self.assertIn("Positive", [s.value for s in at.subheader])

    def assert_model_refreshes(self, atomic):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted_model(), path)
            original = path.stat()
            at = self.run_app(path)
            at.text_area(key="review").input("great tasty love")
            at.button[-1].click().run()
            self.assertIn("Positive", [s.value for s in at.subheader])
            flipped = [{"Positive": "Negative", "Negative": "Positive"}.get(c, c)
                       for c in LABELS]
            replacement = create_nb_pipeline("tfidf").fit(TEXTS, flipped)
            destination = path.with_name("replacement.joblib") if atomic else path
            joblib.dump(replacement, destination)
            if atomic:
                destination.replace(path)
            # Even a preserved mtime must not keep serving the previous model.
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            at.button[-1].click().run()
            self.assertFalse(at.exception)
            self.assertIn("Negative", [s.value for s in at.subheader])

    def test_overwritten_model_is_reloaded(self):
        self.assert_model_refreshes(atomic=False)

    def test_atomically_replaced_model_is_reloaded(self):
        self.assert_model_refreshes(atomic=True)

    def test_unchanged_model_reuses_cached_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted_model(), path)
            with mock.patch("inference.load_model", wraps=load_model) as loader:
                at = self.run_app(path)
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(loader.call_count, 1)

    def test_removed_model_disables_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted_model(), path)
            at = self.run_app(path)
            self.assertFalse(at.button[-1].disabled)
            path.unlink()
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(at.button[-1].disabled)
            self.assertIn("No trained model found", at.info[0].value)


if __name__ == "__main__":
    unittest.main()
