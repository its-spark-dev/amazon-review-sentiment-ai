"""Smoke tests for inference helpers and the Streamlit UI; synthetic data only."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

    def test_missing_model_dir_lists_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"SENTIMENT_MODEL_DIR": str(Path(tmp) / "absent")}):
            os.environ.pop("SENTIMENT_MODEL_PATH", None)
            self.assertEqual(model_paths(), [])

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


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.dict(os.environ, {"SENTIMENT_MODEL_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SENTIMENT_MODEL_PATH", None)  # restored by patch.dict

    def run_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def predict(self, at, text):
        at.text_area(key="review").input(text)
        at.button[-1].click().run()
        self.assertFalse(at.exception)
        return [s.value for s in at.main.subheader]

    def test_empty_directory_shows_notice(self):
        at = self.run_app()
        self.assertFalse(at.exception)
        self.assertIn("No trained model found", at.info[0].value)
        self.assertTrue(at.button[-1].disabled)
        self.assertEqual(len(at.selectbox), 0)

    def test_single_model_has_no_selector(self):
        joblib.dump(fitted_model(), self.root / "only.joblib")
        at = self.run_app()
        self.assertEqual(len(at.selectbox), 0)
        self.assertIn("Negative", self.predict(at, "awful stale bad"))

    def test_selector_switches_models(self):
        flipped = [{"Positive": "Negative", "Negative": "Positive"}.get(c, c) for c in LABELS]
        joblib.dump(fitted_model(), self.root / "a_normal.joblib")
        joblib.dump(create_nb_pipeline("tfidf").fit(TEXTS, flipped), self.root / "b_flipped.joblib")
        at = self.run_app()
        self.assertEqual(at.selectbox(key="model_choice").options, ["a_normal", "b_flipped"])
        self.assertIn("Positive", self.predict(at, "great tasty love"))
        at.selectbox(key="model_choice").select(str(self.root / "b_flipped.joblib"))
        self.assertIn("Negative", self.predict(at, "great tasty love"))

    def test_removed_selection_falls_back_to_default(self):
        for name in ("a.joblib", "b.joblib", "c.joblib"):
            joblib.dump(fitted_model(), self.root / name)
        at = self.run_app()
        at.selectbox(key="model_choice").select(str(self.root / "c.joblib"))
        self.predict(at, "great tasty love")
        (self.root / "c.joblib").unlink()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="model_choice").value, str(self.root / "a.joblib"))
        self.assertFalse(at.button[-1].disabled)

    def test_sidebar_shows_selected_model_and_count(self):
        for name in ("sentiment_pipeline.joblib", "other.joblib"):
            joblib.dump(fitted_model(), self.root / name)
        at = self.run_app()
        captions = [c.value for c in at.sidebar.caption]
        self.assertIn(f"`{self.root / 'sentiment_pipeline.joblib'}`", captions)
        self.assertIn("2 models available", captions)
        at.selectbox(key="model_choice").select(str(self.root / "other.joblib"))
        self.predict(at, "great tasty love")
        self.assertIn(f"`{self.root / 'other.joblib'}`", [c.value for c in at.sidebar.caption])

    def test_selection_persists_across_reruns(self):
        for name in ("a.joblib", "b.joblib"):
            joblib.dump(fitted_model(), self.root / name)
        at = self.run_app()
        at.selectbox(key="model_choice").select(str(self.root / "b.joblib"))
        self.predict(at, "great tasty love")
        at.run()
        self.assertEqual(at.selectbox(key="model_choice").value, str(self.root / "b.joblib"))

    def test_model_added_while_running_appears_in_selector(self):
        joblib.dump(fitted_model(), self.root / "a.joblib")
        at = self.run_app()
        self.assertEqual(len(at.selectbox), 0)
        joblib.dump(fitted_model(), self.root / "b.joblib")
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="model_choice").options, ["a", "b"])
        self.assertIn("2 models available", [c.value for c in at.sidebar.caption])

    def test_corrupt_selection_reports_error_but_allows_switching(self):
        joblib.dump(fitted_model(), self.root / "a_good.joblib")
        (self.root / "b_corrupt.joblib").write_bytes(b"not a joblib file")
        at = self.run_app()
        self.assertFalse(at.error)
        at.selectbox(key="model_choice").select(str(self.root / "b_corrupt.joblib"))
        self.assertEqual(self.predict(at, "great tasty love"), [])  # no prediction from a broken model
        self.assertIn("b_corrupt.joblib", at.error[0].value)
        # Submitting is how a form-held selection is applied, so it must stay possible.
        self.assertFalse(at.button[-1].disabled)
        at.selectbox(key="model_choice").select(str(self.root / "a_good.joblib"))
        self.assertIn("Positive", self.predict(at, "great tasty love"))
        self.assertFalse(at.error)

    def test_single_corrupt_model_disables_submit(self):
        (self.root / "only.joblib").write_bytes(b"not a joblib file")
        at = self.run_app()
        self.assertIn("only.joblib", at.error[0].value)
        self.assertTrue(at.button[-1].disabled)

    def test_each_model_is_cached_separately(self):
        for name in ("a.joblib", "b.joblib"):
            joblib.dump(fitted_model(), self.root / name)
        with mock.patch("inference.load_model", wraps=load_model) as loader:
            at = self.run_app()
            for name in ("b.joblib", "a.joblib", "b.joblib"):
                at.selectbox(key="model_choice").select(str(self.root / name))
                self.predict(at, "great tasty love")
            self.assertEqual(sorted(Path(c.args[0]).name for c in loader.call_args_list),
                             ["a.joblib", "b.joblib"])

    def test_model_path_override_hides_selector(self):
        for name in ("a.joblib", "b.joblib"):
            joblib.dump(fitted_model(), self.root / name)
        os.environ["SENTIMENT_MODEL_PATH"] = str(self.root / "b.joblib")  # restored by patch.dict
        at = self.run_app()
        self.assertEqual(len(at.selectbox), 0)
        self.assertIn(f"`{self.root / 'b.joblib'}`", [c.value for c in at.sidebar.caption])
        self.assertIn("Positive", self.predict(at, "great tasty love"))


if __name__ == "__main__":
    unittest.main()
