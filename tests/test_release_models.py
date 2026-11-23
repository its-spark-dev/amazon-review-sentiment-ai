"""Unit tests for ensure_models_release; network access is always faked."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import streamlit as st
from streamlit.testing.v1 import AppTest

import inference
from inference import RELEASE_MANIFEST, RELEASE_URL, ensure_models_release

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.1.0-rc1"
PAYLOADS = {"a.joblib": b"model a bytes", "b.joblib": b"model b bytes, a bit longer"}
APPROVED_FILES = {"bow_lr.joblib", "bow_nb.joblib", "tfidf_lr.joblib", "tfidf_mlp.joblib", "tfidf_nb.joblib"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeUrlopen:
    """Stand-in for urllib.request.urlopen serving bytes keyed by URL."""

    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        body = self.responses[url]
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body)


def url(name: str, tag: str = TAG) -> str:
    return RELEASE_URL.format(tag=tag, name=name)


class EnsureModelsReleaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model_dir = self.root / "models"
        self.manifest = self.root / "manifest.json"
        self.write_manifest({TAG: {name: sha(data) for name, data in PAYLOADS.items()}})

    def write_manifest(self, data):
        self.manifest.write_text(json.dumps(data))

    def run_ensure(self, responses=None, tag=TAG):
        fake = FakeUrlopen(responses or {url(n): d for n, d in PAYLOADS.items()})
        with mock.patch.object(inference.urllib.request, "urlopen", fake):
            result = ensure_models_release(tag, self.model_dir, manifest_path=self.manifest)
        return result, fake

    def leftovers(self):
        return [p.name for p in self.model_dir.iterdir() if p.suffix == ".part"]

    def test_downloads_missing_files_in_manifest_order(self):
        paths, fake = self.run_ensure()
        self.assertEqual(paths, [self.model_dir / n for n in PAYLOADS])
        self.assertEqual(fake.urls, [url(n) for n in PAYLOADS])
        for name, data in PAYLOADS.items():
            self.assertEqual((self.model_dir / name).read_bytes(), data)
        self.assertEqual(self.leftovers(), [])

    def test_reuses_files_whose_hash_matches(self):
        self.model_dir.mkdir()
        for name, data in PAYLOADS.items():
            (self.model_dir / name).write_bytes(data)
        before = {n: (self.model_dir / n).stat().st_mtime_ns for n in PAYLOADS}
        _, fake = self.run_ensure(responses={})
        self.assertEqual(fake.urls, [])
        self.assertEqual({n: (self.model_dir / n).stat().st_mtime_ns for n in PAYLOADS}, before)

    def test_replaces_only_the_file_with_a_wrong_hash(self):
        self.model_dir.mkdir()
        (self.model_dir / "a.joblib").write_bytes(PAYLOADS["a.joblib"])
        (self.model_dir / "b.joblib").write_bytes(b"corrupted")
        _, fake = self.run_ensure()
        self.assertEqual(fake.urls, [url("b.joblib")])
        self.assertEqual((self.model_dir / "b.joblib").read_bytes(), PAYLOADS["b.joblib"])
        self.assertEqual(self.leftovers(), [])

    def test_downloaded_hash_mismatch_raises_logs_and_keeps_old_file(self):
        self.model_dir.mkdir()
        (self.model_dir / "a.joblib").write_bytes(PAYLOADS["a.joblib"])
        (self.model_dir / "b.joblib").write_bytes(b"old")
        responses = {url("b.joblib"): b"tampered"}
        with self.assertLogs("inference", "ERROR") as logs, self.assertRaises(ValueError):
            self.run_ensure(responses)
        self.assertIn("b.joblib", "\n".join(logs.output))
        self.assertEqual((self.model_dir / "b.joblib").read_bytes(), b"old")
        self.assertEqual(self.leftovers(), [])

    def test_hash_mismatch_does_not_create_missing_target(self):
        responses = {url("a.joblib"): b"tampered"}
        with self.assertLogs("inference", "ERROR"), self.assertRaises(ValueError):
            self.run_ensure(responses)
        self.assertFalse((self.model_dir / "a.joblib").exists())
        self.assertEqual(self.leftovers(), [])

    def test_network_error_raises_oserror_and_logs_filename(self):
        responses = {url("a.joblib"): urllib.error.HTTPError(url("a.joblib"), 404, "Not Found", {}, None)}
        with self.assertLogs("inference", "ERROR") as logs, self.assertRaises(OSError):
            self.run_ensure(responses)
        self.assertIn("a.joblib", "\n".join(logs.output))
        self.assertFalse((self.model_dir / "a.joblib").exists())
        self.assertEqual(self.leftovers(), [])

    def test_interrupted_download_cleans_up_temp_file(self):
        class Broken(io.BytesIO):
            def read(self, *args):
                raise ConnectionResetError("connection reset")

        fake = mock.Mock(return_value=Broken())
        with (
            mock.patch.object(inference.urllib.request, "urlopen", fake),
            self.assertLogs("inference", "ERROR"),
            self.assertRaises(OSError),
        ):
            ensure_models_release(TAG, self.model_dir, manifest_path=self.manifest)
        self.assertEqual(self.leftovers(), [])

    def test_unknown_tag_raises_valueerror_without_network(self):
        with self.assertLogs("inference", "ERROR") as logs, self.assertRaises(ValueError):
            self.run_ensure(responses={}, tag="v9.9.9")  # any network call would KeyError
        self.assertIn("v9.9.9", "\n".join(logs.output))

    def test_rejects_unsafe_filenames_and_bad_hashes(self):
        for entries in ({"../evil.joblib": sha(b"x")}, {"sub/evil.joblib": sha(b"x")},
                        {"evil.py": sha(b"x")}, {"a.joblib": "not-a-hash"}, {}):
            with self.subTest(entries=entries):
                self.write_manifest({TAG: entries})
                fake = FakeUrlopen({})
                with (
                    mock.patch.object(inference.urllib.request, "urlopen", fake),
                    self.assertLogs("inference", "ERROR"),
                    self.assertRaises(ValueError),
                ):
                    ensure_models_release(TAG, self.model_dir, manifest_path=self.manifest)
                self.assertEqual(fake.urls, [])
                self.assertFalse((self.root / "evil.joblib").exists())

    def test_missing_manifest_raises_oserror(self):
        self.manifest.unlink()
        with self.assertLogs("inference", "ERROR"), self.assertRaises(OSError):
            self.run_ensure(responses={})

    def test_release_url_is_pinned_to_tag(self):
        self.assertEqual(
            url("bow_lr.joblib"),
            "https://github.com/its-spark-dev/amazon-review-sentiment-ai/releases/download/v0.1.0-rc1/bow_lr.joblib",
        )


class TrackedManifestTests(unittest.TestCase):
    def test_first_release_pins_the_five_approved_models(self):
        manifest = json.loads(RELEASE_MANIFEST.read_text())
        self.assertIn(TAG, manifest)
        self.assertEqual(set(manifest[TAG]), APPROVED_FILES)
        for name, digest in manifest[TAG].items():
            with self.subTest(name=name):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")


class AppReleaseWiringTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model_dir = tmp.name

    def run_app(self, env):
        with mock.patch.dict(os.environ, env):
            os.environ.pop("SENTIMENT_MODEL_PATH", None)
            return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

    def test_failure_shows_error_and_stops(self):
        env = {"MODELS_VERSION_TAG": TAG, "SENTIMENT_MODEL_DIR": self.model_dir}
        with mock.patch("inference.ensure_models_release", side_effect=OSError("boom")) as ensure:
            at = self.run_app(env)
        ensure.assert_called_once_with(TAG, Path(self.model_dir))
        self.assertEqual([e.value for e in at.error], ["Failed to prepare staging models"])
        self.assertEqual(len(at.title), 0)  # st.stop() before the rest of the page

    def test_success_renders_page(self):
        env = {"MODELS_VERSION_TAG": TAG, "SENTIMENT_MODEL_DIR": self.model_dir}
        with mock.patch("inference.ensure_models_release") as ensure:
            at = self.run_app(env)
        ensure.assert_called_once()
        self.assertEqual(len(at.error), 0)
        self.assertEqual(at.title[0].value, "Amazon Review Sentiment AI")

    def test_not_called_without_tag(self):
        env = {"SENTIMENT_MODEL_DIR": self.model_dir}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODELS_VERSION_TAG", None)
            with mock.patch("inference.ensure_models_release") as ensure:
                self.run_app(env)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
