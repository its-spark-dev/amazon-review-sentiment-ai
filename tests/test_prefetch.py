"""Unit tests for the background prefetch helpers (prefetch.py); synthetic models only."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from inference import predict_review
from ml_baseline import CLASS_LABELS
from prefetch import PrefetchStore, build_view, evaluate, file_version
import synthetic


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=2)
    yield pool
    pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from inference import load_model

    return load_model(synthetic.save_model(tmp_path_factory.mktemp("prefetch") / "m.joblib"))


def test_view_formats_what_the_result_section_shows(model):
    result = predict_review(model, synthetic.PROBES["Positive"])
    view = build_view(result)
    probs = result["probabilities"]
    assert view["label"] == "Positive"
    assert view["metric"] == f"{probs['Positive']:.1%}"
    assert view["probs"] == [probs[c] for c in CLASS_LABELS]
    assert view["breakdown"] == " · ".join(f"{c}: {probs[c]:.1%}" for c in CLASS_LABELS)


def test_view_without_probabilities_has_label_only():
    assert build_view({"label": "Neutral", "probabilities": None}) == {
        "label": "Neutral", "metric": None, "probs": None, "breakdown": None,
    }


@pytest.mark.parametrize("text, message", [
    ("   ", "Enter a review to analyze."),
    ("quuxzzblorb snarglexyz", "No words in this review are recognized"),
])
def test_rejected_text_becomes_the_same_warning(model, text, message):
    outcome = evaluate(model, text)
    assert set(outcome) == {"warning"}
    assert outcome["warning"].startswith(message)


def test_submit_runs_each_key_once(executor):
    store = PrefetchStore(executor)
    calls = []
    first = store.submit(("m", (1,), "t"), lambda: calls.append(1) or {"view": 1})
    again = store.submit(("m", (1,), "t"), lambda: calls.append(2) or {"view": 2})
    assert again is first
    assert first.result() == {"view": 1}
    assert calls == [1]


def test_store_keeps_only_the_most_recent_keys(executor):
    store = PrefetchStore(executor, limit=2)
    for text in ("a", "b", "c"):
        store.submit(("m", (1,), text), lambda: {})
    assert store.get(("m", (1,), "a")) is None
    assert store.get(("m", (1,), "c")) is not None
    assert len(store) == 2


def test_done_outcome_ignores_running_and_failed_jobs(executor):
    store = PrefetchStore(executor)
    gate = threading.Event()
    running = store.submit(("m", (1,), "slow"), lambda: gate.wait(5) and {"view": "late"})
    failed = store.submit(("m", (1,), "bad"), lambda: 1 / 0)
    failed.exception(timeout=5)
    assert store.done_outcome(("m", (1,), "slow")) is None
    assert store.done_outcome(("m", (1,), "bad")) is None
    assert store.done_outcome(("m", (1,), "missing")) is None
    gate.set()
    running.result(timeout=5)
    assert store.done_outcome(("m", (1,), "slow")) == {"view": "late"}


def test_key_tracks_file_identity(tmp_path):
    path = synthetic.save_model(tmp_path / "m.joblib")
    before = PrefetchStore.key(path, file_version(path), "text")
    replacement = synthetic.save_model(tmp_path / "new.joblib", flipped=True)
    replacement.replace(path)
    after = PrefetchStore.key(path, file_version(path), "text")
    assert before != after
    assert before[0] == after[0] == str(path.resolve())
