"""Background prefetch of predictions so a submitted review renders without waiting.

The app learns the pending (not yet submitted) review text and model choice from
the form watcher (``form_watcher.py``). It submits the same work a submit would
do, loading the model and predicting, to a small thread pool, keyed by the model
file's identity and the exact text. On submit, a finished prefetch for that key
is used directly, and anything else falls back to computing on the spot.

Outcomes are plain JSON-serializable dicts, so the same object is rendered by
Streamlit and can be pushed to the browser for an instant preview:

- ``{"view": {"label", "metric", "probs", "breakdown"}}`` for a prediction
- ``{"warning": str}`` when ``predict_review`` rejects the text
"""

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading

from inference import predict_review
from ml_baseline import CLASS_LABELS

PREFETCH_LIMIT = 8  # per session: enough for a few back-and-forth model switches


def file_version(path: Path) -> tuple:
    """Identity of a model file on disk: changes on normal writes and atomic replacements."""
    stat = Path(path).stat()
    return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)


def build_view(result: dict) -> dict:
    """Turn a ``predict_review`` result into everything the result section displays."""
    label = result["label"]
    probs = result["probabilities"]
    view = {"label": label, "metric": None, "probs": None, "breakdown": None}
    if probs:
        view["metric"] = f"{probs[label]:.1%}" if label in probs else "—"
        view["probs"] = [probs[c] for c in CLASS_LABELS]
        view["breakdown"] = " · ".join(f"{c}: {probs[c]:.1%}" for c in CLASS_LABELS)
    return view


def evaluate(model, text) -> dict:
    """Predict ``text`` with ``model`` and return the outcome to render."""
    try:
        return {"view": build_view(predict_review(model, text))}
    except ValueError as exc:
        return {"warning": str(exc)}


class PrefetchStore:
    """Bounded map of ``(model path, file version, text)`` to the Future computing its outcome."""

    def __init__(self, executor: ThreadPoolExecutor, limit: int = PREFETCH_LIMIT):
        self._executor = executor
        self._limit = limit
        self._futures: OrderedDict[tuple, Future] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(path: Path, version: tuple, text: str) -> tuple:
        return (str(Path(path).resolve()), version, text)

    def submit(self, key: tuple, fn: Callable[[], dict]) -> Future:
        """Start ``fn`` for ``key`` unless it is already running or done."""
        with self._lock:
            future = self._futures.get(key)
            if future is not None:
                self._futures.move_to_end(key)
                return future
            future = self._executor.submit(fn)
            self._futures[key] = future
            while len(self._futures) > self._limit:
                self._futures.popitem(last=False)
            return future

    def get(self, key: tuple) -> Future | None:
        with self._lock:
            return self._futures.get(key)

    def done_outcome(self, key: tuple) -> dict | None:
        """The outcome for ``key`` if its prefetch finished without raising, else None."""
        future = self.get(key)
        if future is None or not future.done() or future.cancelled() or future.exception() is not None:
            return None
        return future.result()

    def __len__(self) -> int:
        return len(self._futures)
