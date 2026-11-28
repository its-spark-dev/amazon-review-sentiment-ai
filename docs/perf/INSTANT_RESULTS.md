# Instant results: background prefetch and double-buffered rendering

## Summary

Clicking **Analyze sentiment** used to be the first time the app did any work: it
loaded the model, predicted, and then the browser drew the Vega chart. Almost all
of that wait was the chart drawing (about 400–690 ms); loading and prediction took
about 20–25 ms.

The app now does all of that work before the click and keeps the result hidden
until then. If the user pauses before clicking, the result appears **about 1 ms
after the click** instead of 40–140 ms for the heading and 400–690 ms for the
chart. If the user clicks before anything is ready, the timing is the same as
before.

## How it works

- **Form watcher** (`form_watcher.py`). Streamlit form widgets send nothing to the
  server until the form is submitted. The app adds an invisible `st.components.v2`
  component, built into Streamlit with no new dependency or build step. It finds
  the form's own widgets by their accessible names and reports:
  - the review text 1 s after typing stops;
  - the model choice as soon as it changes.

  The form, text area and selectbox themselves are unchanged.
- **Background prefetch** (`prefetch.py`). Each report starts a load +
  `predict_review` job on a small thread pool. Jobs are keyed by the model file's
  identity (mtime, ctime, size, inode) and the exact text, so a model file that was
  overwritten is never served from an old result. Each session keeps the last 8
  jobs. On submit, a finished job with the same key is reused; anything else is
  computed on the spot as before.
- **Double-buffered result** (`app.py`, `form_watcher.py`). The finished result is
  rendered into a hidden "back" slot, so the browser lays out and draws it, chart
  included, without showing it. When Analyze is clicked (or Ctrl/Cmd+Enter is
  pressed) and the form still holds that exact text and model, the slot is revealed
  in the browser right away. The submit run then renders the same elements into the
  same slot, so the chart isn't drawn again. Two slots take turns, so the result on
  screen stays visible while the next one is drawn behind it.
- **Result survives watcher reruns.** Typing more after an analysis doesn't clear
  the shown result. The kept result is tied to the model that produced it. A
  watcher rerun can't interrupt and drop an in-flight submit.
- **No re-hashing of release models on every rerun** (`inference.py`).
  `ensure_models_release` runs on every rerun, and the watcher adds reruns. Files
  this process has already verified are skipped while their mtime, ctime, size and
  inode are unchanged; any change triggers a full re-hash. In release mode this
  cuts the per-rerun cost from 8.2 ms to 0.06 ms.

## Privacy note

The pending review text reaches the app server about 1 s after the user stops
typing, before they click Analyze. The prediction stays in server memory for that
session only and isn't shown until the click. The [README](../../README.md) and
[model card](../MODEL_CARD.md) both say this.

## Benchmarks

**Method**

- Playwright with headless Chromium against a real `streamlit run`, using the
  `v0.2.0-rc2` release models.
- Each time is measured from the click to the result being actually visible
  (`checkVisibility({visibilityProperty: true})`), checked on every DOM mutation
  and every animation frame:
  - **Heading:** the result label is visible.
  - **Chart:** the Vega chart is visible.
- 1 warm-up run and 10 measured runs per cell; each cell reports the median and
  p90.
- **Before:** `staging` @ `394e0ac`. **After:** `feat/local-prefetch` @ `9a47ac1`.
- **Machine:** 16-core WSL2 (Linux 6.6), Python 3.11, Streamlit 1.67.0,
  scikit-learn 1.9.1.

**Scenarios**

- **Fresh server:** freshly started server; type, wait 1.5 s, click.
- **Model loaded:** model already loaded; new text, wait 1.5 s, click.
- **No pause:** type and click right away, so nothing has been prefetched.

**Summary**, median ms across all 5 models (local mode):

| Scenario | Heading before → after | Chart before → after |
|---|---|---|
| Fresh server (1.5 s pause) | 42–63 → **1.0–1.2** | 609–685 → **1.2** |
| Model loaded, new text (1.5 s pause) | 97–138 → **1.2–1.4** | 395–517 → **1.2–1.4** |
| No pause | 96–112 → 70–131 | 393–423 → 363–417 |

<details>
<summary>Full results: local mode (median / p90, ms)</summary>

| Model | Scenario | Heading before | Heading after | Chart before | Chart after |
|---|---|---|---|---|---|
| bow_lr | fresh server | 57 / 62 | 1.1 / 1.2 | 660 / 699 | 1.2 / 108 |
| bow_lr | model loaded | 97 / 101 | 1.2 / 1.4 | 395 / 406 | 1.3 / 1.4 |
| bow_lr | no pause | 99 / 107 | 102 / 129 | 393 / 408 | 363 / 410 |
| bow_nb | fresh server | 53 / 65 | 1.2 / 1.4 | 634 / 664 | 1.2 / 1.4 |
| bow_nb | model loaded | 101 / 116 | 1.2 / 1.4 | 406 / 417 | 1.2 / 1.4 |
| bow_nb | no pause | 100 / 106 | 112 / 178 | 399 / 426 | 392 / 444 |
| tfidf_lr | fresh server | 42 / 48 | 1.1 / 1.1 | 609 / 626 | 1.2 / 95 |
| tfidf_lr | model loaded | 108 / 130 | 1.2 / 1.5 | 405 / 469 | 1.2 / 1.6 |
| tfidf_lr | no pause | 108 / 117 | 131 / 139 | 419 / 451 | 417 / 434 |
| tfidf_mlp | fresh server | 63 / 70 | 1.1 / 1.7 | 664 / 698 | 1.2 / 1.7 |
| tfidf_mlp | model loaded | 99 / 105 | 1.4 / 1.8 | 400 / 415 | 1.4 / 1.8 |
| tfidf_mlp | no pause | 112 / 130 | 124 / 160 | 423 / 451 | 406 / 448 |
| tfidf_nb | fresh server | 56 / 62 | 1.0 / 1.1 | 685 / 708 | 1.2 / 105 |
| tfidf_nb | model loaded | 138 / 161 | 1.2 / 1.4 | 517 / 536 | 1.2 / 1.5 |
| tfidf_nb | no pause | 96 / 108 | 70 / 114 | 405 / 410 | 365 / 389 |

</details>

<details>
<summary>Full results: release mode, <code>MODELS_VERSION_TAG=v0.2.0-rc2</code> (median / p90, ms)</summary>

| Model | Scenario | Heading before | Heading after | Chart before | Chart after |
|---|---|---|---|---|---|
| tfidf_lr | fresh server | 47 / 51 | 0.9 / 1.0 | 618 / 633 | 92 / 108 |
| tfidf_lr | model loaded | 100 / 107 | 1.1 / 1.5 | 399 / 418 | 1.2 / 1.5 |
| tfidf_lr | no pause | 107 / 111 | 104 / 109 | 412 / 418 | 372 / 381 |
| tfidf_mlp | fresh server | 67 / 71 | 1.0 / 1.3 | 672 / 704 | 1.3 / 106 |
| tfidf_mlp | model loaded | 107 / 110 | 1.1 / 1.3 | 403 / 413 | 1.2 / 1.3 |
| tfidf_mlp | no pause | 104 / 114 | 118 / 169 | 406 / 417 | 385 / 435 |

`ensure_models_release` per rerun: 8.2 ms → 0.06 ms.

</details>

<details>
<summary>Server-side split (offline, median ms)</summary>

| Model | joblib load | predict_review | view build |
|---|---|---|---|
| bow_lr | 23.0 | 0.62 | 0.006 |
| bow_nb | 19.8 | 0.70 | 0.006 |
| tfidf_lr | 21.0 | 1.52 | 0.007 |
| tfidf_mlp | 25.7 | 1.55 | 0.007 |
| tfidf_nb | 19.3 | 1.51 | 0.006 |

The model was never the slow part. Most of the old click time was the browser
drawing the Vega chart, and that now happens before the click.

</details>

**Notes on the numbers**

- **Fresh-server chart outliers.** In 3–4 of 10 fresh-server samples for some
  models, the chart took 70–280 ms instead of about 1 ms, because the first chart on
  a new page hadn't finished drawing in the hidden slot within the 1.5 s pause.
  This is why release-mode tfidf_lr has a 92 ms median and why several p90s are
  around 100 ms. Even these slow samples are 6–9× faster than before.
- **No pause: no real change.** Medians are within noise of before, about 100 ms
  for the heading and about 400 ms for the chart. A few heading p90s are higher (up
  to about 170 ms), possibly because a watcher rerun overlaps with the submit.

## Known issue

Once, in an earlier benchmark run, a no-pause click on tfidf_mlp right after a
model switch showed nothing within 60 s, and the server logged no errors. It didn't
happen again in 130 targeted retries or in the full rerun, which had 0 failed
samples. It's rare, but a real bug in the prefetch path hasn't been ruled out.

## Verification

At `9a47ac1`:

- `ruff check .`: all checks pass
- Unit (`-m "not integration and not e2e"`): 146 passed
- Integration (`-m integration`): 50 passed
- E2E (`-m e2e --browser chromium`): 21 passed

The tests for this feature cover:

- prefetch keying, including a replaced model file not being served from an old
  result;
- the per-session limit of 8 jobs;
- submit reusing a prefetched result, and computing on a text or model mismatch;
- rejected text giving the same warning as before;
- the result staying visible through watcher reruns;
- model-switch and sample-button prefetch;
- the hidden slot being revealed on click, and an edited text not revealing an old
  result;
- skipping re-hashes of unchanged release models.
