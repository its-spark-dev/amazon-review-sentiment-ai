# Portfolio readiness review

Status updated **November 24, 2026 (EST)** against main `0b85192`. The original
audit was November 23 against `2445b30`; dated observations below distinguish
that history from completed follow-ups. This is an evidence audit, not a new
experiment or a claim of defect-free software. The original [PR #22](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/22)
changed documentation and a verified demo screenshot only and is now merged.

**Serving follow-up, November 24:** [PR #23](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/23)
merged at `9a29272` and implements the zero-feature guard,
TF-IDF LR initial selection and explicit uncalibrated-probability wording from
items 2–3 below. Synthetic regression tests cover rejection, recovery, unchanged
supported predictions and preserved user choices. The original observations below
remain dated audit evidence. Main was then rebooted and
[verified live](M3_RESULTS.md#hosted-main-status): initial selection/probability
wording, 10 matching rc2 predictions, 15 rejected zero-feature inputs, five
recoveries and two existing validation checks. Staging was not reverified.
The serving changes do not alter rc2 artifacts, preprocessing, model-selection
protocol, metrics or model weights.

**Development/CI follow-up, November 24:**
[PR #27](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/27) merged
at `0b85192`. Codespaces now enables CORS/XSRF protection with the forwarded
hostname, and CI uses Node.js 24 actions. All six
[main CI jobs](https://github.com/its-spark-dev/amazon-review-sentiment-ai/actions/runs/45348633211)
passed, totaling **195 tests**, with zero check annotations and no Node.js 20
deprecation warnings. The five added tests exercise the launcher and a real local
server with simulated forwarding metadata; hosted Codespaces proxy verification
remains pending. Model files, hashes and runtime dependency versions are unchanged.

## Conclusion

The repository can tell a credible **educational ML engineering** story: sparse
features, classical and neural baselines, training-only selection, reproducible
artifact handoff, a working demo and layered tests. It should not be described as
state-of-the-art NLP, a calibrated sentiment service, an unseen-data benchmark,
or a completed final course submission. Those claims are not supported.

M1 and M2 submission are confirmed by the user. M3's implementation is complete
and its PDF needs updating. [PR #21](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/21)
merged at `5d83c12`: the [frozen rc2 error analysis](M4_ERROR_ANALYSIS.md), three
figures, eight inspected paraphrases and public reproduction path are now in main.
M4 analysis materials are ready; the final report and presentation remain pending.
No new course-submission completion is inferred from these merges.

[PR #25](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/25) merged
at `5545a8b`. MIT now covers the team's code, original documentation/figures and
project-trained models; Kaggle's CC0 data designation is separate. GitHub detects
MIT. See [licensing scope](LICENSING.md).

## Portfolio work now available

| Area | Change | Evidence boundary |
| --- | --- | --- |
| First impression | CI/Python/model/demo badges, live app link, actual app screenshot | No static accuracy badge or invented benchmark ranking |
| Results | rc2 comparison and validation recommendation before course history | All numbers read from M3_RELEASE.json; M2 remains a separate linked experiment |
| Setup | Dataset-free release-demo command first; training/testing separately | Existing commands and dependency files retained |
| Status | M1/M2 Submitted, M3 PDF update pending, M4 analysis materials ready; final report/presentation pending | Submission is team-confirmed, not inferred from commits |
| Model transparency | MODEL_CARD.md | Intended use, method, limitations, runtime/artifact contract |
| Collaboration | CONTRIBUTING.md | Branch/PR workflow, validation, attribution and local-artifact policy |
| Navigation | README links to the merged M4 error analysis and licensing scope | Analysis availability does not establish final course submission |
| Older guides | Removed stale placeholder, submission and missing-validation statements | Historical experiments and measured results are preserved |

## Repository and ML evidence checked

- **Tests:** main `0b85192` passed all six CI jobs after PR #27: **195 tests**
  (136 unit, 42 integration and 17 browser), including five development-container
  tests and the existing 14 focused error-analysis tests. Lint, dependency and
  tracked-file hygiene checks passed. The earlier 190-test PR #21 revalidation
  and main `5545a8b` CI remain historical evidence; the November 23 count of 158
  predates the serving and error-analysis regression tests.
- **Historical coverage:** the PR #21 190-test run reported 89% statement coverage over 824
  measured statements (94 missed). app.py was 99%, inference.py 98%, the error
  analysis CLI 91%, ml_baseline.py 100% and scripts/split_reviews.py 46%. Scripts
  not imported in the measured process can be absent, so this is not coverage of
  every repository file or a claim of complete testing. CLI coverage remains a
  possible improvement.
- **Release contract:** the three audited rc2 models' SHA-256 hashes and the
  canonical test CSV hash still match M3_RELEASE.json. The training/selection
  code fits only train/inner-fit inputs, freezes selection before final test
  evaluation, and preserves the shared outer split.
- **Clean-checkout demo:** a temporary snapshot of main contained no dataset or
  preinstalled model binaries. With the documented rc2 environment setting, the
  app downloaded and hash-verified all five artifacts, exposed five choices and
  reproduced the documented TF-IDF LR Positive example (98.6% probability).
  AppTest reported no application errors. This is a serving check, not a new
  accuracy measurement.
- **Presentation checks:** documentation paths/anchors and genuine deployment
  captures were validated in the documentation follow-ups. The README now has
  CI, Python, model-release, demo and MIT badges. Its November 24 screenshots
  show the current probability explanation and input guard, not mockups. The
  [versioned rc2 release description](releases/v0.2.0-rc2.md) records main,
  merged error analysis, latest 195-test validation and MIT/data scope, while
  retaining the earlier 190-test verification as history.
- **Software tests vs scientific evidence:** synthetic tests check contracts and
  regressions; the real-data report provides the measured model scores. Neither
  is a substitute for the other. No real-data retraining was performed here.
- **Tracked-file hygiene:** no dataset/model binary or named secret file is
  tracked on main. A focused recognizable-token/private-key signature scan of
  tracked text found no matches, and available Git history contained no matching
  dataset/model/secret filenames. This is a limited check, not proof that every
  possible secret or personal detail has been ruled out.
- **Metadata:** repository is public; description and NLP topics already exist.
  Homepage remains empty. GitHub now detects MIT after PR #25; the original
  missing-license finding is resolved. Repository settings remain unchanged.
- **Attribution:** the team roster, Jacob's UI/data-script contributions, Sang's
  ML contributions and existing AI-assistance disclosure remain visible.

## Prioritized follow-up work

### A. Publication decisions and small serving fixes

**1. Code and artifact licensing (implemented in the licensing follow-up).**
The original audit found no LICENSE and could not read Kaggle's license label.
The team has since authorized MIT for its original code, documentation, figures
and project-trained model artifacts. The [MIT License](../LICENSE) and
[licensing scope](LICENSING.md) record this grant without changing existing model
bytes or release assets. The upstream Kaggle metadata was checked on November 24
and identifies the dataset as **CC0: Public Domain**; reviews and local dataset
derivatives are explicitly excluded from the project's MIT grant. Third-party
materials retain their own terms. The original license-gap finding is historical,
not an unresolved requirement to obtain another public-sharing approval.

**2. Zero-feature input guard (implemented in the serving follow-up).**
At the original audit, [inference.py](../inference.py) checked blank/long/markup-only
input, but did not check whether the chosen vectorizer recognized any feature.
Probes `🍪🍪🍪` and
`이 음식은 맛있고 신선해요` both yielded zero nonzero features for the audited
BoW LR, TF-IDF LR and MLP artifacts. Nonetheless each returned Positive, with
probabilities approximately 74.01%, 39.09% and 77.81%, respectively. These values
are **probe outputs**, not accuracy or language benchmarks. This establishes a
serving limitation, not a reason to retrain or silently alter rc2 weights.
Acceptance: zero-feature input produces a clear insufficient-vocabulary message;
normal inputs retain the same probabilities; unit/AppTest coverage exercises
both paths with synthetic artifacts. Do not claim a general language detector.

**3. Align serving explanation and initial selection (implemented in the follow-up).**
At the audit, TF-IDF LR was recommended while [app.py](../app.py) initially used
the first manifest entry, BoW LR. The follow-up initially selects tfidf_lr when
available, preserves valid user choices and retains local single-file overrides.
It replaces “Confidence” with “Predicted-class probability”, displays one decimal
place and explains that probabilities are uncalibrated estimates. Acceptance:
synthetic switching/cache tests still pass; no score or model artifact changes;
labels explain actual behavior. Calibration itself remains separate future work.

**4. Codespaces protection settings (implemented and merged in PR #27).**
The original [devcontainer.json](../.devcontainer/devcontainer.json) disabled both
CORS and XSRF protection; that was a development configuration, not evidence about
the deployed app. The launcher now explicitly enables both and uses the current
Codespace's forwarded hostname as `browser.serverAddress`. Installation uses only
the existing pinned runtime requirements, and port 8501 opens in a browser tab.
Regression tests exercise startup, allowed/rejected WebSocket origins and XSRF
token enforcement on a real local server with simulated forwarding metadata.
This does not establish end-to-end verification of GitHub's authenticated proxy
or a hosted Codespace. See the [development-container guide](../.devcontainer/README.md).

### B. Evidence improvements for a stronger ML portfolio

**5. Define the generalization question before creating new splits.** Current
grouping prevents exact cleaned Text overlap, not shared users/products. An audit
of the frozen CSVs found:

| Field | Shared distinct values across train/test | Test rows whose value occurs in train |
| --- | ---: | ---: |
| Id | 0 | 0 / 9,255 |
| Text | 0 | 0 / 9,255 |
| UserId | 1,353 | 1,650 / 9,255 |
| ProductId | 3,673 | 6,289 / 9,255 |

IDs are not model features. These aggregates do not alone invalidate the course's
shared-split task, but do rule out describing it as an unseen-user/product test.
Near-duplicates and temporal leakage were not measured. Optional future work:
predeclare a cold-start or temporal evaluation and preserve it as a separate
experiment. Do not replace rc2 or repeatedly optimize the shared test.

**6. Add uncertainty and robustness evidence only under a declared protocol.**
The current comparison has one split/seed and no confidence intervals. Paired
bootstrap intervals could describe sampling variability of the frozen test
predictions but would not fix prior test inspection, dependence between related
reviews, or training-seed uncertainty. Training stability requires separately
planned training-only repeated validation. Treat class imbalance, rating/text
ambiguity, unigram negation scope and out-of-domain review text as limitations,
not measured causes unless an analysis demonstrates them.

**7. Assess probability calibration before offering confidence-based decisions.**
predict_proba alone does not establish calibration; see
[sklearn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
No calibration measurement is recorded in the current release. A future study
can report reliability diagnostics and proper scoring rules, then fit any
calibrator on training-only out-of-fold or separate calibration data. Do not
fit it on the shared test or promise that class weighting improves calibration.
Any changed pipeline needs a new version and its own validation.

**8. Frozen error analysis (completed and merged in PR #21).**
The analysis provides paired error overlap, class/length diagnostics and eight
inspected paraphrases. The CLI needs only approved release models and the
canonical test CSV; a retained raw evaluation export is an optional cross-check.
Both paths reproduce the same aggregates without training. All 27,765 retained
predictions and eight case mappings matched, and all three figures reproduced
pixel-identically. These retrospective cases are illustrative, not prevalence
estimates or a completed M4 report/presentation.

### C. Reproduction and maintenance polish

**9. Add a complete serving-environment capture if long-term maintenance is a goal.**
requirements.txt pins direct packages; transitive dependencies are not locked.
Training, main and staging also use different Python versions. Current hashes
and successful serving probes are valuable but do not promise byte-identical
retraining on all platforms. A separately validated environment/container could
improve this. Avoid upgrading sklearn under existing joblib artifacts without
revalidation; [sklearn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html)
describes compatibility and trust constraints.

**10. Maintain documentation consistency and improve CLI coverage.**
The older ML/data guides' obsolete placeholder and missing-validation statements
are corrected here. Some milestone-relative historical plans remain, and the
requirements comment now identifies the current rc2 artifact contract without
changing package pins. Completed PR #21 and MIT work are recorded above. Preserve
historical experiment results while keeping current status explicit. The clean-checkout smoke
check and local documentation-link check passed here; automating them and
improving CLI branch coverage are possible follow-ups. No badge should promise
100% coverage.
Avoid adding a package framework or experiment platform just for appearance.
The [documentation PR's CI run](https://github.com/its-spark-dev/amazon-review-sentiment-ai/actions/runs/45306214735)
reported Node.js 20 deprecation warnings for the old checkout/setup/upload
actions, which the runner forced onto Node.js 24. Merged PR #27 uses
official Node.js 24 releases of checkout, setup-python, upload-artifact and cache
and retains all six CI jobs; its main run has no Node.js 20 deprecation warnings.
The Python/model serving environment is unchanged;
Actions runtime versions are separate from runtime package versions.

**11. Prepare repository metadata after maintainer review.** Suggested homepage:
`https://amazon-sentiment-ai.streamlit.app/`. Existing topics already cover the
stack. A social preview can use the real demo image or a simple project graphic;
do not imply university endorsement or invent an award. A citation file can be
added after contributors confirm names and preferred attribution. The licensing
follow-up adds the license files; social-preview, description/homepage and access
settings remain unchanged.

## Review and publication checks

Before sharing the portfolio link, keep model/metric claims tied to rc2 and
confirm the demo is reachable
at the time of sharing. Keep M3 PDF updating and the M4 report/presentation as
course tasks; a polished README cannot stand in for their submission. For any
future serving fix, repeat model selection/prediction and error-path checks.

The README screenshots are genuine main-app captures from the
November 24 deployment verification, replacing the original November 23 image.
No third-party photos, logos, reviewer
identifiers or invented outputs were added. Image access and readable captions
matter more than decorative badges.

## Primary guidance consulted

- [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes): lead with purpose, use and navigation.
- [Hugging Face model-card guidance](https://huggingface.co/docs/hub/model-cards): document intended use, data, evaluation and limitations; this project is not claiming Hub publication.
- [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): training-only fitting and separation of selection/evaluation.
- [SNAP data description](https://snap.stanford.edu/data/web-FineFoods.html): dataset provenance and historical scope.

Implementation evidence comes from the reviewed source and canonical local
artifacts; the external guidance supplies review criteria, not evidence that
this particular model is reliable. Serving fixes, licensing and frozen error
analysis are completed work. Merged PR #27 addresses the Codespaces
protection and Actions runtime findings. Generalization/calibration studies,
complete environment capture and repository metadata polish remain proposed
improvements; hosted Codespaces forwarding still needs an end-to-end check.
