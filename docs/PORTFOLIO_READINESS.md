# Portfolio readiness review

Reviewed **November 23, 2026 (EST)** against main `2445b30`. This is a public
presentation and evidence audit, not a new experiment or a claim of defect-free
software. This PR changes documentation and a verified demo screenshot only.
It does not change models, dependencies, GitHub settings or deployment settings.

## Conclusion

The repository can tell a credible **educational ML engineering** story: sparse
features, classical and neural baselines, training-only selection, reproducible
artifact handoff, a working demo and layered tests. It should not be described as
state-of-the-art NLP, a calibrated sentiment service, an unseen-data benchmark,
or a completed final course submission. Those claims are not supported.

M1 and M2 submission are confirmed by the user. M3's implementation is complete
and its PDF needs updating. M4's formal final report/presentation is **not started**.
The separately open [PR #21](https://github.com/its-spark-dev/amazon-review-sentiment-ai/pull/21)
contains an error-analysis draft; it is not merged into the reviewed main and
does not mean the M4 milestone is complete. Keep its results out of main claims
until separately reviewed and merged.

## Changes prepared here

| Area | Change | Evidence boundary |
| --- | --- | --- |
| First impression | CI/Python/model/demo badges, live app link, actual app screenshot | No static accuracy badge or invented benchmark ranking |
| Results | rc2 comparison and validation recommendation before course history | All numbers read from M3_RELEASE.json; M2 remains a separate linked experiment |
| Setup | Dataset-free release-demo command first; training/testing separately | Existing commands and dependency files retained |
| Status | M1/M2 Submitted, M3 PDF update pending, M4 Not started | Submission is team-confirmed, not inferred from commits |
| Model transparency | MODEL_CARD.md | Intended use, method, limitations, runtime/artifact contract |
| Collaboration | CONTRIBUTING.md | Branch/PR workflow, validation, attribution and local-artifact policy |
| Navigation | Short documentation paths and README links | PR #21 linked as an open contribution rather than a local main file |
| Older guides | Removed stale placeholder, submission and missing-validation statements | Historical experiments and measured results are preserved |

## Repository and ML evidence checked

- **Tests:** all 158 main-branch unit/integration/E2E tests passed locally. Ruff
  and pip check passed. Main's latest CI run was successful. This count excludes
  the seven new tests in unmerged PR #21.
- **Coverage:** pytest-cov reported 88% statement coverage over 676 measured
  statements (81 missed). app.py was 99%, inference.py 97%, ml_baseline.py 100%,
  and scripts/split_reviews.py 46%. A script with no imported execution can be
  absent from this report, so this is not coverage of every repository file or
  a reason to claim complete testing. Several remaining misses are CLI paths.
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
- **Presentation checks:** 39 local Markdown paths/anchors resolve; all four
  badge URLs returned SVGs. The screenshot is an unchanged byte-for-byte copy
  of the recorded main deployment capture, not a mockup.
- **Software tests vs scientific evidence:** synthetic tests check contracts and
  regressions; the real-data report provides the measured model scores. Neither
  is a substitute for the other. No real-data retraining was performed here.
- **Tracked-file hygiene:** no dataset/model binary or named secret file is
  tracked on main. A focused recognizable-token/private-key signature scan of
  tracked text found no matches, and available Git history contained no matching
  dataset/model/secret filenames. This is a limited check, not proof that every
  possible secret or personal detail has been ruled out.
- **Metadata:** repository is public; description and NLP topics already exist.
  Homepage is empty and GitHub reports no detected license. No settings were changed.
- **Attribution:** the team roster, Jacob's UI/data-script contributions, Sang's
  ML contributions and existing AI-assistance disclosure remain visible.

## Prioritized follow-up work

### A. Publication decisions and small serving fixes

**1. Confirm code and artifact licensing with the team.** No LICENSE is present.
Do not put an MIT/Apache/open-source badge on the repository without agreement
from the appropriate contributors. Public visibility and a code reuse license
are different; [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
explains this distinction. Dataset terms are separate. The Kaggle page did not
expose readable license text to this audit, and the SNAP source page does not
settle all redistribution questions. Do not infer data/model licensing from a
third-party repository. Acceptance: maintainers choose and record accurate reuse
terms and any dataset/artifact restrictions; only then add a corresponding badge.

**2. Add a zero-feature input guard in a separate serving PR.**
[inference.py](../inference.py) checks blank/long/markup-only input, but does not
check whether the chosen vectorizer recognizes any feature. Probes `🍪🍪🍪` and
`이 음식은 맛있고 신선해요` both yielded zero nonzero features for the audited
BoW LR, TF-IDF LR and MLP artifacts. Nonetheless each returned Positive, with
probabilities approximately 74.01%, 39.09% and 77.81%, respectively. These values
are **probe outputs**, not accuracy or language benchmarks. This establishes a
serving limitation, not a reason to retrain or silently alter rc2 weights.
Acceptance: zero-feature input produces a clear insufficient-vocabulary message;
normal inputs retain the same probabilities; unit/AppTest coverage exercises
both paths with synthetic artifacts. Do not claim a general language detector.

**3. Align serving explanation and initial selection.** The recommendation is
TF-IDF LR, while [app.py](../app.py) initially uses the first manifest entry,
BoW LR. The README now explains the difference. A future UI change could make
the recommended model explicit/default while preserving user choice. The UI's
“Confidence” label also needs a plain explanation that probabilities are not
validated reliability estimates. Acceptance: synthetic switching/cache tests
still pass; no score or model artifact changes; labels explain actual behavior.

**4. Review the Codespaces startup command before advertising that path.**
[devcontainer.json](../.devcontainer/devcontainer.json) starts Streamlit with
CORS/XSRF protections disabled. This is a development configuration, not evidence
that the deployed app uses those flags. A separate dev-environment fix should
test the normal protected configuration and document any demonstrated requirement
before keeping exceptions. The public quickstart here uses normal local launch.

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

**8. Finish and review the error-analysis draft independently.** PR #21 adds
paired error overlap, class/length diagnostics and eight inspected paraphrases.
Its local prerequisites include a retained canonical raw evaluation export,
which is intentionally not public. Before merging, review that provenance and
reproduction boundary. Do not market those cases as a representative prevalence
study or call the optional analysis draft a completed M4 deliverable.

### C. Reproduction and maintenance polish

**9. Add a complete serving-environment capture if long-term maintenance is a goal.**
requirements.txt pins direct packages; transitive dependencies are not locked.
Training, main and staging also use different Python versions. Current hashes
and successful serving probes are valuable but do not promise byte-identical
retraining on all platforms. A separately validated environment/container could
improve this. Avoid upgrading sklearn under existing joblib artifacts without
revalidation; [sklearn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html)
describes compatibility and trust constraints.

**10. Improve documentation consistency and CLI coverage in focused changes.**
The older ML/data guides' obsolete placeholder and missing-validation statements
are corrected here. Some milestone-relative historical plans remain, and the
requirements comment names rc1 while rc2 is current. Distinguish history from
current status rather than silently rewriting results. The clean-checkout smoke
check and local documentation-link check passed here; automating them and
improving CLI branch coverage are possible follow-ups. No badge should promise
100% coverage.
Avoid adding a package framework or experiment platform just for appearance.
The [documentation PR's CI run](https://github.com/its-spark-dev/amazon-review-sentiment-ai/actions/runs/45306214735)
also reports Node.js 20 deprecation warnings for the existing checkout/setup/
upload actions, which the runner forces onto Node.js 24. Review supported action
versions in a separate CI-maintenance change and rerun all jobs; do not confuse
this Actions runtime warning with the Python/model serving environment.

**11. Prepare repository metadata after maintainer review.** Suggested homepage:
`https://amazon-sentiment-ai.streamlit.app/`. Existing topics already cover the
stack. A social preview can use the real demo image or a simple project graphic;
do not imply university endorsement or invent an award. A citation file can be
added after contributors confirm names and preferred attribution. License,
social-preview, description/homepage and access settings were not modified here.

## Review and publication checks

Before sharing the portfolio link, review this documentation PR and PR #21
separately, keep model/metric claims tied to rc2, and confirm the demo is reachable
at the time of sharing. Keep M3 PDF updating and the M4 report/presentation as
course tasks; a polished README cannot stand in for their submission. For any
future serving fix, repeat model selection/prediction and error-path checks.

The screenshot added here is the team's genuine main-app capture from the
November 23 deployment verification. No third-party photos, logos, reviewer
identifiers or invented outputs were added. Image access and readable captions
matter more than decorative badges.

## Primary guidance consulted

- [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes): lead with purpose, use and navigation.
- [Hugging Face model-card guidance](https://huggingface.co/docs/hub/model-cards): document intended use, data, evaluation and limitations; this project is not claiming Hub publication.
- [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): training-only fitting and separation of selection/evaluation.
- [SNAP data description](https://snap.stanford.edu/data/web-FineFoods.html): dataset provenance and historical scope.

Implementation evidence comes from the reviewed source and canonical local
artifacts; the external guidance supplies review criteria, not evidence that
this particular model is reliable. The action list is prepared work, not a
claim that those improvements have already been implemented.
