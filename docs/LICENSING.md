# Licensing and reuse

## Project-authored work: MIT

The [MIT License](../LICENSE) applies to the contributors' original code, tests,
scripts, documentation, figures, screenshots and project-trained model artifacts,
to the extent the contributors hold rights in those materials. Copyright is
credited collectively to **Amazon Review Sentiment AI contributors (2026)**;
individual contributions remain attributed through Git history and team reports.
This does not transfer contributors' ownership to the repository account.

MIT permits use, modification and redistribution, including commercial use.
Retain the copyright and permission notice when distributing copies or substantial
portions of the covered work. The license includes the standard warranty and
liability disclaimer. These terms do not relicense third-party material.

## Trained model artifacts

The MIT grant expressly includes the team's project-trained model artifacts.
The approved [v0.2.0-rc2 release](https://github.com/its-spark-dev/amazon-review-sentiment-ai/releases/tag/v0.2.0-rc2)
contains `bow_lr.joblib`, `bow_nb.joblib`, `tfidf_lr.joblib`, `tfidf_mlp.joblib`
and `tfidf_nb.joblib`. Their approved bytes are identified in
[release_manifest.json](../models/release_manifest.json) and
[M3_RELEASE.json](M3_RELEASE.json). Include the MIT notice when redistributing
covered artifacts. This scope document can accompany distributions for clarity;
it adds no conditions to the MIT grant.

Licensing is declared here without repackaging or modifying existing model files,
their hashes or release assets. This is a grant for the team's material, not a
claim of rights over the training reviews or dependency implementations. Model
purpose, training provenance and limitations remain in the [model card](MODEL_CARD.md).

## Amazon review data: upstream CC0 designation

The Amazon Fine Food Reviews dataset is **excluded from the project's MIT grant**.
The [Kaggle dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
is designated **CC0: Public Domain** by its publisher. That designation was
verified on November 24, 2026 through Kaggle's
[official metadata API](https://www.kaggle.com/api/v1/datasets/view/snap/amazon-fine-food-reviews)
(`licenseName`: `CC0: Public Domain`). See the
[CC0 1.0 terms](https://creativecommons.org/publicdomain/zero/1.0/).

Downloaded reviews and local sampled, cleaned or split CSVs retain their upstream
data context; our processing scripts do not convert those reviews to MIT.
CC0 permits reuse, including commercial reuse, within the scope of the rights
waived. It does not waive other people's privacy/publicity or trademark rights,
and the upstream designation is not a project warranty about every review.

Repository policy remains to keep datasets, review identifiers and raw evaluation
exports out of Git. Use the [data guide](../data/README.md) and source download
instead. This repository policy is not an additional restriction on CC0.
Retain source information in research reports and cite McAuley and Leskovec,
*From amateurs to connoisseurs: modeling the evolution of user expertise through
online reviews*, WWW 2013, as requested by the
[SNAP source page](https://snap.stanford.edu/data/web-FineFoods.html). This research
attribution request is separate from CC0's legal permissions.

## Third-party materials

Dependencies retain their own licenses; installing them from requirements files
does not make them MIT-licensed project work. Preserve applicable license and
attribution notices when redistributing bundled dependencies or incorporated
third-party material. Existing third-party notices take precedence for their
respective components. Screenshots and project attribution do not grant rights
to third-party names/logos or imply university, Amazon or library endorsement.

Contributors should follow [CONTRIBUTING.md](../CONTRIBUTING.md) and identify
third-party material rather than assigning it the project's license.
