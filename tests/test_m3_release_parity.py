"""Deterministic feature cutoff and shared training/inference text contract."""
from pathlib import Path
import tempfile
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from inference import predict_review
from m3_training import fit_features, vectorizer
from scripts.clean_reviews import clean_text


@pytest.mark.parametrize('representation', ['bow', 'tfidf'])
def test_feature_cutoff_ties_use_lexical_order(representation):
    texts = ['zebra apple', 'banana carrot', 'apple zebra']
    vec = vectorizer(representation, 3)
    fit_features(vec, texts)
    assert vec.get_feature_names_out().tolist() == ['apple', 'banana', 'zebra']
    other = vectorizer(representation, 3)
    fit_features(other, texts[::-1])
    assert vec.vocabulary_ == other.vocabulary_
    if representation == 'tfidf':
        np.testing.assert_allclose(vec.idf_, other.idf_)
    assert 'heldout' not in vec.vocabulary_
    assert type(vec) in (CountVectorizer, type(vectorizer('tfidf', 3)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'vectorizer.joblib'
        joblib.dump(vec, path)
        np.testing.assert_allclose(joblib.load(path).transform(texts).toarray(),
                                   vec.transform(texts).toarray())


@pytest.mark.parametrize('raw', [
    '<span title="awful">Excellent &amp; fresh!</span>',
    '  Good\n\tFOOD &lt;3  ',
    'plain review',
])
def test_inference_applies_same_cleaner_as_training(raw):
    model = Mock()
    model.predict.return_value = ['Positive']
    model.classes_ = np.array(['Negative', 'Neutral', 'Positive'])
    model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
    result = predict_review(model, raw)
    model.predict.assert_called_once_with([clean_text(raw)])
    model.predict_proba.assert_called_once_with([clean_text(raw)])
    assert result['label'] == 'Positive'


def test_markup_only_input_is_rejected_before_model_call():
    model = Mock()
    with pytest.raises(ValueError, match='HTML cleanup'):
        predict_review(model, '<br/> &nbsp; <p></p>')
    model.predict.assert_not_called()


def test_report_provenance_matches_manifest_and_current_training_source():
    import hashlib
    import json

    from inference import RELEASE_MANIFEST, REPORT_MODEL_RELEASE
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'docs/M3_RELEASE.json').read_text())
    manifest = json.loads(RELEASE_MANIFEST.read_text())
    assert provenance['release_tag'] == REPORT_MODEL_RELEASE
    assert provenance['model_sha256'] == manifest[REPORT_MODEL_RELEASE]
    for name, expected in provenance['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for run in provenance['runs'].values():
        assert sum(map(sum, run['metrics']['confusion_matrix'])) == provenance['test_size']
