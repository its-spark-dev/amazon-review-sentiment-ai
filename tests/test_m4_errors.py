"""Known-label error overlap and class/length denominators, without real data."""
from copy import deepcopy

import pytest

from scripts.analyze_m4_errors import MODELS, summarize


def records():
    truth = ['Negative', 'Neutral', 'Positive', 'Neutral']
    labels = [('Negative', 'Positive', 'Positive', 'Negative'),
              ('Positive', 'Neutral', 'Positive', 'Negative'),
              ('Negative', 'Positive', 'Neutral', 'Positive')]
    return {name: [dict(test_index=i, review_id=str(i), text='word ' * length,
                        true_label=y, predicted_label=p)
                   for i, (y, p, length) in enumerate(zip(truth, pred, [20, 21, 50, 101]))]
            for name, pred in zip(MODELS, labels)}


def test_error_overlap_and_pairwise_are_exact_and_input_is_unchanged():
    rows = records()
    before = deepcopy(rows)
    result = summarize(rows)
    assert rows == before
    assert result['error_counts'] == dict(zip(MODELS, [2, 2, 3]))
    assert result['overlap_masks'] == {
        '000': 0, '001': 1, '010': 1, '011': 0,
        '100': 0, '101': 1, '110': 0, '111': 1}
    assert result['pairwise'][0] == {'left': 'tfidf_lr', 'right': 'bow_nb',
        'different_prediction': 2, 'left_only_correct': 1, 'right_only_correct': 1}


def test_length_boundaries_condition_on_true_class_and_empty_bins_are_missing():
    result = summarize(records())
    bins = {(s['class'], s['length']): s for s in result['length_strata']}
    assert bins['Negative', '1–20']['support'] == 1
    assert bins['Neutral', '21–50']['support'] == 1
    assert bins['Positive', '21–50']['support'] == 1
    assert bins['Neutral', '101+']['error_rate']['tfidf_lr'] == 1
    assert bins['Neutral', '51–100']['error_rate']['tfidf_lr'] is None
    assert bins['Negative', '1–20']['errors']['bow_nb'] == 1


@pytest.mark.parametrize('problem', ['text', 'order', 'length', 'label'])
def test_inconsistent_exports_are_rejected(problem):
    rows = records()
    if problem == 'text':
        rows['bow_nb'][0]['text'] = 'different review'
    elif problem == 'order':
        rows['bow_nb'].reverse()
    elif problem == 'length':
        rows['bow_nb'].pop()
    else:
        rows['bow_nb'][0]['predicted_label'] = 'unknown'
    with pytest.raises(ValueError):
        summarize(rows)


def test_empty_export_is_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        summarize({n: [] for n in MODELS})


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    """Tiny synthetic artifacts with their own canonical report; no real data."""
    import json

    import joblib
    import pandas as pd
    from sklearn.pipeline import Pipeline

    from ml_baseline import create_nb_pipeline, evaluate_predictions
    from scripts import analyze_m4_errors as analysis

    texts = ['awful stale', 'ordinary average', 'fresh tasty', 'fresh delicious']
    labels = ['Negative', 'Neutral', 'Positive', 'Positive']
    frame = pd.DataFrame({'Id': ['1', '2', '3', '4'], 'Text': texts, 'Sentiment': labels})
    test = tmp_path / 'test.csv'
    frame.to_csv(test, index=False)
    models = tmp_path / 'models'
    models.mkdir()
    report = {'release_tag': 'synthetic', 'input_sha256': {'test': analysis.sha256(test)},
              'test_size': 4, 'model_sha256': {}, 'runs': {}}
    export = {'metadata': {'input_sha256': report['input_sha256']}, 'runs': {}}
    for name in MODELS:
        model = create_nb_pipeline('bow').fit(texts, labels)
        path = models / f'{name}.joblib'
        joblib.dump(model, path)
        predicted = model.predict(texts).tolist()
        report['model_sha256'][path.name] = analysis.sha256(path)
        report['runs'][name] = {'metrics': evaluate_predictions(labels, predicted)}
        export['runs'][name] = {'predictions': [dict(test_index=i, review_id=str(row.Id),
            text=row.Text, true_label=row.Sentiment, predicted_label=label)
            for i, (row, label) in enumerate(zip(frame.itertuples(index=False), predicted))]}
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/M3_RELEASE.json').write_text(json.dumps(report))
    export_path = tmp_path / 'evaluation.json'
    export_path.write_text(json.dumps(export))
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)

    def no_fit(*args, **kwargs):
        pytest.fail('analysis must never train a pipeline')
    monkeypatch.setattr(Pipeline, 'fit', no_fit)
    return analysis, models, test, export_path, report


def test_reproduce_without_private_export_and_optional_export_agrees(frozen_run, tmp_path):
    import json

    analysis, models, test, export, _ = frozen_run
    fresh = analysis.run(models, test, tmp_path / 'fresh', [0, 2])
    checked = analysis.run(models, test, tmp_path / 'checked', [0, 2], evaluation_path=export)
    assert {k: v for k, v in fresh.items() if k != 'provenance'} == {
        k: v for k, v in checked.items() if k != 'provenance'}
    assert fresh['provenance']['evaluation_sha256'] is None
    assert checked['provenance']['evaluation_sha256'] == analysis.sha256(export)
    assert 'review_id' not in json.dumps(fresh) and 'fresh tasty' not in json.dumps(fresh)
    cases = json.loads((tmp_path / 'fresh/cases.local.json').read_text())
    assert [row['review_id'] for row in cases] == ['1', '3']
    for filename in ('confusion_by_class.png', 'error_overlap.png', 'length_by_class.png'):
        assert (tmp_path / 'fresh' / filename).is_file()


@pytest.mark.parametrize('problem', ['test_hash', 'model_hash', 'metrics', 'export_input', 'export_row', 'export_prediction'])
def test_run_rejects_noncanonical_inputs(frozen_run, tmp_path, problem):
    import json

    analysis, models, test, export, report = frozen_run
    evaluation = None
    if problem == 'test_hash':
        test.write_text(test.read_text() + '\n')
    elif problem == 'model_hash':
        (models / 'bow_nb.joblib').write_bytes(b'not an approved artifact')
    elif problem == 'metrics':
        report['runs']['bow_nb']['metrics']['accuracy'] = 0.0
        (tmp_path / 'docs/M3_RELEASE.json').write_text(json.dumps(report))
    else:
        saved = json.loads(export.read_text())
        if problem == 'export_input':
            saved['metadata']['input_sha256'] = {}
        elif problem == 'export_row':
            for name in MODELS:
                saved['runs'][name]['predictions'][0]['text'] = 'different text'
        else:
            saved['runs']['bow_nb']['predictions'][0]['predicted_label'] = 'Positive'
        export.write_text(json.dumps(saved))
        evaluation = export
    output = tmp_path / 'bad-output'
    with pytest.raises(ValueError):
        analysis.run(models, test, output, evaluation_path=evaluation)
    assert not output.exists()
