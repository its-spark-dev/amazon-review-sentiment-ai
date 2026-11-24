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
