"""M2 orchestration and reporting for already prepared, externally split texts.

No loading, cleaning, sampling, splitting, tuning, or model serialization.
"""

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from ml_baseline import CLASS_LABELS, create_nb_pipeline, evaluate_predictions


def _labels(values):
    if isinstance(values, str):
        raise ValueError("labels must be a sequence, not a string")
    values = list(values)
    if not values or any(value not in CLASS_LABELS for value in values):
        raise ValueError(f"nonempty labels must belong to {CLASS_LABELS}")
    return values


def _texts(values, labels):
    if isinstance(values, str):
        raise ValueError("texts must be a sequence, not a string")
    values = list(values)
    if len(values) != len(labels) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("texts must be nonblank strings aligned with labels")
    return values


def informative_features(pipeline, top_n=10):
    """Rank features by log P(token|Positive) - log P(token|Negative).

    Returns up to top_n strictly positive contrasts per sentiment direction.
    Neutral remains in the trained classifier but is not part of this pairwise
    explanation. Scores are model associations, not causal effects/frequencies.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    vectorizer = pipeline.named_steps['vectorizer']
    classifier = pipeline.named_steps['classifier']
    check_is_fitted(classifier)
    classes = classifier.classes_.tolist()
    if not {'Positive', 'Negative'}.issubset(classes):
        raise ValueError("feature contrasts require Positive and Negative training labels")
    names = vectorizer.get_feature_names_out().tolist()
    contrast = (classifier.feature_log_prob_[classes.index('Positive')]
                - classifier.feature_log_prob_[classes.index('Negative')])
    result = {'vocabulary_size': len(names),
              'method': 'log P(token|class) - log P(token|opposite sentiment)'}
    for label, sign in [('Positive', 1), ('Negative', -1)]:
        ranked = sorted(
            [(name, float(sign * score)) for name, score in zip(names, contrast)
             if sign * score > 0], key=lambda pair: (-pair[1], pair[0])
        )[:top_n]
        result[label] = [{'feature': name, 'score': score} for name, score in ranked]
    return result


def run_baselines(train_texts, train_labels, test_texts, test_labels, *,
                  data_description, review_ids=None, top_n=10):
    """Fit BoW/TF-IDF NB and train-majority baseline on the SAME supplied split.

    Caller owns duplicate-safe splitting and data provenance. All three training
    labels are required; the held-out set may omit classes. review_ids are unique
    nonblank strings aligned with test rows, or omitted for index-only tracing.
    Returns {'metadata': ..., 'runs': ...}; each run includes its fitted estimator,
    metrics, prediction rows, error rows, and (NB only) informative features.
    Nothing is written until export_results is called. Test texts never fit an
    estimator. data_description should identify synthetic vs real data honestly.
    """
    if not isinstance(data_description, str) or not data_description.strip():
        raise ValueError('data_description must identify the input data')
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError('top_n must be a positive integer')
    train_labels, test_labels = _labels(train_labels), _labels(test_labels)
    train_texts, test_texts = _texts(train_texts, train_labels), _texts(test_texts, test_labels)
    if set(train_labels) != set(CLASS_LABELS):
        raise ValueError('training labels must include all three sentiment classes')
    if review_ids is not None:
        if isinstance(review_ids, str):
            raise ValueError('review_ids must be a sequence')
        review_ids = list(review_ids)
        if (len(review_ids) != len(test_labels)
                or any(not isinstance(i, str) or not i.strip() for i in review_ids)
                or len(set(review_ids)) != len(review_ids)):
            raise ValueError('review_ids must be unique nonblank strings aligned with test rows')
    else:
        review_ids = [''] * len(test_labels)

    result = {'metadata': {
        'data_description': data_description, 'sklearn_version': sklearn.__version__,
        'train_size': len(train_labels), 'test_size': len(test_labels),
        'class_order': list(CLASS_LABELS),
        'train_class_counts': {c: train_labels.count(c) for c in CLASS_LABELS},
        'test_class_counts': {c: test_labels.count(c) for c in CLASS_LABELS},
        'configuration': {'features': 'lowercase word unigrams; no stop-word removal',
                          'tfidf': 'smoothed IDF; L2 normalization', 'nb_alpha': 1.0},
    }, 'runs': {}}
    for name in ('majority', 'bow_nb', 'tfidf_nb'):
        if name == 'majority':
            estimator = DummyClassifier(strategy='most_frequent')
            estimator.fit([[0]] * len(train_labels), train_labels)
            predictions = estimator.predict([[0]] * len(test_labels)).tolist()
            features = None
        else:
            estimator = create_nb_pipeline(name.removesuffix('_nb'))
            estimator.fit(train_texts, train_labels)
            predictions = estimator.predict(test_texts).tolist()
            features = informative_features(estimator, top_n)
        rows = [dict(test_index=i, review_id=review_ids[i], text=text,
                     true_label=true, predicted_label=pred)
                for i, (text, true, pred) in enumerate(zip(test_texts, test_labels, predictions))]
        result['runs'][name] = {
            'estimator': estimator, 'metrics': evaluate_predictions(test_labels, predictions),
            'predictions': rows,
            'errors': [row for row in rows if row['true_label'] != row['predicted_label']],
            'features': features,
        }
    return result


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(result, output_dir):
    """Export JSON/CSV and PNG charts into a new or EMPTY local directory.

    Exports all errors, without guessing error patterns or selecting a model.
    Raw review text is included in CSV files: keep outputs local/ignored. Fitted
    estimators are excluded from JSON and are never serialized to model files.
    """
    from matplotlib.figure import Figure

    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output_dir must be new or empty to avoid overwriting results')
    output.mkdir(parents=True, exist_ok=True)
    report = {'metadata': result['metadata'], 'runs': {
        name: {key: value for key, value in run.items() if key != 'estimator'}
        for name, run in result['runs'].items()}}
    (output / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    metric_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    _write_csv(output / 'metrics.csv', [
        {'model': name, **{key: run['metrics'][key] for key in metric_names}}
        for name, run in result['runs'].items()], ['model', *metric_names])
    for name, run in result['runs'].items():
        for key in ('predictions', 'errors'):
            _write_csv(output / f'{name}_{key}.csv', run[key],
                       ['test_index', 'review_id', 'text', 'true_label', 'predicted_label'])
        figure = Figure(figsize=(6, 5), layout='constrained')
        ax = figure.subplots()
        matrix = run['metrics']['confusion_matrix']
        ax.imshow(matrix, cmap='Blues', vmin=0)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_LABELS,
               yticklabels=CLASS_LABELS, xlabel='Predicted label', ylabel='True label',
               title=fill(f'{name} | {result["metadata"]["data_description"]}', 55))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                        color='white' if matrix[i][j] > max(map(max, matrix)) / 2 else 'black')
        figure.savefig(output / f'{name}_confusion.png', dpi=140)
        if run['features'] is not None:
            figure = Figure(figsize=(10, 6), layout='constrained')
            axes = figure.subplots(1, 2)
            for label, ax in zip(('Negative', 'Positive'), axes):
                entries = list(reversed(run['features'][label]))
                ax.barh([e['feature'] for e in entries], [e['score'] for e in entries])
                ax.set(title=f'{label} vs opposite sentiment', xlabel='Log-probability contrast')
            figure.suptitle(fill(f'{name} | {result["metadata"]["data_description"]}', 90))
            figure.savefig(output / f'{name}_features.png', dpi=140)


def main():
    parser = argparse.ArgumentParser(description='Synthetic-only M2 smoke example; no real data loaded.')
    parser.add_argument('--synthetic-demo', action='store_true', required=True)
    parser.add_argument('--output-dir', default='outputs/m2/synthetic-demo')
    args = parser.parse_args()
    result = run_baselines(
        ['awful stale terrible', 'terrible spoiled food', 'ordinary average okay',
         'average ordinary food', 'excellent delicious fresh', 'delicious tasty excellent',
         'excellent fresh tasty'],
        ['Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Positive', 'Positive'],
        ['awful stale', 'ordinary okay', 'excellent fresh', 'delicious excellent'],
        ['Negative', 'Neutral', 'Positive', 'Negative'],
        review_ids=['synthetic-1', 'synthetic-2', 'synthetic-3', 'synthetic-4'],
        data_description='SYNTHETIC demo - not Amazon performance',
    )
    export_results(result, args.output_dir)
    print(f'Synthetic-only analysis written to {args.output_dir}; no real-data results.')


if __name__ == '__main__':
    main()
