"""Describe frozen rc2 errors without fitting, selecting, or changing models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import joblib
from matplotlib.figure import Figure
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_baseline import CLASS_LABELS, evaluate_predictions  # noqa: E402
from scripts.run_m2_baseline import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('tfidf_lr', 'bow_nb', 'tfidf_mlp')
BINS = ('1–20', '21–50', '51–100', '101+')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(predictions):
    """Aggregate aligned prediction records; no text/IDs enter the returned summary.

    Overlap bit order is MODELS: 1 means wrong. Length uses whitespace tokens,
    not vectorizer tokens. Strata are descriptive, not tuning criteria.
    """
    base = predictions[MODELS[0]]
    if not base:
        raise ValueError('predictions must be nonempty')
    for name in MODELS:
        rows = predictions[name]
        if len(rows) != len(base):
            raise ValueError('prediction lengths differ')
        for i, (a, b) in enumerate(zip(base, rows)):
            if a['test_index'] != i or any(a[k] != b[k] for k in
                    ('test_index', 'review_id', 'text', 'true_label')):
                raise ValueError('prediction records are not aligned')
            if not isinstance(b['text'], str) or not b['text'].strip():
                raise ValueError('review text must be nonblank')
    truth = [r['true_label'] for r in base]
    labels = {n: [r['predicted_label'] for r in predictions[n]] for n in MODELS}
    metrics = {n: evaluate_predictions(truth, labels[n]) for n in MODELS}
    masks = [''.join('1' if labels[n][i] != truth[i] else '0' for n in MODELS)
             for i in range(len(base))]
    overlaps = Counter(masks)
    lengths = [len(r['text'].split()) for r in base]
    strata = []
    for label in CLASS_LABELS:
        for bucket, (lo, hi) in zip(BINS, ((1, 20), (21, 50), (51, 100), (101, float('inf')))):
            indices = [i for i, y in enumerate(truth) if y == label and lo <= lengths[i] <= hi]
            count = len(indices)
            errors = {n: sum(labels[n][i] != truth[i] for i in indices) for n in MODELS}
            strata.append({'class': label, 'length': bucket, 'support': count, 'errors': errors,
                           'error_rate': {n: errors[n] / count if count else None for n in MODELS}})
    pairwise = []
    for j, left in enumerate(MODELS):
        for right in MODELS[j + 1:]:
            pairwise.append({'left': left, 'right': right,
                'different_prediction': sum(a != b for a, b in zip(labels[left], labels[right])),
                'left_only_correct': sum(a == y and b != y for a, b, y in
                                         zip(labels[left], labels[right], truth)),
                'right_only_correct': sum(a != y and b == y for a, b, y in
                                          zip(labels[left], labels[right], truth))})
    return {'rows': len(base), 'class_order': list(CLASS_LABELS), 'model_order': list(MODELS),
            'class_support': dict(Counter(truth)), 'metrics': metrics,
            'error_counts': {n: sum(a != b for a, b in zip(truth, labels[n])) for n in MODELS},
            'overlap_masks': {mask: overlaps[mask] for mask in
                              ('000', '001', '010', '011', '100', '101', '110', '111')},
            'pairwise': pairwise, 'length_strata': strata}


def plot_summary(summary, output):
    fig = Figure(figsize=(11, 3.8), layout='constrained')
    axes = fig.subplots(1, 3)
    for ax, name in zip(axes, MODELS):
        matrix = np.array(summary['metrics'][name]['confusion_matrix'])
        support = matrix.sum(axis=1, keepdims=True)
        rates = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support != 0)
        ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f'{matrix[row, col]:,}\n{rates[row, col]:.1%}',
                        ha='center', va='center', color='white' if rates[row, col] > .55 else 'black')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_LABELS,
               yticklabels=CLASS_LABELS, xlabel='Predicted', ylabel='True', title=name)
    fig.suptitle('rc2 fixed test: counts and within-true-class percentages')
    fig.savefig(output / 'confusion_by_class.png', dpi=160)

    fig = Figure(figsize=(8, 4), layout='constrained')
    ax = fig.subplots()
    masks = list(summary['overlap_masks'])[1:]
    counts = [summary['overlap_masks'][m] for m in masks]
    bars = ax.bar(masks, counts, color='#4378a8')
    ax.bar_label(bars)
    ax.set(xlabel='Error mask (TF-IDF LR, BoW NB, MLP); 1 = incorrect', ylabel='Reviews',
           title=f"rc2 error overlap; all three correct: {summary['overlap_masks']['000']:,}")
    ax.set_ylim(0, max(counts) * 1.2 or 1)
    fig.savefig(output / 'error_overlap.png', dpi=160)

    fig = Figure(figsize=(11, 3.8), layout='constrained')
    for ax, label in zip(fig.subplots(1, 3), CLASS_LABELS):
        strata = [s for s in summary['length_strata'] if s['class'] == label]
        for name in MODELS:
            ax.plot(range(4), [s['error_rate'][name] if s['error_rate'][name] is not None
                              else np.nan for s in strata], marker='o', label=name)
        ax.set(xticks=range(4), xticklabels=[f"{s['length']}\nn={s['support']}" for s in strata],
               ylim=(0, 1), title=label, ylabel='Error rate', xlabel='Whitespace-token length')
        ax.legend(fontsize=8)
    fig.suptitle('Descriptive test diagnostics, conditioned on rating-derived class')
    fig.savefig(output / 'length_by_class.png', dpi=160)


def run(run_dir, test_path, output, case_indices=(), *, evaluation_path=None):
    """Predict with frozen artifacts; optionally cross-check a retained evaluation export.

    Only the canonical test CSV and approved models are required. This uses
    model.predict on already-cleaned Text to reproduce the reported classifier
    metrics, rather than applying the app's input guard or refitting anything.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be new or empty')
    run_dir = Path(run_dir)
    report = json.loads((ROOT / 'docs/M3_RELEASE.json').read_text())
    if sha256(test_path) != report['input_sha256']['test']:
        raise ValueError('test file does not match canonical rc2')
    test = load_split(test_path)
    if len(test) != report['test_size']:
        raise ValueError('test size differs')
    exported = None
    if evaluation_path is not None:
        analysis = json.loads(Path(evaluation_path).read_text())
        if analysis['metadata']['input_sha256'] != report['input_sha256']:
            raise ValueError('evaluation input provenance differs')
        exported = {n: analysis['runs'][n]['predictions'] for n in MODELS}
        summarize(exported)  # validate export alignment before comparing with canonical rows
    predictions = {}
    with threadpool_limits(limits=1):
        for name in MODELS:
            path = run_dir / f'{name}.joblib'
            if sha256(path) != report['model_sha256'][path.name]:
                raise ValueError(f'model hash differs: {name}')
            # Trusted team artifact with the approved hash; predict only, never fit.
            model = joblib.load(path)
            actual = model.predict(test.Text).tolist()
            if len(actual) != len(test):
                raise ValueError(f'prediction length differs: {name}')
            rows = [dict(test_index=i, review_id=str(row.Id), text=row.Text,
                         true_label=row.Sentiment, predicted_label=label)
                    for i, (row, label) in enumerate(zip(test.itertuples(index=False), actual))]
            if exported is not None:
                saved = exported[name]
                if len(saved) != len(rows) or any(
                    any(row[key] != old[key] for key in row)
                    for row, old in zip(rows, saved)
                ):
                    raise ValueError(f'evaluation rows or predictions differ: {name}')
            predictions[name] = rows
    summary = summarize(predictions)
    for name in MODELS:
        if summary['metrics'][name] != report['runs'][name]['metrics']:
            raise ValueError(f'canonical metrics differ: {name}')
    if any(i < 0 or i >= len(test) for i in case_indices):
        raise ValueError('case index outside test rows')
    summary['provenance'] = {'release': report['release_tag'], 'test_sha256': sha256(test_path),
        'evaluation_sha256': sha256(evaluation_path) if evaluation_path is not None else None,
        'model_sha256': {f'{n}.joblib': report['model_sha256'][f'{n}.joblib'] for n in MODELS},
        'method': 'Frozen model.predict on canonical cleaned Text; no training, selection or model writes.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    cases = [dict(case=f'E{j + 1}', **predictions[MODELS[0]][i],
                  predictions={n: predictions[n][i]['predicted_label'] for n in MODELS})
             for j, i in enumerate(case_indices)]
    # Raw case texts and identifiers stay only under the ignored output directory.
    (output / 'cases.local.json').write_text(json.dumps(cases, indent=2) + '\n')
    plot_summary(summary, output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/m3/release-rc2')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/test_reviews.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/m3/m4-error-analysis')
    parser.add_argument('--case-indices', type=int, nargs='*', default=[])
    parser.add_argument('--evaluation', type=Path, default=None,
                        help='Optional retained evaluation/analysis.json to cross-check; never required.')
    args = parser.parse_args()
    run(args.run_dir, args.test, args.output_dir, args.case_indices, evaluation_path=args.evaluation)
