"""Select M3 settings on inner validation; evaluate the unchanged shared test."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import joblib
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m2_analysis import export_results  # noqa: E402
from m3_training import disjoint, final_comparison, select_models  # noqa: E402
from scripts.run_m2_baseline import _sha256, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(train_path, test_path, output, *, epochs=15, max_features=10000, seed=48):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be a new or empty directory')
    hashes = {'train': _sha256(train_path), 'test': _sha256(test_path)}
    training, test = load_split(train_path), load_split(test_path)
    disjoint(training, test)
    print('Selecting LR settings and MLP epoch on training-internal validation...', flush=True)
    selection = select_models(training, seed=seed, epochs=epochs, max_features=max_features)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze selection BEFORE fitting final models or evaluating test labels.
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print('Selection frozen. Refitting full train and evaluating fixed test...', flush=True)
    runs = final_comparison(training, test, selection)
    if hashes != {'train': _sha256(train_path), 'test': _sha256(test_path)}:
        raise RuntimeError('input files changed during run')
    metadata = {'data_description': 'M3 fixed outer split; validation-selected settings',
                'input_sha256': hashes, 'train_size': len(training), 'test_size': len(test),
                'selection': selection,
                'feature_tie_break': 'descending training token count, then lexical token order',
                'timing_protocol': 'single-thread feature fitting + classifier fitting; excludes selection/test/export',
                'environment': {'python': platform.python_version(), **{n: version(n) for n in
                    ['scikit-learn', 'numpy', 'scipy', 'pandas', 'joblib', 'matplotlib', 'threadpoolctl']}}}
    # Export into an empty subdirectory because selection.json already exists.
    export_results({'metadata': metadata, 'runs': runs}, output / 'evaluation')
    figure = Figure(figsize=(7, 4), layout='constrained')
    ax = figure.subplots()
    history = selection['mlp_history']
    for key in ('train_loss', 'validation_loss'):
        ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    ax.axvline(selection['chosen']['tfidf_mlp']['epochs'], linestyle=':', color='gray', label='selected epoch')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy (no regularization term)', title='MLP inner-training / validation loss')
    ax.legend()
    figure.savefig(output / 'mlp_loss.png', dpi=150)
    rows = ['| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Training seconds |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, result in runs.items():
        m = result['metrics']
        rows.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {result['training_seconds']:.3f} |")
        if name != 'majority':
            joblib.dump(result['estimator'], output / f'{name}.joblib')
    (output / 'comparison.md').write_text('\n'.join(rows) + '\n')
    (output / 'requirements-run.txt').write_text('\n'.join(
        f'{n}=={v}' for n, v in metadata['environment'].items() if n != 'python') + '\n')
    model_hashes = {p.name: _sha256(p) for p in sorted(output.glob('*.joblib'))}
    (output / 'model_hashes.json').write_text(json.dumps(model_hashes, indent=2) + '\n')
    print('\n'.join(rows), flush=True)
    return selection, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/train_reviews.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/test_reviews.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/m3/comparison')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max-features', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=48)
    args = parser.parse_args()
    run(args.train, args.test, args.output_dir, epochs=args.epochs,
        max_features=args.max_features, seed=args.seed)


if __name__ == '__main__':
    main()
