"""Evaluate fixed M2 baselines on externally prepared CSV splits; never resplit."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m2_analysis import export_results, run_baselines  # noqa: E402
from ml_baseline import CLASS_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_split(path):
    """Read Id/Text/Sentiment without interpreting literal text as NA values."""
    frame = pd.read_csv(path, dtype={'Id': 'string', 'Text': 'string', 'Sentiment': 'string'},
                        keep_default_na=False)
    required = {'Id', 'Text', 'Sentiment'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{path}: required columns are {sorted(required)}')
    if frame.empty:
        raise ValueError(f'{path}: split is empty')
    for column in required:
        if frame[column].isna().any() or frame[column].str.strip().eq('').any():
            raise ValueError(f'{path}: blank {column}')
    if not frame['Id'].is_unique:
        raise ValueError(f'{path}: duplicate Id')
    if not frame['Sentiment'].isin(CLASS_LABELS).all():
        raise ValueError(f'{path}: unknown sentiment label')
    if 'Score' in frame:
        expected = pd.to_numeric(frame['Score'], errors='coerce').map(
            {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'})
        if not expected.eq(frame['Sentiment']).all():
            raise ValueError(f'{path}: Score/Sentiment mismatch')
    return frame


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate_split(train_path, test_path, *, data_description):
    """Validate the shared split, fit training only, and return M2 results."""
    train, test = load_split(train_path), load_split(test_path)
    for column in ('Id', 'Text'):
        if set(train[column]) & set(test[column]):
            raise ValueError(f'train/test overlap in {column}')
    result = run_baselines(
        train['Text'].tolist(), train['Sentiment'].tolist(),
        test['Text'].tolist(), test['Sentiment'].tolist(),
        review_ids=test['Id'].tolist(), data_description=data_description)
    result['metadata']['input_files'] = {
        name: {'filename': Path(path).name,
               'sha256': _sha256(path)}
        for name, path in [('train', train_path), ('test', test_path)]}
    result['metadata']['environment'] = {
        'python': platform.python_version(),
        **{name: version(name) for name in
           ['scikit-learn', 'numpy', 'scipy', 'pandas', 'joblib', 'matplotlib', 'streamlit']}}
    result['metadata']['evaluation_protocol'] = (
        'Fixed default NB settings; shared split unchanged; no test-based tuning or model selection. '
        'Features fitted on training Text only; Summary/Score are not input features.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/train_reviews.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/test_reviews.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/m2/real-baseline')
    parser.add_argument('--data-description', default='Amazon Group Psi - shared seed48 split')
    parser.add_argument('--model-output', type=Path, help='Optional local BoW NB artifact for app')
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('output directory must be new or empty')
    if args.model_output is not None and args.model_output.exists():
        parser.error('refusing to overwrite an existing model artifact')
    result = evaluate_split(args.train, args.test, data_description=args.data_description)
    export_results(result, args.output_dir)
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        # Predeclared demonstration artifact, not selected by test performance.
        joblib.dump(result['runs']['bow_nb']['estimator'], args.model_output)
    versions = result['metadata']['environment']
    (args.output_dir / 'requirements-run.txt').write_text(
        '\n'.join(f'{name}=={value}' for name, value in versions.items() if name != 'python') + '\n')
    print(json.dumps({name: run['metrics'] for name, run in result['runs'].items()}, indent=2))
    print(f'Results saved locally: {args.output_dir}')


if __name__ == '__main__':
    main()
