"""CSV handoff tests using synthetic, independently supplied partitions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from scripts.run_m2_baseline import evaluate_split, load_split


class PreparedSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / 'train.csv'
        self.test = self.root / 'test.csv'
        pd.DataFrame({'Id': ['01', '02', '03'], 'Text': ['bad stale', 'okay plain', 'good tasty'],
                      'Sentiment': ['Negative', 'Neutral', 'Positive'], 'Score': [1, 3, 5]}).to_csv(self.train, index=False)
        pd.DataFrame({'Id': ['04'], 'Text': ['bad horrible'], 'Sentiment': ['Negative'],
                      'Score': [2]}).to_csv(self.test, index=False)

    def test_csv_handoff_and_provenance(self):
        self.assertEqual(load_split(self.train)['Id'].tolist(), ['01', '02', '03'])
        result = evaluate_split(self.train, self.test, data_description='SYNTHETIC')
        self.assertEqual(result['metadata']['train_size'], 3)
        self.assertEqual(len(result['metadata']['input_files']['train']['sha256']), 64)
        self.assertEqual(result['runs']['bow_nb']['predictions'][0]['review_id'], '04')
        self.assertNotIn('horrible', result['runs']['bow_nb']['estimator'].named_steps['vectorizer'].vocabulary_)

    def test_cross_partition_overlap_rejected(self):
        for column, value in [('Id', '01'), ('Text', 'bad stale')]:
            frame = pd.read_csv(self.test, dtype=str)
            frame.loc[0, column] = value
            frame.to_csv(self.test, index=False)
            with self.subTest(column=column), self.assertRaises(ValueError):
                evaluate_split(self.train, self.test, data_description='SYNTHETIC')
            frame.loc[0, 'Id'] = '04'
            frame.loc[0, 'Text'] = 'bad horrible'
            frame.to_csv(self.test, index=False)

    def test_invalid_csv_rejected(self):
        original = pd.read_csv(self.train, dtype=str)
        cases = [original.drop(columns='Sentiment'), original.iloc[:0],
                 pd.concat([original, original.iloc[:1]], ignore_index=True)]
        for column, value in [('Text', ' '), ('Id', ''), ('Sentiment', 'unknown'), ('Score', '5')]:
            frame = original.copy()
            frame.loc[0, column] = value
            cases.append(frame)
        for frame in cases:
            frame.to_csv(self.train, index=False)
            with self.subTest(frame=frame.shape), self.assertRaises(ValueError):
                load_split(self.train)

    def test_cli_exports_and_preserves_inputs(self):
        before = self.train.read_bytes(), self.test.read_bytes()
        command = [sys.executable, 'scripts/run_m2_baseline.py', '--train', str(self.train),
                   '--test', str(self.test), '--output-dir', str(self.root / 'results'),
                   '--model-output', str(self.root / 'model.joblib'), '--data-description', 'SYNTHETIC']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        metadata = json.loads((self.root / 'results/analysis.json').read_text())['metadata']
        self.assertEqual(metadata['data_description'], 'SYNTHETIC')
        self.assertTrue((self.root / 'model.joblib').is_file())
        self.assertEqual(before, (self.train.read_bytes(), self.test.read_bytes()))
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
