"""Synthetic M2 integration checks; no real data or external services."""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ml_baseline import CLASS_LABELS, create_nb_pipeline
from m2_analysis import export_results, informative_features, run_baselines


class M2Tests(unittest.TestCase):
    def setUp(self):
        self.train = ['bad sour', 'bad stale', 'okay plain',
                      'good fresh', 'good tasty', 'good great']
        self.labels = ['Negative', 'Negative', 'Neutral',
                       'Positive', 'Positive', 'Positive']
        self.test = ['bad sour', 'good great', 'okay plain', 'good secretword']
        self.truth = ['Negative', 'Negative', 'Neutral', 'Positive']

    def run_example(self, **overrides):
        args = dict(train_texts=self.train, train_labels=self.labels,
                    test_texts=self.test, test_labels=self.truth,
                    data_description='SYNTHETIC unit test',
                    review_ids=['r1', 'r2', 'r3', 'r4'])
        args.update(overrides)
        return run_baselines(**args)

    def test_train_majority_is_not_test_majority(self):
        result = self.run_example()
        run = result['runs']['majority']
        self.assertEqual([row['predicted_label'] for row in run['predictions']],
                         ['Positive'] * 4)
        self.assertEqual(run['metrics']['accuracy'], 0.25)
        self.assertEqual(run['metrics']['confusion_matrix'],
                         [[0, 0, 2], [0, 0, 1], [0, 0, 1]])
        self.assertAlmostEqual(run['metrics']['macro_f1'], 2 / 15)
        self.assertEqual(result['metadata']['train_class_counts']['Positive'], 3)

    def test_predictions_and_errors_preserve_traceability(self):
        result = self.run_example()
        for name in ('bow_nb', 'tfidf_nb'):
            with self.subTest(name=name):
                run = result['runs'][name]
                self.assertEqual(len(run['predictions']), 4)
                self.assertEqual([row['predicted_label'] for row in run['predictions']],
                                 (['Negative', 'Positive', 'Neutral', 'Positive'] if name == 'bow_nb'
                                  else ['Negative', 'Positive', 'Positive', 'Positive']))
                expected_errors = [{
                    'test_index': 1, 'review_id': 'r2', 'text': 'good great',
                    'true_label': 'Negative', 'predicted_label': 'Positive'}]
                if name == 'tfidf_nb':
                    expected_errors.append({
                        'test_index': 2, 'review_id': 'r3', 'text': 'okay plain',
                        'true_label': 'Neutral', 'predicted_label': 'Positive'})
                self.assertEqual(run['errors'], expected_errors)
                self.assertEqual(run['metrics']['accuracy'],
                                 0.75 if name == 'bow_nb' else 0.5)
                self.assertEqual(run['metrics']['class_order'], list(CLASS_LABELS))

    def test_vocabulary_and_idf_are_fitted_on_training_only(self):
        result = self.run_example()
        for representation in ('bow', 'tfidf'):
            with self.subTest(representation=representation):
                actual = result['runs'][representation + '_nb']['estimator']
                vectorizer = actual.named_steps['vectorizer']
                expected = create_nb_pipeline(representation).fit(self.train, self.labels)
                self.assertNotIn('secretword', vectorizer.vocabulary_)
                self.assertEqual(vectorizer.vocabulary_,
                                 expected.named_steps['vectorizer'].vocabulary_)
                if representation == 'tfidf':
                    self.assertTrue((vectorizer.idf_ ==
                                     expected.named_steps['vectorizer'].idf_).all())

    def test_feature_contrasts_match_hand_calculation(self):
        for representation in ('bow', 'tfidf'):
            pipeline = create_nb_pipeline(representation).fit(
                ['bad bad', 'okay okay', 'good good'], list(CLASS_LABELS))
            features = informative_features(pipeline, top_n=10)
            self.assertEqual(features['vocabulary_size'], 3)
            self.assertEqual([r['feature'] for r in features['Positive']], ['good'])
            self.assertEqual([r['feature'] for r in features['Negative']], ['bad'])
            # Counts: (2+1)/(0+1)=3. TF-IDF single-token rows have weight 1.
            expected = math.log(3 if representation == 'bow' else 2)
            self.assertAlmostEqual(features['Positive'][0]['score'], expected)
            self.assertAlmostEqual(features['Negative'][0]['score'], expected)

    def test_feature_limit_and_invalid_top_n(self):
        result = self.run_example(top_n=1)
        for name in ('bow_nb', 'tfidf_nb'):
            for label in ('Negative', 'Positive'):
                self.assertLessEqual(len(result['runs'][name]['features'][label]), 1)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_example(top_n=invalid)

    def test_features_require_both_polarities(self):
        pipeline = create_nb_pipeline().fit(['okay plain', 'great tasty'],
                                             ['Neutral', 'Positive'])
        with self.assertRaises(ValueError):
            informative_features(pipeline)

    def test_invalid_inputs_fail_before_training(self):
        cases = [
            {'train_texts': self.train[:-1]}, {'test_labels': []},
            {'test_texts': [' ', *self.test[1:]]},
            {'train_labels': ['unknown'] * 6},
            {'train_labels': ['Positive'] * 6},
            {'test_labels': ['bad'] * 4}, {'test_texts': 'one string'},
            {'data_description': ''}, {'review_ids': ['r1']},
            {'review_ids': ['same'] * 4}, {'review_ids': [1, 2, 3, 4]},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_example(**changes)

    def test_absent_test_classes_and_no_errors(self):
        result = self.run_example(test_texts=['good great'],
                                  test_labels=['Positive'], review_ids=None)
        for run in result['runs'].values():
            self.assertEqual(run['errors'], [])
            self.assertEqual(run['predictions'][0]['review_id'], '')
            self.assertEqual(run['metrics']['class_order'], list(CLASS_LABELS))
            self.assertAlmostEqual(run['metrics']['macro_f1'], 1 / 3)

    def test_export_contains_metrics_features_rows_and_valid_images(self):
        from PIL import Image

        result = self.run_example()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'analysis'
            export_results(result, output)
            report = json.loads((output / 'analysis.json').read_text())
            self.assertEqual(report['metadata']['data_description'], 'SYNTHETIC unit test')
            self.assertNotIn('estimator', report['runs']['bow_nb'])
            self.assertEqual(report['runs']['bow_nb']['features']['vocabulary_size'], 9)
            self.assertEqual(report['runs']['bow_nb']['metrics'],
                             result['runs']['bow_nb']['metrics'])
            with (output / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['model'] for r in rows], ['majority', 'bow_nb', 'tfidf_nb'])
            self.assertEqual(float(rows[0]['accuracy']), 0.25)
            with (output / 'bow_nb_errors.csv').open() as stream:
                errors = list(csv.DictReader(stream))
            self.assertEqual(errors[0]['review_id'], 'r2')
            for path in output.glob('*.png'):
                with Image.open(path) as image:
                    image.verify()
            self.assertEqual(len(list(output.glob('*.png'))), 5)
            with self.assertRaises(ValueError):
                export_results(result, output)

    def test_empty_error_export_keeps_headers(self):
        result = self.run_example(test_texts=['good great'],
                                  test_labels=['Positive'], review_ids=None)
        with tempfile.TemporaryDirectory() as directory:
            export_results(result, directory)
            with (Path(directory) / 'bow_nb_errors.csv').open() as stream:
                reader = csv.DictReader(stream)
                self.assertIn('true_label', reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_synthetic_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'demo'
            result = subprocess.run([
                sys.executable, '-m', 'm2_analysis', '--synthetic-demo',
                '--output-dir', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Synthetic-only', result.stdout)
            self.assertIn('SYNTHETIC', json.loads(
                (output / 'analysis.json').read_text())['metadata']['data_description'])


if __name__ == '__main__':
    unittest.main()
