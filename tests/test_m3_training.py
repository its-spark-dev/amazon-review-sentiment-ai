"""Synthetic validation isolation, loss history, and end-to-end M3 checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

import m3_training as m3
from scripts.run_m3_comparison import run


def frame():
    records = []
    for label, text in [('Negative', 'awful spoiled'), ('Neutral', 'ordinary okay'),
                        ('Positive', 'excellent delicious')]:
        for i in range(20):
            records.append(dict(Id=f'{label}-{i}', Text=f'{text} token{i}', Sentiment=label))
    return pd.DataFrame(records)


class M3Tests(unittest.TestCase):
    def test_inner_split_reproducible_group_safe_and_train_only_features(self):
        training = pd.concat([frame(), frame().iloc[:3].assign(Id=['dup1', 'dup2', 'dup3'])], ignore_index=True)
        seen = []
        factory = m3.vectorizer

        def tracked(*args):
            vec = factory(*args)
            fit = vec.fit_transform
            def record(texts, *a, **kw):
                seen.append(list(texts))
                return fit(texts, *a, **kw)
            vec.fit_transform = record
            return vec

        with patch('m3_training.vectorizer', side_effect=tracked):
            a = m3.select_models(training, epochs=2, max_features=30)
        b = m3.select_models(training, epochs=2, max_features=30)
        self.assertEqual(a['chosen'], b['chosen'])
        self.assertEqual(a['recommended_model'], max(a['validation_scores'], key=a['validation_scores'].get))
        self.assertEqual(a['validation_ids'], b['validation_ids'])
        inner = training[training.Id.isin(a['inner_fit_ids'])]
        validation = training[training.Id.isin(a['validation_ids'])]
        self.assertFalse(set(inner.Text) & set(validation.Text))
        self.assertFalse(set(inner.Id) & set(validation.Id))
        self.assertEqual(len(inner) + len(validation), len(training))
        for texts in seen:
            self.assertEqual(texts, inner.Text.tolist())

    def test_loss_history_and_epoch_selection(self):
        selection = m3.select_models(frame(), epochs=3, max_features=30)
        history = selection['mlp_history']
        self.assertEqual([r['epoch'] for r in history], [1, 2, 3])
        for row in history:
            self.assertGreaterEqual(row['train_loss'], 0)
            self.assertGreaterEqual(row['validation_loss'], 0)
        best = max(history, key=lambda r: r['validation_macro_f1'])
        self.assertEqual(selection['chosen']['tfidf_mlp']['epochs'], best['epoch'])
        for candidates in selection['lr_candidates'].values():
            self.assertEqual(len(candidates), 4)

    def test_final_refit_never_changes_selection_and_test_words_stay_unseen(self):
        train = frame()
        test = pd.DataFrame([dict(Id='heldout', Text='excellent neverseenword', Sentiment='Positive')])
        selection = m3.select_models(train, epochs=2, max_features=30)
        original = deepcopy(selection)
        runs = m3.final_comparison(train, test, selection)
        self.assertEqual(selection, original)
        self.assertEqual(len(runs), 6)
        for name, result in runs.items():
            self.assertGreaterEqual(result['training_seconds'], 0)
            self.assertEqual(len(result['predictions']), 1)
            self.assertEqual(result['metrics']['class_order'], ['Negative', 'Neutral', 'Positive'])
            if name != 'majority':
                model = result['estimator']
                self.assertNotIn('neverseenword', model.named_steps['vectorizer'].vocabulary_)
                self.assertAlmostEqual(model.predict_proba(test.Text).sum(), 1.0)

    def test_nonconvergence_is_not_silently_accepted(self):
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        def unconverged(*args, **kwargs):
            warnings.warn('iteration budget reached', ConvergenceWarning, stacklevel=2)
            return None
        with patch('m3_training.LogisticRegression') as factory:
            factory.return_value.fit.side_effect = unconverged
            with self.assertRaises(RuntimeError):
                m3.fit_lr([[1], [2]], ['Negative', 'Positive'], 0.5, None)

    def test_overlapping_partitions_and_bad_configuration_rejected(self):
        with self.assertRaises(ValueError):
            m3.disjoint(frame(), frame().iloc[:1])
        for kwargs in [dict(epochs=0), dict(max_features=0)]:
            with self.assertRaises(ValueError):
                m3.select_models(frame(), **kwargs)

    def test_end_to_end_artifacts_and_input_preservation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = root/'train.csv', root/'test.csv'
            frame().to_csv(train, index=False)
            pd.DataFrame([dict(Id='test1', Text='awful heldoutword', Sentiment='Negative')]).to_csv(test, index=False)
            before = train.read_bytes(), test.read_bytes()
            selection, runs = run(train, test, root/'out', epochs=2, max_features=30)
            self.assertEqual(before, (train.read_bytes(), test.read_bytes()))
            saved = json.loads((root/'out/selection.json').read_text())
            self.assertEqual(saved, selection)
            report = json.loads((root/'out/evaluation/analysis.json').read_text())
            self.assertEqual(report['metadata']['test_size'], 1)
            with Image.open(root/'out/mlp_loss.png') as image:
                image.verify()
            for name in ('bow_lr', 'tfidf_lr', 'tfidf_mlp'):
                model = joblib.load(root/f'out/{name}.joblib')
                self.assertEqual(model.predict(['awful heldoutword']).tolist(),
                                 [r['predicted_label'] for r in runs[name]['predictions']])
            with self.assertRaises(ValueError):
                run(train, test, root/'out', epochs=2, max_features=30)


if __name__ == '__main__':
    unittest.main()
