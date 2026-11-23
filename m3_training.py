"""Validation-selected M3 models; the external test split never selects settings."""
from copy import deepcopy
import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ml_baseline import CLASS_LABELS, evaluate_predictions
from scripts.split_reviews import split


def vectorizer(representation, max_features):
    factory = {'bow': CountVectorizer, 'tfidf': TfidfVectorizer}[representation]
    return factory(max_features=max_features, dtype=np.float64)


def disjoint(left, right):
    for column in ('Id', 'Text'):
        if set(left[column]) & set(right[column]):
            raise ValueError(f'partitions overlap in {column}')


def fit_lr(x, y, c, weight):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = LogisticRegression(C=c, class_weight=weight, max_iter=5000).fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Logistic Regression did not converge within 5000 iterations')
    return model


def train_mlp(x, y, vx=None, vy=None, *, epochs=15, seed=48):
    """One fixed 32-ReLU architecture. Select epoch by validation macro-F1.

    Curves use unregularized cross-entropy on full inner training/validation
    partitions after each epoch, not sklearn's batch-averaged regularized loss.
    Without validation, refit for the already selected epoch count.
    """
    model = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                          alpha=0.001, batch_size=min(256, len(y)),
                          learning_rate_init=0.001, random_state=seed, shuffle=True)
    history, best, best_score, best_epoch = [], None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.partial_fit(x, y, classes=list(CLASS_LABELS))
        if vx is not None:
            score = evaluate_predictions(vy, model.predict(vx))['macro_f1']
            history.append({'epoch': epoch,
                            'train_loss': float(log_loss(y, model.predict_proba(x), labels=model.classes_)),
                            'validation_loss': float(log_loss(vy, model.predict_proba(vx), labels=model.classes_)),
                            'validation_macro_f1': score})
            if score > best_score:  # deterministic earliest-epoch tie handling
                best, best_score, best_epoch = deepcopy(model), score, epoch
    return (best, history, best_epoch) if vx is not None else (model, [], epochs)


def select_models(training, *, seed=48, max_features=10000, epochs=15):
    """Only outer training enters this function. Group-safe 80/20 inner split."""
    if epochs < 1 or max_features < 1:
        raise ValueError('epochs and max_features must be positive')
    fit, validation = split(training, test_size=0.2, seed=seed)
    disjoint(fit, validation)
    if set(fit.Sentiment) != set(CLASS_LABELS) or set(validation.Sentiment) != set(CLASS_LABELS):
        raise ValueError('inner partitions must contain all classes')
    selection = {'seed': seed, 'max_features': max_features, 'max_epochs': epochs,
                 'inner_fit_ids': fit.Id.tolist(), 'validation_ids': validation.Id.tolist(),
                 'criterion': 'validation macro-F1; first candidate/earliest epoch wins ties',
                 'lr_candidates': {}, 'chosen': {}, 'mlp_history': [], 'validation_scores': {}}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        majority = DummyClassifier(strategy='most_frequent').fit([[0]] * len(fit), fit.Sentiment)
        selection['validation_scores']['majority'] = evaluate_predictions(
            validation.Sentiment, majority.predict([[0]] * len(validation)))['macro_f1']
        for representation in ('bow', 'tfidf'):
            vec = vectorizer(representation, max_features)
            x = vec.fit_transform(fit.Text)
            vx = vec.transform(validation.Text)
            nb = MultinomialNB(alpha=1.0).fit(x, fit.Sentiment)
            selection['validation_scores'][representation + '_nb'] = evaluate_predictions(
                validation.Sentiment, nb.predict(vx))['macro_f1']
            candidates = []
            for c in (0.5, 2.0):
                for weight in (None, 'balanced'):
                    model = fit_lr(x, fit.Sentiment, c, weight)
                    score = evaluate_predictions(validation.Sentiment, model.predict(vx))['macro_f1']
                    candidates.append({'C': c, 'class_weight': weight, 'validation_macro_f1': score})
            name = representation + '_lr'
            selection['lr_candidates'][name] = candidates
            selection['chosen'][name] = max(candidates, key=lambda row: row['validation_macro_f1'])
            selection['validation_scores'][name] = selection['chosen'][name]['validation_macro_f1']
            if representation == 'tfidf':
                _, history, epoch = train_mlp(x, fit.Sentiment, vx, validation.Sentiment,
                                              epochs=epochs, seed=seed)
                selection['mlp_history'] = history
                selection['chosen']['tfidf_mlp'] = {'epochs': epoch,
                    'validation_macro_f1': history[epoch - 1]['validation_macro_f1']}
    selection['validation_scores']['tfidf_mlp'] = selection['chosen']['tfidf_mlp']['validation_macro_f1']
    selection['recommended_model'] = max(selection['validation_scores'], key=selection['validation_scores'].get)
    selection['selection_seconds'] = time.perf_counter() - started
    return selection


def final_comparison(training, test, selection):
    """Refit selected settings on full outer train, then evaluate fixed test.

    Timings include each model's own vectorizer fit/transform of TRAIN and
    classifier fit, excluding test prediction, serialization, and model selection.
    All fitting uses one numerical-library thread for comparable local timings.
    """
    disjoint(training, test)
    runs = {}
    with threadpool_limits(limits=1):
        for name in ('majority', 'bow_nb', 'tfidf_nb', 'bow_lr', 'tfidf_lr', 'tfidf_mlp'):
            started = time.perf_counter()
            if name == 'majority':
                model = DummyClassifier(strategy='most_frequent').fit([[0]] * len(training), training.Sentiment)
                vocabulary_size = 0
            else:
                representation, kind = name.split('_')
                vec = vectorizer(representation, selection['max_features'])
                x = vec.fit_transform(training.Text)
                if kind == 'nb':
                    classifier = MultinomialNB(alpha=1.0).fit(x, training.Sentiment)
                elif kind == 'lr':
                    cfg = selection['chosen'][name]
                    classifier = fit_lr(x, training.Sentiment, cfg['C'], cfg['class_weight'])
                else:
                    classifier, _, _ = train_mlp(x, training.Sentiment,
                        epochs=selection['chosen'][name]['epochs'], seed=selection['seed'])
                model = Pipeline([('vectorizer', vec), ('classifier', classifier)])
                vocabulary_size = len(vec.vocabulary_)
            elapsed = time.perf_counter() - started
            predicted = model.predict([[0]] * len(test) if name == 'majority' else test.Text).tolist()
            rows = [dict(test_index=i, review_id=str(r.Id), text=r.Text,
                         true_label=r.Sentiment, predicted_label=predicted[i])
                    for i, r in enumerate(test.itertuples(index=False))]
            runs[name] = {'estimator': model,
                          'metrics': evaluate_predictions(test.Sentiment, predicted),
                          'training_seconds': elapsed, 'vocabulary_size': vocabulary_size,
                          'predictions': rows, 'errors': [r for r in rows if r['true_label'] != r['predicted_label']],
                          'features': None}
    return runs
