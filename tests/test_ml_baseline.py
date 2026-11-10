"""Small synthetic checks; no dataset, network access, or model files required."""

import json
import unittest

from ml_baseline import (
    CLASS_LABELS,
    create_nb_pipeline,
    create_vectorizer,
    evaluate_predictions,
    rating_to_sentiment,
)


class RatingTests(unittest.TestCase):
    def test_all_star_ratings(self):
        for rating, expected in enumerate(
            ["Negative", "Negative", "Neutral", "Positive", "Positive"], start=1
        ):
            with self.subTest(rating=rating):
                self.assertEqual(rating_to_sentiment(rating), expected)

    def test_out_of_range(self):
        for rating in (-1, 0, 6, 100):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                rating_to_sentiment(rating)

    def test_non_integer_values(self):
        for rating in (True, False, 3.0, 2.5, "3", None, float("nan"), [3]):
            with self.subTest(rating=rating), self.assertRaises(TypeError):
                rating_to_sentiment(rating)


class VectorizerTests(unittest.TestCase):
    def test_bow_counts_and_case(self):
        vectorizer = create_vectorizer("bow")
        matrix = vectorizer.fit_transform(["Tasty tasty food", "plain food"])
        self.assertEqual(vectorizer.get_feature_names_out().tolist(), ["food", "plain", "tasty"])
        self.assertEqual(matrix.toarray().tolist(), [[1, 0, 2], [1, 1, 0]])

    def test_tfidf_weights_and_normalization(self):
        matrix = create_vectorizer("tfidf").fit_transform(
            ["tasty tasty food", "plain food"]
        )
        self.assertTrue((matrix.data >= 0).all())
        self.assertTrue((matrix.data <= 1).all())
        for norm_squared in matrix.multiply(matrix).sum(axis=1).flat:
            self.assertAlmostEqual(float(norm_squared), 1.0)
        # Rare 'tasty' receives more weight than shared 'food' in the first text.
        self.assertGreater(matrix[0, 2], matrix[0, 0])

    def test_unseen_tokens_do_not_enter_vocabulary(self):
        for representation in ("bow", "tfidf"):
            with self.subTest(representation=representation):
                vectorizer = create_vectorizer(representation)
                self.assertFalse(hasattr(vectorizer, "vocabulary_"))
                vectorizer.fit(["tasty food", "plain food"])
                vocabulary = dict(vectorizer.vocabulary_)
                matrix = vectorizer.transform(["unseenword"])
                self.assertEqual(matrix.nnz, 0)
                self.assertEqual(vectorizer.vocabulary_, vocabulary)

    def test_invalid_representation(self):
        with self.assertRaises(ValueError):
            create_vectorizer("unsupported")


class PipelineTests(unittest.TestCase):
    def test_fit_and_predict_with_each_representation(self):
        texts = [
            "awful stale terrible", "terrible awful spoiled",
            "average ordinary okay", "okay average ordinary",
            "delicious fresh excellent", "excellent delicious tasty",
        ]
        labels = ["Negative", "Negative", "Neutral", "Neutral", "Positive", "Positive"]
        for representation in ("bow", "tfidf"):
            with self.subTest(representation=representation):
                pipeline = create_nb_pipeline(representation)
                pipeline.fit(texts, labels)
                vectorizer = pipeline.named_steps["vectorizer"]
                vocabulary = dict(vectorizer.vocabulary_)
                idf = vectorizer.idf_.copy() if representation == "tfidf" else None
                predictions = pipeline.predict([
                    "awful terrible unseenword", "ordinary okay", "excellent delicious",
                ])
                self.assertEqual(predictions.tolist(), list(CLASS_LABELS))
                self.assertEqual(vectorizer.vocabulary_, vocabulary)
                self.assertNotIn("unseenword", vectorizer.vocabulary_)
                if idf is not None:
                    self.assertTrue((idf == vectorizer.idf_).all())

    def test_factories_return_independent_unfitted_objects(self):
        first, second = create_nb_pipeline(), create_nb_pipeline()
        first.fit(["terrible food", "excellent food"], ["Negative", "Positive"])
        self.assertFalse(hasattr(second.named_steps["vectorizer"], "vocabulary_"))
        self.assertFalse(hasattr(second.named_steps["classifier"], "classes_"))


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_and_matrix_orientation(self):
        result = evaluate_predictions(
            ["Negative", "Negative", "Neutral", "Neutral", "Positive", "Positive"],
            ["Negative", "Positive", "Neutral", "Negative", "Positive", "Positive"],
        )
        self.assertEqual(set(result), {
            "accuracy", "macro_precision", "macro_recall", "macro_f1",
            "per_class", "class_order", "confusion_matrix",
        })
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_precision"], 13 / 18)
        self.assertAlmostEqual(result["macro_recall"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 59 / 90)
        self.assertEqual(result["class_order"], ["Negative", "Neutral", "Positive"])
        self.assertEqual(result["confusion_matrix"], [[1, 0, 1], [1, 1, 0], [0, 0, 2]])
        expected = {
            "Negative": (0.5, 0.5, 0.5),
            "Neutral": (1.0, 0.5, 2 / 3),
            "Positive": (2 / 3, 1.0, 0.8),
        }
        for label, values in expected.items():
            for metric, value in zip(("precision", "recall", "f1"), values):
                self.assertAlmostEqual(result["per_class"][label][metric], value)
        json.dumps(result)  # Output can be passed directly to reporting tools.

    def test_absent_classes_are_kept_and_zero_filled(self):
        result = evaluate_predictions(["Negative"], ["Negative"])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        for metric in ("macro_precision", "macro_recall", "macro_f1"):
            self.assertAlmostEqual(result[metric], 1 / 3)
        for label in ("Neutral", "Positive"):
            self.assertEqual(result["per_class"][label], {"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_invalid_evaluation_inputs(self):
        for true, predicted in (
            ([], []), (["Negative"], []),
            (["Negative"], ["Negative", "Positive"]),
            (["unknown"], ["Positive"]), (["Neutral"], ["unknown"]),
            ([1], ["Positive"]), ("Negative", "Negative"),
        ):
            with self.subTest(true=true, predicted=predicted), self.assertRaises(ValueError):
                evaluate_predictions(true, predicted)


if __name__ == "__main__":
    unittest.main()
