"""Synthetic checks for data/split_reviews.py; no dataset required."""

import unittest

import pandas as pd

from data.split_reviews import split

COLUMNS = ["Id", "UserId", "Score", "Text", "Sentiment"]
CLASS_COUNTS = {"Positive": 70, "Negative": 20, "Neutral": 10}
SCORES = {"Positive": 5, "Negative": 1, "Neutral": 3}


def make_frame(*extra_rows):
    """Build 100 cleaned reviews with unique text, plus any extra rows."""
    records = []
    for sentiment, count in CLASS_COUNTS.items():
        for _ in range(count):
            i = len(records) + 1
            records.append({
                "Id": i, "UserId": f"U{i}", "Score": SCORES[sentiment],
                "Text": f"review text {i}", "Sentiment": sentiment,
            })
    for overrides in extra_rows:
        i = len(records) + 1
        record = {"Id": i, "UserId": f"U{i}", "Score": 5, "Text": f"review text {i}",
                  "Sentiment": "Positive"}
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records, columns=COLUMNS)


class SizeTests(unittest.TestCase):
    def test_default_is_80_20(self):
        train, test = split(make_frame())
        self.assertEqual((len(train), len(test)), (80, 20))

    def test_test_size_is_respected(self):
        train, test = split(make_frame(), test_size=0.3)
        self.assertEqual((len(train), len(test)), (70, 30))

    def test_every_row_lands_on_exactly_one_side(self):
        frame = make_frame()
        train, test = split(frame)
        self.assertEqual(sorted(train["Id"].tolist() + test["Id"].tolist()), frame["Id"].tolist())

    def test_stratified_by_sentiment(self):
        train, test = split(make_frame())
        self.assertEqual(train["Sentiment"].value_counts().to_dict(),
                         {"Positive": 56, "Negative": 16, "Neutral": 8})
        self.assertEqual(test["Sentiment"].value_counts().to_dict(),
                         {"Positive": 14, "Negative": 4, "Neutral": 2})


class RepeatedTextTests(unittest.TestCase):
    def assert_no_shared_text(self, train, test):
        self.assertEqual(set(train["Text"]) & set(test["Text"]), set())

    def test_identical_text_stays_on_one_side(self):
        # Every text appears three times, so a row-level split would separate many of them.
        frame = pd.concat([make_frame()] * 3, ignore_index=True)
        frame["Id"] = range(1, len(frame) + 1)
        for seed in range(5):
            with self.subTest(seed=seed):
                train, test = split(frame, seed=seed)
                self.assert_no_shared_text(train, test)
                self.assertEqual((len(train), len(test)), (240, 60))

    def test_repeats_with_different_labels_stay_together(self):
        repeats = [{"Text": f"review text {i}", "Score": 3, "Sentiment": "Neutral"}
                   for i in range(1, 31)]  # texts 1-30 are Positive in the base frame
        frame = make_frame(*repeats)
        train, test = split(frame)
        self.assert_no_shared_text(train, test)
        self.assertEqual(len(train) + len(test), len(frame))


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_gives_same_split(self):
        first, second = split(make_frame()), split(make_frame())
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_different_seed_gives_different_split(self):
        _, test_a = split(make_frame(), seed=1)
        _, test_b = split(make_frame(), seed=2)
        self.assertNotEqual(test_a["Id"].tolist(), test_b["Id"].tolist())


class OutputTests(unittest.TestCase):
    def test_columns_unchanged(self):
        for part in split(make_frame()):
            self.assertEqual(part.columns.tolist(), COLUMNS)

    def test_row_order_kept_and_index_reset(self):
        for part in split(make_frame()):
            self.assertTrue(part["Id"].is_monotonic_increasing)
            self.assertEqual(part.index.tolist(), list(range(len(part))))

    def test_input_not_mutated(self):
        frame = make_frame()
        original = frame.copy(deep=True)
        split(frame)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
