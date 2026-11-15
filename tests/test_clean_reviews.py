"""Synthetic checks for data/clean_reviews.py; no dataset required."""

import unittest

import pandas as pd

from data.clean_reviews import clean

COLUMNS = [
    "Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
    "HelpfulnessDenominator", "Score", "Time", "Summary", "Text",
]


def make_frame(*rows):
    """Build a review frame; each row overrides defaults that are unique per row."""
    records = []
    for i, overrides in enumerate(rows, start=1):
        record = {
            "Id": i, "ProductId": f"P{i}", "UserId": f"U{i}", "ProfileName": f"Name {i}",
            "HelpfulnessNumerator": 0, "HelpfulnessDenominator": 0, "Score": 5,
            "Time": 1_300_000_000 + i, "Summary": f"summary {i}", "Text": f"review text {i}",
        }
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records, columns=COLUMNS)


SAME_REVIEW = {"UserId": "U1", "ProfileName": "Ann", "Time": 1_300_000_000, "Text": "great tea"}


class DuplicateTests(unittest.TestCase):
    def test_exact_key_duplicate_keeps_first(self):
        result = clean(make_frame(SAME_REVIEW, SAME_REVIEW, {}))
        self.assertEqual(result["Id"].tolist(), [1, 3])
        self.assertEqual(result["ProductId"].tolist(), ["P1", "P3"])

    def test_key_differences_are_not_duplicates(self):
        for field, value in (("UserId", "U2"), ("Time", 1_300_000_999), ("ProfileName", "Bob")):
            with self.subTest(field=field):
                result = clean(make_frame(SAME_REVIEW, {**SAME_REVIEW, field: value}))
                self.assertEqual(len(result), 2)

    def test_duplicates_use_raw_text(self):
        result = clean(make_frame(SAME_REVIEW, {**SAME_REVIEW, "Text": "Great tea"}))
        self.assertEqual(len(result), 2)

    def test_missing_profile_name_still_deduplicated(self):
        row = {**SAME_REVIEW, "ProfileName": None}
        result = clean(make_frame(row, row))
        self.assertEqual(result["Id"].tolist(), [1])
        self.assertTrue(result["ProfileName"].isna().all())

    def test_conflicting_scores_drop_whole_group(self):
        result = clean(make_frame(
            {**SAME_REVIEW, "Score": 5}, {**SAME_REVIEW, "Score": 1},
            {**SAME_REVIEW, "Score": 5}, {},
        ))
        self.assertEqual(result["Id"].tolist(), [4])


class InvalidRowTests(unittest.TestCase):
    def test_missing_or_blank_text_dropped(self):
        result = clean(make_frame({"Text": None}, {"Text": ""}, {"Text": "  \n\t "}, {}))
        self.assertEqual(result["Id"].tolist(), [4])

    def test_text_blank_after_cleaning_dropped(self):
        result = clean(make_frame({"Text": "<br /><br />&nbsp;"}, {}))
        self.assertEqual(result["Id"].tolist(), [2])

    def test_missing_summary_and_profile_name_kept(self):
        result = clean(make_frame({"Summary": None}, {"ProfileName": None}))
        self.assertEqual(result["Id"].tolist(), [1, 2])

    def test_invalid_scores_dropped(self):
        result = clean(make_frame(
            {"Score": 0}, {"Score": 6}, {"Score": None}, {"Score": 2.5}, {"Score": 3},
        ))
        self.assertEqual(result["Id"].tolist(), [5])
        self.assertEqual(result["Score"].tolist(), [3])


class TextTests(unittest.TestCase):
    def cleaned(self, text):
        return clean(make_frame({"Text": text}))["Text"].iloc[0]

    def test_html_tags_removed_with_word_boundaries(self):
        cases = {
            "good<br />taste": "good taste",
            "first<br>second<BR/>third": "first second third",
            'try <a href="http://www.amazon.com/gp/product/B000">this brand</a> now':
                "try this brand now",
            '<span class="tiny"> Length:: 0:26 Mins </span>nice<p>video': "length:: 0:26 mins nice video",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.cleaned(raw), expected)

    def test_entities_unescaped(self):
        cases = {
            "caf&eacute; latte": "café latte",
            "the &quot;best&quot; snack": 'the "best" snack',
            "i &lt;3 this &amp; that": "i <3 this & that",
            "5 &lt; 6 and 7 &gt; 6": "5 < 6 and 7 > 6",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.cleaned(raw), expected)

    def test_non_tag_angle_brackets_kept(self):
        self.assertEqual(self.cleaned("price < 5 and taste > ok"), "price < 5 and taste > ok")

    def test_whitespace_collapsed_and_stripped(self):
        self.assertEqual(self.cleaned("  too   many\n\nspaces\there  "), "too many spaces here")

    def test_lowercased(self):
        self.assertEqual(self.cleaned("LOVED It. Not BAD!"), "loved it. not bad!")

    def test_other_text_columns_untouched(self):
        result = clean(make_frame({"Summary": "Great<br />  STUFF", "ProfileName": "Ann  B"}))
        self.assertEqual(result["Summary"].iloc[0], "Great<br />  STUFF")
        self.assertEqual(result["ProfileName"].iloc[0], "Ann  B")


class OutputTests(unittest.TestCase):
    def test_sentiment_labels(self):
        result = clean(make_frame(*({"Score": score} for score in range(1, 6))))
        self.assertEqual(
            result["Sentiment"].tolist(),
            ["Negative", "Negative", "Neutral", "Positive", "Positive"],
        )

    def test_columns_are_original_plus_sentiment(self):
        self.assertEqual(clean(make_frame({})).columns.tolist(), COLUMNS + ["Sentiment"])

    def test_index_reset(self):
        result = clean(make_frame({"Text": None}, {}, SAME_REVIEW, SAME_REVIEW, {}))
        self.assertEqual(result.index.tolist(), [0, 1, 2])

    def test_input_not_mutated(self):
        frame = make_frame({"Text": "Good<br />TEA"}, SAME_REVIEW, SAME_REVIEW, {"Text": None})
        original = frame.copy(deep=True)
        clean(frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_empty_input(self):
        result = clean(make_frame())
        self.assertEqual(len(result), 0)
        self.assertEqual(result.columns.tolist(), COLUMNS + ["Sentiment"])


if __name__ == "__main__":
    unittest.main()
