"""Tests for TweetClaw export conversion."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tweetclaw_to_cleaned_tweets import convert_rows, read_export, write_csv


class TestTweetClawToCleanedTweets(unittest.TestCase):
    def test_jsonl_export_converts_to_cleaned_schema(self):
        path = self._write_file(
            "\n".join(
                [
                    json.dumps({"text": "@Airline Delayed again!!!"}),
                    json.dumps({"full_text": "Cabin crew was helpful."}),
                ]
            ),
            ".jsonl",
        )

        rows = list(convert_rows(read_export(path), "negative"))

        self.assertEqual(rows[0]["airline_sentiment"], "negative")
        self.assertEqual(rows[0]["text"], "@Airline Delayed again!!!")
        self.assertEqual(rows[0]["clean_text"], "delayed")
        self.assertEqual(rows[1]["clean_text"], "cabin crew helpful")

    def test_csv_export_writes_header(self):
        source_path = self._write_file("text\nLoved the flight\n", ".csv")
        output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name)

        write_csv(convert_rows(read_export(source_path), "positive"), output_path)

        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["airline_sentiment"], "positive")
        self.assertEqual(rows[0]["clean_text"], "loved flight")

    def test_rows_without_text_are_skipped(self):
        rows = list(convert_rows([{"created_at": "2026-06-20"}], "neutral"))

        self.assertEqual(rows, [])

    def _write_file(self, content, suffix):
        handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix, encoding="utf-8")
        with handle:
            handle.write(content)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
