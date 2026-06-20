"""Convert reviewed TweetClaw exports into cleaned_tweets.csv rows."""

import argparse
import csv
import json
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords

TEXT_FIELDS = ("text", "tweet_text", "tweet", "full_text", "content", "body")
FIELDNAMES = ("airline_sentiment", "text", "clean_text")

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def read_export(path):
    content = Path(path).read_text(encoding="utf-8-sig").strip()
    if not content:
        return []

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(content)
    if suffix in (".jsonl", ".ndjson"):
        return _read_json_lines(content)
    if content[0] in "[{":
        return _rows_from_payload(json.loads(content))
    return _read_json_lines(content)


def convert_rows(rows, sentiment):
    for row in rows:
        text = _first_text(row)
        if not text:
            continue
        yield {
            "airline_sentiment": sentiment,
            "text": text,
            "clean_text": clean_text(text),
        }


def write_csv(rows, output_path):
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def _read_csv_rows(content):
    first_line = content.splitlines()[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    return [dict(row) for row in csv.DictReader(content.splitlines(), delimiter=delimiter)]


def _read_json_lines(content):
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _rows_from_payload(payload):
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("results", "tweets", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _first_text(row):
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Convert reviewed TweetClaw exports into cleaned tweet rows."
    )
    parser.add_argument("input", help="TweetClaw JSON, JSONL, NDJSON, or CSV export.")
    parser.add_argument("output", help="Output CSV path.")
    parser.add_argument(
        "--sentiment",
        required=True,
        choices=("positive", "neutral", "negative"),
        help="Reviewed label to apply to every exported row.",
    )
    args = parser.parse_args()

    write_csv(convert_rows(read_export(args.input), args.sentiment), args.output)


if __name__ == "__main__":
    main()
