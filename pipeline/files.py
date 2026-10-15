# -*- coding: utf-8 -*-
"""Reading and writing files.

Two jobs:

  * decoding the source text, which for old books is often not UTF-8;
  * JSONL input and output, which is the pipeline's only intermediate format.

JSONL is used everywhere on purpose. Every step can be stopped, inspected in a
text editor, and restarted, and every step appends its results line by line, so
an interrupted run loses at most one item.
"""
import json
import os
import unicodedata
from pathlib import Path


def _mojibake_score(text):
    """How wrong a decoding looks, from 0.0 (fine) to 1.0 (garbage).

    An encoding that is wrong for the file usually still "succeeds" — cp1251
    will happily decode any byte sequence. What gives it away is the result:
    control characters and private-use codepoints that no real text contains.
    """
    sample = text[:8000]
    if not sample:
        return 1.0
    bad = 0
    for ch in sample:
        category = unicodedata.category(ch)
        # Cc = control, Co = private use, Cn = unassigned. Tabs and newlines are
        # control characters too, hence the explicit allowance.
        if category in ("Cc", "Co", "Cn") and ch not in "\t\n\r":
            bad += 1
        elif ch == "\ufffd":  # replacement character, a decode already went wrong
            bad += 1
    return bad / len(sample)


def read_text(path, encodings=None, verbose=True):
    """Read a text file, trying the configured encodings in order.

    Returns the decoded text. Raises if nothing decodes acceptably, which in
    practice means the file is not text at all.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"source text not found: {path}\n"
            f"Put a UTF-8 .txt file there, or fix input.path in the config.")
    encodings = encodings or ["utf-8-sig", "utf-8", "latin-1"]
    raw = open(path, "rb").read()
    best = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        score = _mojibake_score(text)
        if score < 0.001:
            if verbose:
                print(f"[encoding: {enc}]")
            return text
        # Keep the least bad candidate in case none is clean.
        if best is None or score < best[0]:
            best = (score, enc, text)
    if best is None:
        raise RuntimeError(f"could not decode {path} with any of: {', '.join(encodings)}")
    score, enc, text = best
    if verbose:
        print(f"[encoding: {enc}, but {score:.1%} of characters look wrong — "
              f"check the file before trusting the output]")
    return text


def read_jsonl(path):
    """Read a JSONL file into a list. A missing file is an empty list."""
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: broken JSON line ({exc})") from exc
    return items


def write_jsonl(path, items):
    """Write a list to JSONL, replacing whatever was there."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path, item):
    """Append one record. This is what makes every step resumable."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_dataset(path):
    """Read a ShareGPT dataset from .jsonl or .parquet.

    Parquet is what most Hugging Face datasets are distributed as; it needs
    pandas, which is an optional dependency.
    """
    if path.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                f"{path} is parquet, which needs pandas: pip install pandas pyarrow\n"
                f"Or download the .jsonl version of the dataset instead.")
        frame = pd.read_parquet(path)
        column = "conversations" if "conversations" in frame.columns else frame.columns[0]
        return [{"conversations": list(value)} for value in frame[column]]
    return read_jsonl(path)


def require_file(path, produced_by):
    """Stop with a useful message when a previous step has not been run.

    `produced_by` names the script that creates the file, because "file not
    found" on its own does not tell you which step you skipped.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"\nMissing input: {path}\n"
            f"That file is produced by {produced_by}. Run it first, then retry this step.\n")
    return path
