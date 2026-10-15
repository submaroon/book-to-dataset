# -*- coding: utf-8 -*-
"""Step 01 — cut the source text into scenes.

Two passes, in this order:

  1. A structural pass, which is free: chapter headings, explicit scene breaks,
     blank-line blocks. Most well-formatted books are almost fully segmented by
     this alone.
  2. An LLM pass, which is not free, and therefore runs only inside blocks that
     have no structural markers and are longer than segment.max_block_chars.
     The model receives numbered paragraphs and returns the numbers of the
     paragraphs before which a new scene starts.

The result is post-processed: scenes below the minimum length are glued to
their neighbour, scenes above the maximum are split on paragraph borders.

    Output: <workdir>/scenes.jsonl   {id, chapter, text, chars}

    python 01_segment.py [--config config.yaml]
"""
import os
import re
import sys

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import read_text, write_jsonl
from pipeline.lang import chapter_regex, load_pack, render
from pipeline.llm import LLM, extract_json

SCENE_BREAK_RE = re.compile(r"^\s*([*#~—\-–_=]\s*){3,}\s*$")

# Keeps the terminator with the sentence it ends. The optional whitespace matters:
# CJK punctuation is not followed by a space.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s*")


def normalize(text):
    """Unify line endings and collapse runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def structural_blocks(text, chapter_re):
    """Split on chapter headings and explicit scene breaks.

    Returns [(chapter_title, block_text)]. The chapter title is carried along so
    the scenes can be traced back to their place in the source.
    """
    blocks, current, chapter = [], [], ""
    for line in text.split("\n"):
        if chapter_re.match(line) and len(line.strip()) < 60:
            if current and "".join(current).strip():
                blocks.append((chapter, "\n".join(current).strip()))
            current, chapter = [], line.strip()
        elif SCENE_BREAK_RE.match(line):
            if current and "".join(current).strip():
                blocks.append((chapter, "\n".join(current).strip()))
            current = []
        else:
            current.append(line)
    if current and "".join(current).strip():
        blocks.append((chapter, "\n".join(current).strip()))
    return blocks


def paragraphs_of(block):
    return [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]


def llm_boundaries(llm, pack, paragraphs, window_chars, overlap, debug_dir):
    """Find scene boundaries in one long block, with a sliding window.

    The window budget is measured in characters, not in paragraphs: a single
    paragraph can be several thousand characters long, and a fixed paragraph
    count would either overflow the context or waste it. The two-paragraph
    minimum keeps a single enormous paragraph from stalling the loop.
    """
    boundaries = set()
    start = 0
    while start < len(paragraphs):
        end, budget = start, 0
        while end < len(paragraphs) and (budget + len(paragraphs[end]) <= window_chars
                                         or end - start < 2):
            budget += len(paragraphs[end]) + 8
            end += 1

        numbered = "\n\n".join(f"[{i}] {paragraphs[i]}" for i in range(start, end))
        data, raw = None, ""
        for attempt in range(2):
            suffix = pack.require("segment.strict_suffix") if attempt else ""
            # The first attempt is cheap. The retry gets room, because a
            # reasoning model needs it and a truncated reasoning block is
            # exactly what made the first attempt unparseable.
            raw = llm.chat(
                pack.require("segment.system"),
                render(pack.require("segment.user"), paragraphs=numbered) + suffix,
                temperature=0.0,
                max_tokens=400 if attempt == 0 else 3000,
            )
            try:
                data = extract_json(raw)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt:
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, f"segfail_{start}_{end}.txt")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(raw)
                    print(f"  [warn] window {start}-{end}: unparseable after retry ({exc}); "
                          f"raw reply saved to {debug_file}", file=sys.stderr)

        if data is not None:
            for value in data.get("boundaries", []):
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                # Boundaries at a window edge are ignored: the neighbouring
                # window sees the same spot with proper context on both sides.
                if start < index < end:
                    boundaries.add(index)

        if end == len(paragraphs):
            break
        start = max(start + 1, end - overlap)
    return boundaries


def split_by_boundaries(paragraphs, boundaries):
    scenes, current = [], []
    for index, paragraph in enumerate(paragraphs):
        if index in boundaries and current:
            scenes.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        scenes.append("\n\n".join(current))
    return scenes


def merge_short(scenes, min_chars):
    """Glue undersized scenes to their neighbour."""
    out = []
    for scene in scenes:
        if out and (len(scene) < min_chars or len(out[-1]) < min_chars):
            out[-1] = out[-1] + "\n\n" + scene
        else:
            out.append(scene)
    return out


def hard_split(text, max_chars):
    """Last-resort split of a single oversized paragraph.

    Paragraph borders are the preferred cut, but some files have none: a text
    dumped without blank lines is one paragraph from the first character to the
    last. Without this fallback such a file produces a single scene the size of
    the whole book, which then overflows the context window in step 02 and fails
    there instead of here, with a much less obvious error.

    Sentence borders are tried first. A "sentence" longer than the cap on its own
    is split on whitespace, and text with no whitespace at all is cut at the cap.
    """
    pieces, current = [], ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
            continue
        if current:
            pieces.append(current.strip())
            current = ""
        while len(sentence) > max_chars:
            # Prefer a space near the cap so a word is not cut in half.
            cut = sentence.rfind(" ", int(max_chars * 0.8), max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].lstrip()
        current = sentence
    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def cap_long(scenes, max_chars):
    """Split oversized scenes on paragraph borders.

    Never inside a paragraph while a paragraph border is available: a scene that
    ends mid-sentence produces a transcript that ends mid-sentence. When a single
    paragraph is itself over the cap, hard_split takes over.
    """
    out = []
    for scene in scenes:
        if len(scene) <= max_chars:
            out.append(scene)
            continue
        current, current_len = [], 0
        for paragraph in paragraphs_of(scene):
            for piece in ([paragraph] if len(paragraph) <= max_chars
                          else hard_split(paragraph, max_chars)):
                if current_len + len(piece) > max_chars and current:
                    out.append("\n\n".join(current))
                    current, current_len = [], 0
                current.append(piece)
                current_len += len(piece) + 2
        if current:
            out.append("\n\n".join(current))
    return out


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    segment = cfg["segment"]
    llm = LLM(cfg)

    text = normalize(read_text(cfg["input"]["path"], cfg["input"]["encodings"]))
    blocks = structural_blocks(text, chapter_regex(pack, full_line=True))
    print(f"Structural pass: {len(blocks)} blocks")

    long_blocks = [b for b in blocks if len(b[1]) > segment["max_block_chars"]]
    if long_blocks:
        print(f"{len(long_blocks)} of them need the LLM pass "
              f"(longer than segment.max_block_chars = {segment['max_block_chars']})")

    debug_dir = work_path(cfg, "debug")
    collected = []
    for index, (chapter, block) in enumerate(blocks):
        if len(block) <= segment["max_block_chars"]:
            collected.append((chapter, block))
            continue
        paragraphs = paragraphs_of(block)
        print(f"Block {index} ({chapter or 'no chapter'}): {len(block)} chars, "
              f"{len(paragraphs)} paragraphs -> LLM")
        boundaries = llm_boundaries(llm, pack, paragraphs, segment["llm_window_chars"],
                                    segment["llm_window_overlap"], debug_dir)
        for scene in split_by_boundaries(paragraphs, boundaries):
            collected.append((chapter, scene))

    texts = [t for _, t in collected]
    texts = merge_short(texts, segment["min_scene_chars"])
    texts = cap_long(texts, segment["max_scene_chars"])
    assert texts, "segmentation produced no scenes — check the source file"

    # Re-attach chapter titles after merging and capping changed the boundaries.
    # A scene keeps the title of the block its opening came from; scenes created
    # by a split inherit the last title seen.
    items, chapter_cursor = [], ""
    for scene_id, scene_text in enumerate(texts):
        head = scene_text[:60]
        for chapter, block_text in collected:
            if block_text.startswith(head):
                chapter_cursor = chapter
                break
        items.append({"id": scene_id, "chapter": chapter_cursor,
                      "text": scene_text, "chars": len(scene_text)})

    output = work_path(cfg, "scenes.jsonl")
    write_jsonl(output, items)

    lengths = sorted(item["chars"] for item in items)
    oversized = [item for item in items if item["chars"] > segment["max_scene_chars"]]
    print(f"\nDone: {len(items)} scenes -> {output}")
    if oversized:
        print(f"[!] {len(oversized)} scenes are still over segment.max_scene_chars "
              f"({segment['max_scene_chars']}), largest {max(i['chars'] for i in oversized)}. "
              f"They will be expensive in step 02 and may overflow the context.")
    print(f"Scene length: min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}")
    print("\nCheckpoint: if the median is far below segment.min_scene_chars, the "
          "chunker is shredding the text — raise the thresholds and rerun.")


if __name__ == "__main__":
    main()
