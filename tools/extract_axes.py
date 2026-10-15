# -*- coding: utf-8 -*-
"""Harvest behavioural directives from SillyTavern preset files. Optional.

Format axes — quote style, person, tense — are declared in the language pack and
need nothing from you. This tool is about the other kind of instruction: "do not
act for the user", "no purple prose", "keep NPCs coherent". Those are what
real presets are full of, and mixing them into the system blocks of step 07
teaches the model to follow instructions it has not seen verbatim.

    python tools/extract_axes.py preset1.json preset2.json
    python tools/extract_axes.py                 # every .json in the folder
    python tools/extract_axes.py --clean         # tidy an existing axes.yaml

The output is a draft. Read it: the "unsorted" group always contains skeleton
tags with no instructional content, and those should be deleted. --clean does
the obvious part of that automatically.

Only the `autonomy` and `prose` groups are used by step 07. The rest are kept in
the file for reference, and anything listed in augment.holdout_axes is
deliberately never trained on so that generalisation can be measured.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.axes import clean_macros  # noqa: E402

# Order matters: specific patterns come before general ones. Without that, a
# preset section about visual formatting matches "person" because the word
# "perspective" appears somewhere in its body.
AXIS_HINTS = [
    ("html", r"(html|css|<font|<details|gfx|visual toolkit|code block|yaml|<!--)"),
    ("reasoning", r"(chain of thought|\bcot\b|reasoning|рассужд)"),
    ("autonomy", r"(autonom|mary sue|plot armor|preferential|free will|hovering|"
                 r"anti-parrot|never act for|no user control|автоном|поблажк|не действуй за)"),
    ("length", r"(\blength\b|\d+\s*(?:to|-|–)\s*\d+\s*words|under \d+ words|длин|абзац)"),
    ("person", r"(\bpov\b|point of view|(?:1st|2nd|3rd|first|second|third)[\s-]person|"
               r"narrative style|лиц[оа]\b|перспектив)"),
    ("tense", r"(\btense\b|прошедш|настоящ время)"),
    ("thoughts", r"(internal thought|\bthoughts?\b|мысл|размышл)"),
    ("nsfw", r"(nsfw|adult mode|explicit|vulgar|gore)"),
    ("prose", r"(prose|slop|clich|repetition|writing guideline|стил|клише)"),
]

UNSORTED = "unsorted"

# For --clean: what to keep in the unsorted group, and what to delete outright.
KEEP_RE = re.compile(
    r"(npc|autonom|parrot|user control|mary sue|plot armor|preferential|free will|"
    r"hovering|agency|coheren|cognitive|автоном|поблажк|не действуй за)", re.I)
DROP_RE = re.compile(
    r"(^//|persona start|persona end|character start|character end|world definition|"
    r"^<|tier \d|html|<font|<details|css|gfx)", re.I)


def guess_axis(name, content):
    blob = f"{name} {content[:600]}".lower()
    for axis, pattern in AXIS_HINTS:
        if re.search(pattern, blob, re.I):
            return axis
    return UNSORTED


def extract(paths, out_path, include_disabled=True):
    """Read presets and write a draft axes.yaml.

    Disabled entries are included by default and that is deliberate: a preset
    normally enables one value of an axis — third person, say — and keeps the
    alternatives sitting next to it, switched off. Those alternatives are
    exactly what makes the axis varied.
    """
    draft = defaultdict(list)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        preset_name = os.path.splitext(os.path.basename(path))[0]
        prompts = {p.get("identifier"): p for p in preset.get("prompts", [])}
        orders = preset.get("prompt_order", [])
        if not orders:
            print(f"  {path}: no prompt_order, skipped")
            continue
        order = max(orders, key=lambda x: len(x.get("order", [])))
        for entry in order.get("order", []):
            enabled = bool(entry.get("enabled"))
            if not enabled and not include_disabled:
                continue
            prompt = prompts.get(entry.get("identifier"))
            if not prompt:
                continue
            content = clean_macros(prompt.get("content") or "")
            if len(content) < 25:  # skeleton tags such as "<lore>"
                continue
            draft[guess_axis(prompt.get("name", ""), content)].append({
                "preset": preset_name,
                "name": (prompt.get("name") or "").strip(),
                "enabled_in_preset": enabled,
                "text": content,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(draft), f, allow_unicode=True, sort_keys=False, width=100)

    print(f"\nDraft -> {out_path}")
    for axis, items in sorted(draft.items(), key=lambda kv: -len(kv[1])):
        print(f"  {axis:<12} {len(items):>3}")
    print(f"\nNext: run --clean to sort the '{UNSORTED}' group, then read the file. "
          f"Only 'autonomy' and 'prose' are used by step 07.")


def clean(path):
    """Sort the unsorted group: keep behavioural entries, drop skeleton tags."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unsorted_items = data.pop(UNSORTED, []) or []
    moved = 0
    for item in unsorted_items:
        name, text = item.get("name", ""), item.get("text", "")
        if DROP_RE.search(name) or DROP_RE.search(text[:200]):
            continue
        if KEEP_RE.search(name + " " + text[:400]) and 60 < len(text) < 2500:
            data.setdefault("autonomy", []).append(item)
            moved += 1

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=100)

    print(f"Moved into autonomy: {moved}")
    print(f"Dropped from '{UNSORTED}': {len(unsorted_items) - moved}")
    print("\nFinal pool used by step 07:")
    for key in ("autonomy", "prose"):
        for item in data.get(key, []):
            print(f"  [{key}] {len(item['text']):>5} chars  "
                  f"{item['preset'][:14]:<14} {item['name'][:45]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("presets", nargs="*", help="preset .json files (default: ./*.json)")
    parser.add_argument("--out", default="axes.yaml", help="output file (default: axes.yaml)")
    parser.add_argument("--only-enabled", action="store_true",
                        help="skip entries disabled in the preset")
    parser.add_argument("--clean", action="store_true",
                        help="tidy an existing axes.yaml instead of extracting")
    args = parser.parse_args()

    if args.clean:
        clean(args.out)
        return
    paths = args.presets or [p for p in os.listdir(".") if p.endswith(".json")]
    if not paths:
        sys.exit("No preset .json files given or found in the current folder.")
    extract(paths, args.out, not args.only_enabled)


if __name__ == "__main__":
    main()
