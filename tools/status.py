# -*- coding: utf-8 -*-
"""Show where a run stands: the funnel, the rejections, the characters.

Read this between steps. It answers the two questions that matter — how much
data survived each stage, and is there enough of it to continue.

    python tools/status.py [--config config.yaml]
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config, work_path  # noqa: E402
from pipeline.files import read_jsonl  # noqa: E402
from pipeline.lang import load_pack  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    unknown = pack.unknown_speaker.lower()
    generic = pack.generic_speakers

    def load(name):
        return read_jsonl(work_path(cfg, name))

    scenes = load("scenes.jsonl")
    attributed = load("scenes_attr.jsonl")
    raw = load("rp_raw.jsonl")
    clean = load("rp_clean.jsonl")
    rejected = load("rp_rejected.jsonl")
    synth = load("synth_clean.jsonl")
    synth_rejected = load("synth_rejected.jsonl")
    refusals = sum(len(load(f"refusals_{n}.jsonl")) for n in ("02", "03", "05", "08"))

    print(f"workdir: {cfg['workdir']}   language: {cfg['lang']}")
    print(f"01 scenes:              {len(scenes)}")
    print(f"02 attributed:          {len(attributed)}/{len(scenes)} "
          f"({100 * len(attributed) // max(1, len(scenes))}%)")

    dialogue = [s for s in attributed
                if s.get("dialogue_share", 0) >= cfg["convert"]["min_dialogue_share"]
                and s.get("lines")]
    print(f"   with dialogue:       {len(dialogue)}")

    characters = Counter()
    for scene in attributed:
        for line in scene.get("lines", []):
            name = str(line.get("speaker", "")).strip()
            if len(name) > 1 and name.lower() != unknown and name.lower() not in generic:
                characters[name] += 1
    top = ", ".join(f"{name} ({count})" for name, count in characters.most_common(6))
    print(f"   top characters:      {top or '—'}")

    print(f"03 transcripts:         {len(raw)}")
    print(f"04 accepted/rejected:   {len(clean)}/{len(rejected)}")
    for reason, count in Counter(str(r.get("reject_reason", ""))[:45]
                                 for r in rejected).most_common(5):
        print(f"     {count:>5}  {reason}")
    print(f"05 synthetic:           {len(synth)} accepted / {len(synth_rejected)} rejected")
    for reason, count in Counter(str(r.get("reject_reason", ""))[:45]
                                 for r in synth_rejected).most_common(3):
        print(f"     {count:>5}  {reason}")
    print(f"model refusals:         {refusals}")

    for label, key, section in (("06 base dataset", "output", "format"),
                                ("07 augmented", "output", "augment"),
                                ("08 final", "output", "assemble")):
        path = cfg[section][key]
        mark = f"{len(read_jsonl(path))} sessions" if os.path.exists(path) else "not built"
        print(f"{label:<23} {mark}   ({path})")

    print("\n--- readiness ---")
    strong = [count for count in characters.values() if count >= 5]
    if not scenes:
        print("NOT READY: no scenes. Start with 01_segment.py.")
    elif len(attributed) < len(scenes):
        print(f"NOT READY for step 05: attribution covers {len(attributed)} of "
              f"{len(scenes)} scenes. Finish step 02 first, or the anchors and the "
              f"character rotation will be thin.")
    elif len(strong) < 2:
        print("NOT READY for step 05: fewer than two characters with 5+ lines. "
              "List the names by hand in synth.chars.")
    else:
        print(f"READY: {len(dialogue)} anchor scenes, {len(strong)} characters to rotate.")

    if clean and len(clean) < 100:
        print(f"NOTE: only {len(clean)} accepted transcripts. That is a small dataset — "
              f"lean harder on step 05 (raise synth.count) and step 07.")


if __name__ == "__main__":
    main()
