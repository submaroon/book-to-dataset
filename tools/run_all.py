# -*- coding: utf-8 -*-
"""Run several steps in sequence, with quality gates between them.

This is what you start before going to bed. Each step is retried on failure —
resume makes that cheap — and between steps the data is checked, so a run does
not spend six hours converting scenes whose attribution is already broken.

    python tools/run_all.py                    # steps 01 through 08
    python tools/run_all.py --from 02 --to 04  # a range
    python tools/run_all.py --force            # ignore the gates

The gates:

  after 02   Dialogue markers in the scene text are counted and compared with
             the number of attributed lines. If attribution is losing lines in
             bulk, step 03 is not started.
  after 03   Transcript length against source scene length. A transcript far
             shorter than its scene means the converter truncated it.

A gate stops the run rather than warning, because the failure it detects makes
everything downstream worthless, and it is much cheaper to notice at 1 a.m.
than at 8 a.m.
"""
import argparse
import importlib.util
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config, work_path  # noqa: E402
from pipeline.files import read_jsonl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("01", "01_segment.py", "segmentation"),
    ("02", "02_attribute.py", "attribution"),
    ("03", "03_convert.py", "conversion"),
    ("04", "04_validate.py", "validation"),
    ("05", "05_synthesize.py", "synthesis"),
    ("06", "06_format.py", "formatting"),
    ("07", "07_augment.py", "augmentation"),
    ("08", "08_assemble.py", "assembly"),
]

# A line opening with a dialogue marker, in the common typographies.
DIALOGUE_MARKER = re.compile(r"^\s*[—–\-«\"„\u201c]\s?\S", re.M)


def load_step(path):
    spec = importlib.util.spec_from_file_location("step_module", os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_step(label, path, config_path, attempts=3):
    """Run one step, retrying on failure. Resume makes a retry cheap."""
    module = load_step(path)
    argv_backup = sys.argv
    for attempt in range(1, attempts + 1):
        try:
            print(f"\n===== {label}: starting (attempt {attempt}/{attempts}) =====", flush=True)
            sys.argv = [path, "--config", config_path]
            module.main()
            return True
        except KeyboardInterrupt:
            raise
        except SystemExit as exc:
            # A step that stops on purpose — a missing input, an empty config
            # value — should not be retried; the message already says what to fix.
            if exc.code not in (0, None):
                print(f"[{label}] stopped deliberately: {exc}", file=sys.stderr)
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] failed: {exc}; waiting 30s and retrying (resume will "
                  f"pick up where it stopped)", file=sys.stderr)
            time.sleep(30)
        finally:
            sys.argv = argv_backup
    print(f"[{label}] did not finish in {attempts} attempts — stopping", file=sys.stderr)
    return False


def gate_attribution(cfg):
    """Compare dialogue markers in the text with attributed lines."""
    scenes = read_jsonl(work_path(cfg, "scenes_attr.jsonl"))
    suspicious = dialogue_scenes = 0
    for scene in scenes:
        markers = len(DIALOGUE_MARKER.findall(scene["text"]))
        if markers < 4:
            continue
        dialogue_scenes += 1
        if len(scene.get("lines", [])) < markers * 0.5:
            suspicious += 1
            print(f"  [suspect] scene {scene['id']}: {markers} markers, "
                  f"{len(scene.get('lines', []))} lines, {scene['chars']} chars")
    return suspicious, dialogue_scenes


def gate_truncation(cfg):
    """Compare transcript length with source scene length."""
    scenes = {s["id"]: s for s in read_jsonl(work_path(cfg, "scenes.jsonl"))}
    suspicious = total = 0
    for transcript in read_jsonl(work_path(cfg, "rp_raw.jsonl")):
        total += 1
        scene = scenes.get(transcript["source_id"])
        if not scene:
            continue
        ratio = sum(len(t["text"]) for t in transcript["turns"]) / max(1, scene["chars"])
        if ratio < 0.5 and scene["chars"] > 4000:
            suspicious += 1
            print(f"  [short] rp {transcript['id']} (scene {transcript['source_id']}): "
                  f"ratio {ratio:.2f}, scene {scene['chars']} chars")
    return suspicious, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--from", dest="start", default="01", help="first step, e.g. 03")
    parser.add_argument("--to", dest="end", default="08", help="last step, e.g. 06")
    parser.add_argument("--force", action="store_true", help="run past a failed gate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    numbers = [number for number, _path, _label in STEPS]
    assert args.start in numbers, f"--from must be one of: {', '.join(numbers)}"
    assert args.end in numbers, f"--to must be one of: {', '.join(numbers)}"

    selected = STEPS[numbers.index(args.start):numbers.index(args.end) + 1]
    print(f"Running: {', '.join(n for n, _p, _l in selected)}   config: {args.config}")

    for number, path, label in selected:
        if not run_step(f"step {number} ({label})", path, args.config):
            sys.exit(1)

        if number == "02":
            print("\n===== gate: attribution =====")
            suspicious, dialogue_scenes = gate_attribution(cfg)
            print(f"suspicious: {suspicious} of {dialogue_scenes} dialogue scenes")
            if suspicious > max(5, dialogue_scenes // 10) and not args.force:
                sys.exit(
                    "\nSTOP: attribution is losing lines in bulk, and step 03 would turn "
                    "that into garbage.\nInspect scenes_attr.jsonl in the morning. "
                    "To run anyway: python tools/run_all.py --force\n")

        if number == "03":
            print("\n===== gate: truncation =====")
            suspicious, total = gate_truncation(cfg)
            print(f"short transcripts: {suspicious} of {total}")
            if total and suspicious > total // 3 and not args.force:
                sys.exit(
                    "\nSTOP: a third of the transcripts are far shorter than their source "
                    "scenes.\nUsually llm.max_tokens is too low for the scene size. "
                    "To run anyway: --force\n")

    print("\n===== FINISHED =====")
    print("Morning routine: read the reports above, then read 5-10 transcripts from "
          "rp_raw.jsonl against their scenes, then look at the rejection reasons in "
          "rp_rejected.jsonl.")


if __name__ == "__main__":
    main()
