# -*- coding: utf-8 -*-
"""Step 03 — turn each scene into a roleplay transcript.

The conversion rules, and why they are what they are:

  * char is the most talkative character of the scene, or the one named in
    convert.char. Its lines plus the surrounding narration become one turn.
  * The narration is carried over with minimal edits rather than retold. If the
    converter is allowed to paraphrase, the dataset ends up in the converter's
    voice instead of the source's, which defeats the whole exercise.
  * user turns are the interlocutor's lines only, with no narration. That is
    deliberate asymmetry: it teaches the model to answer richly to thin input,
    which is what a real chat session looks like.
  * In a crowd scene, everyone else becomes part of the world; their lines go
    inside char's narration.

    Input:  <workdir>/scenes_attr.jsonl
    Output: <workdir>/rp_raw.jsonl   {id, source_id, char, user, turns}

    python 03_convert.py [--config config.yaml]
"""
import sys
from collections import Counter

from pipeline.config import load_config, parse_args, work_path
from pipeline.files import append_jsonl, read_jsonl, require_file
from pipeline.lang import load_pack, render
from pipeline.llm import LLM, extract_json, make_refusal_checker, parallel_map


def pick_pair(scene, forced_char, unknown, generic):
    """Choose (char, user) for a scene, or (None, None) if it is unusable.

    A speaker is only a candidate if it is a real name: not the unknown marker,
    not a pronoun or a role word, and longer than one character. Those come back
    from attribution regularly, and a transcript whose char is "the voice"
    teaches nothing.
    """
    speakers = []
    for line in scene.get("lines", []):
        name = str(line.get("speaker", "")).strip()
        if not name or len(name) <= 1:
            continue
        if name.lower() == unknown or name.lower() in generic:
            continue
        speakers.append(name)
    if not speakers:
        return None, None

    counts = Counter(speakers)
    if forced_char and forced_char.lower() != "auto":
        if forced_char not in counts:
            return None, None
        char = forced_char
    else:
        char = counts.most_common(1)[0][0]

    others = [name for name, _ in counts.most_common() if name != char]
    if not others:
        return None, None
    return char, others[0]  # user = the second most active speaker


def main():
    args = parse_args(__doc__.splitlines()[0])
    cfg = load_config(args.config)
    pack = load_pack(cfg["lang"])
    convert = cfg["convert"]
    llm = LLM(cfg)
    workers = cfg["llm"]["workers"]
    is_refusal = make_refusal_checker(pack.refusal_markers)
    unknown = pack.unknown_speaker.lower()
    generic = pack.generic_speakers

    scenes = read_jsonl(require_file(work_path(cfg, "scenes_attr.jsonl"), "02_attribute.py"))
    out_path = work_path(cfg, "rp_raw.jsonl")
    refusal_path = work_path(cfg, "refusals_03.jsonl")

    existing = read_jsonl(out_path)
    done = {item["source_id"] for item in existing}
    next_id = len(existing)
    if done:
        print(f"[resume] {next_id} transcripts already written, continuing")

    pending, skipped = [], 0
    for scene in scenes:
        if scene["id"] in done:
            continue
        char, user = pick_pair(scene, convert.get("char", "auto"), unknown, generic)
        if not char or scene.get("dialogue_share", 0) < convert["min_dialogue_share"]:
            skipped += 1
            continue
        scene["_char"], scene["_user"] = char, user
        pending.append(scene)

    print(f"To convert: {len(pending)} scenes ({skipped} skipped as unusable)")
    if not pending:
        print("Nothing to do. If everything was skipped, check convert.char and "
              "convert.min_dialogue_share, and re-read the step 02 output.")
        return

    def work(scene):
        attribution = "; ".join(
            f"\"{line.get('quote', '')}...\" — {line.get('speaker', '')}"
            for line in scene.get("lines", []))
        return llm.chat(
            pack.require("convert.system"),
            render(pack.require("convert.user"),
                   scene=scene["text"], char=scene["_char"],
                   user=scene["_user"], attribution=attribution),
            temperature=convert["temperature"],
        )

    n_refusals = 0
    for scene, raw, error in parallel_map(work, pending, workers):
        if error is not None:
            print(f"  [warn] scene {scene['id']}: {error} — will retry on the next run",
                  file=sys.stderr)
            continue
        try:
            data = extract_json(raw)
        except Exception:  # noqa: BLE001
            if is_refusal(raw):
                append_jsonl(refusal_path, {"source_id": scene["id"], "head": raw[:300]})
                n_refusals += 1
                print(f"scene {scene['id']}: model refused -> {refusal_path}, not marked done")
            else:
                print(f"  [warn] scene {scene['id']}: invalid JSON — will retry",
                      file=sys.stderr)
            continue

        turns = [
            turn for turn in data.get("turns", [])
            if isinstance(turn, dict)
            and turn.get("role") in ("char", "user")
            and str(turn.get("text", "")).strip()
        ]
        if len(turns) < convert["min_turns"] or not any(t["role"] == "char" for t in turns):
            skipped += 1
            continue

        append_jsonl(out_path, {"id": next_id, "source_id": scene["id"],
                                "char": scene["_char"], "user": scene["_user"],
                                "turns": turns})
        next_id += 1
        print(f"scene {scene['id']} -> {len(turns)} turns "
              f"({scene['_char']} / {scene['_user']})")

    print(f"\nDone: {next_id} transcripts, {skipped} skipped -> {out_path}")
    if n_refusals:
        print(f"{n_refusals} refusals. Rerun with a different llm.base_url to pick up only those.")
    print("Checkpoint: read 5-10 transcripts side by side with their source scenes. "
          "If the narration has been retold rather than carried over, fix the "
          "convert prompt in the language pack before running step 04.")


if __name__ == "__main__":
    main()
