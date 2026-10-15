# -*- coding: utf-8 -*-
"""The ShareGPT output format.

One session is one JSON object:

    {"conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "..."},
        {"from": "gpt",    "value": "..."}
    ]}

Axolotl, Unsloth and most other trainers read this directly. Train with
`train_on_inputs: false` so the loss is computed on the `gpt` turns only.

Three rules are enforced everywhere a session is built, because a trainer will
happily accept a session that breaks them and produce a worse adapter:

  1. A session must contain at least one `gpt` turn, or it teaches nothing.
  2. A session must not end on a `human` turn — that is an unanswered prompt.
  3. A session should start with `human`, so the model learns to respond rather
     than to open. Sessions that start on a character turn get a short scene
     opener prepended.
"""


# Role names seen in the wild, mapped onto the three the pipeline uses.
ROLE_MAP = {
    "human": "human", "user": "human", "USER": "human", "User": "human",
    "gpt": "gpt", "assistant": "gpt", "character": "gpt", "CHARACTER": "gpt",
    "char": "gpt", "bot": "gpt", "model": "gpt", "Assistant": "gpt",
    "system": "system", "SYSTEM": "system", "System": "system",
}


def merge_turns(turns):
    """Join consecutive turns of the same role.

    Two `gpt` turns in a row become one message on training, so merging them
    here is what the trainer would do anyway — doing it explicitly keeps the
    turn counts in the reports honest.
    """
    merged = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["text"] += "\n\n" + turn["text"]
        else:
            merged.append(dict(turn))
    return merged


def to_sharegpt(turns, system_value, opener=None):
    """Build one ShareGPT session from internal turns.

    `turns` is a list of {"role": "char"|"user", "text": str}.
    `opener` is used only when the transcript starts on a character turn.
    Returns None when the result would violate one of the three rules above.
    """
    merged = merge_turns(turns)
    while merged and merged[-1]["role"] == "user":
        merged.pop()
    if not merged:
        return None

    conversation = [{"from": "system", "value": system_value}]
    if merged[0]["role"] == "char":
        if not opener:
            return None
        conversation.append({"from": "human", "value": opener})
    for turn in merged:
        role = "human" if turn["role"] == "user" else "gpt"
        conversation.append({"from": role, "value": turn["text"]})

    if not any(message["from"] == "gpt" for message in conversation):
        return None
    return {"conversations": conversation}


def normalize_entry(entry):
    """Bring a third-party record into the pipeline's ShareGPT shape.

    Accepts both the `conversations`/`from`/`value` and the
    `messages`/`role`/`content` conventions. Returns None if the record cannot
    be used — most often because it has no assistant turn at all.
    """
    conversation = entry.get("conversations") or entry.get("messages") or []
    out = []
    for message in conversation:
        role = ROLE_MAP.get(message.get("from") or message.get("role") or "")
        value = (message.get("value") or message.get("content") or "").strip()
        if not role or not value:
            continue
        out.append({"from": role, "value": value})
    if not any(message["from"] == "gpt" for message in out):
        return None
    return out


def trim_session(conversation, max_chars, opener):
    """Cut a session down to a character budget, on turn borders.

    Trimming mid-turn would produce a truncated answer as a training target,
    which teaches the model to stop mid-sentence. Whole turns are dropped from
    the end instead, and the result is repaired so it still starts on `human`
    and ends on `gpt`.
    """
    system = [m for m in conversation if m["from"] == "system"][:1]
    turns = [m for m in conversation if m["from"] != "system"]
    budget = max_chars - sum(len(m["value"]) for m in system)

    kept, used = [], 0
    for turn in turns:
        if used + len(turn["value"]) > budget and kept:
            break
        kept.append(turn)
        used += len(turn["value"])

    while kept and kept[-1]["from"] == "human":
        kept.pop()
    if not kept or not any(m["from"] == "gpt" for m in kept):
        return None
    if kept[0]["from"] == "gpt":
        kept.insert(0, {"from": "human", "value": opener})
    return system + kept


def session_stats(rows):
    """Aggregate numbers for the reports at the end of each step."""
    n_messages = 0
    gpt_lengths = []
    system_blocks = set()
    for row in rows:
        for message in row.get("conversations", []):
            n_messages += 1
            if message["from"] == "system":
                system_blocks.add(message["value"])
            elif message["from"] == "gpt":
                gpt_lengths.append(len(message["value"].split()))
    gpt_lengths.sort()
    return {
        "sessions": len(rows),
        "messages": n_messages,
        "unique_system_blocks": len(system_blocks),
        "gpt_words": gpt_lengths,
    }
