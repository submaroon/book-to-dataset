# -*- coding: utf-8 -*-
"""A fake OpenAI-compatible server, for testing the pipeline without a model.

Point `llm.base_url` at this and every step runs end to end in seconds. It
answers each step with well-formed, plausible output derived from the actual
prompt, so the plumbing gets exercised for real: JSON parsing, the resume logic,
the quality gates, the ShareGPT conversion, the reports.

What it does NOT test is quality. The text it returns is mechanical. Use it to
check that a config change did not break the pipeline, never to judge whether a
dataset is good.

    python tools/mock_llm_server.py --port 8999
    python 01_segment.py --config config.test.yaml

Standard library only, no dependencies.
"""
import argparse
import hashlib
import json
import random
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

# Which step is calling is worked out from the prompt itself, using markers that
# are the same in every language: JSON key names, and the [role] prefixes the
# pipeline puts on rendered transcripts.
PARA_RE = re.compile(r"^\[(\d+)\]\s", re.M)
# The last turn must stop at the closing fence. Without the `\n---` branch the
# non-greedy match runs to the end of the prompt and swallows the directive
# list, which then shows up as foreign vocabulary and trips the overlap guard.
TURN_RE = re.compile(r"^\[(char|user)\]\s(.*?)(?=\n\n\[(?:char|user)\]|\n---|\Z)", re.M | re.S)

FILLER = [
    "wind", "table", "silence", "corridor", "answer", "shoulder", "glass", "letter",
    "distance", "promise", "footsteps", "shadow", "question", "threshold", "candle",
    "matter", "gesture", "pause", "reason", "weight", "corner", "reply", "memory",
]


def detect_stage(system, user):
    """Identify the calling step from language-independent markers."""
    if '"boundaries"' in user:
        return "segment"
    if "voice_preserved" in user:
        return "validate"
    if '"characters"' in user:
        return "attribute"
    if "[char] " in user or "[user] " in user:
        return "augment"
    if '"user"|"char"' in user:
        return "synth"
    if '"turns"' in user:
        return "convert"
    return "translate"


def extract_scene(user):
    """The text between the first pair of --- fences, or the whole prompt."""
    parts = user.split("---")
    for part in parts[1:]:
        if len(part.strip()) > 40:
            return part.strip()
    return user.strip()


def paragraphs_of(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def sentences_of(text):
    return [s.strip() for s in re.split(r"(?<=[.!?…»\"])\s+", text) if s.strip()]


def reply_segment(user):
    """Mark a boundary before every third numbered paragraph in the window."""
    indices = [int(m) for m in PARA_RE.findall(user)]
    if len(indices) < 4:
        return {"boundaries": []}
    inner = indices[1:-1]
    return {"boundaries": inner[2::3]}


def reply_attribute(user):
    """Invent an attribution from the lines that look like direct speech."""
    scene = extract_scene(user)
    names = ["Alex", "Morgan", "Riley"]
    lines = []
    for raw_line in scene.split("\n"):
        stripped = raw_line.strip()
        if not stripped or not re.match(r'^[—–\-«"„\u201c]', stripped):
            continue
        quote = " ".join(stripped.lstrip('—–-«"„\u201c').split()[:8])
        if quote:
            lines.append({"speaker": names[len(lines) % 2], "quote": quote})
    if not lines:
        # No visible dialogue markers: still return something, so the scene is
        # not silently dropped and the downstream filters get exercised.
        for sentence in sentences_of(scene)[:4]:
            lines.append({"speaker": names[len(lines) % 2],
                          "quote": " ".join(sentence.split()[:8])})
    return {"characters": names[:2], "lines": lines}


def reply_convert(user):
    """Rebuild the scene as alternating turns, keeping the original wording."""
    scene = extract_scene(user)
    blocks = paragraphs_of(scene) or sentences_of(scene)
    turns = []
    for index, block in enumerate(blocks[:8]):
        if index % 3 == 1:
            # A user turn: short, no narration, as the real prompt asks for.
            turns.append({"role": "user", "text": " ".join(block.split()[:12])})
        else:
            turns.append({"role": "char", "text": block})
    if not any(t["role"] == "char" for t in turns):
        turns.insert(0, {"role": "char", "text": scene[:400]})
    return {"turns": turns}


def reply_synth(system, user):
    """Generate a session that is deliberately unlike the anchors.

    The anti-copy and dedupe filters in step 05 exist precisely to catch output
    that echoes its anchors, so the mock has to produce something distinct per
    call or the smoke test would only ever exercise the rejection path.
    """
    seed = int(hashlib.sha1((system + user).encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    tag = f"{seed % 100000:05d}"
    turns = []
    for index in range(rng.randint(3, 5)):
        words = rng.sample(FILLER, 8)
        if index % 2 == 0:
            turns.append({"role": "user",
                          "text": f"Case {tag}-{index}: {' '.join(words[:4])}?"})
        else:
            turns.append({"role": "char",
                          "text": f"Marker {tag}-{index}. {' '.join(words)}. "
                                  f"Something about {words[0]} and {words[1]}, "
                                  f"weighed against {words[2]}."})
    if turns[-1]["role"] == "user":
        words = rng.sample(FILLER, 6)
        turns.append({"role": "char",
                      "text": f"Marker {tag}-end. {' '.join(words)} and nothing further."})
    return {"turns": turns}


def reply_augment(user):
    """Return the transcript unchanged, so the guards see a legitimate rewrite."""
    turns = [{"role": role, "text": text.strip()} for role, text in TURN_RE.findall(user)]
    if not turns:
        turns = [{"role": "char", "text": extract_scene(user)[:400]}]
    return {"turns": turns}


def build_reply(system, user):
    stage = detect_stage(system, user)
    if stage == "segment":
        return json.dumps(reply_segment(user), ensure_ascii=False)
    if stage == "attribute":
        return json.dumps(reply_attribute(user), ensure_ascii=False)
    if stage == "convert":
        return json.dumps(reply_convert(user), ensure_ascii=False)
    if stage == "validate":
        return json.dumps({"voice_preserved": True, "no_invention": True,
                           "split_correct": True, "comment": "mock verdict"},
                          ensure_ascii=False)
    if stage == "synth":
        return json.dumps(reply_synth(system, user), ensure_ascii=False)
    if stage == "augment":
        return json.dumps(reply_augment(user), ensure_ascii=False)
    # translate: echo the input, which is the identity translation
    return user


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - the name is fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")

        content = build_reply(system, user)
        body = json.dumps({
            "id": "mock",
            "object": "chat.completion",
            "model": payload.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *fmt_args):
        pass  # the pipeline's own logging is enough


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Mock LLM server on http://{args.host}:{args.port} — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
