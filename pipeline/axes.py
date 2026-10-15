# -*- coding: utf-8 -*-
"""Format axes.

The problem this solves: if every training row carries the same system prompt,
the model never sees the instruction change, so it learns not to read it. On
inference the user's own preset is then ignored. The fix is to vary the
formatting instructions across rows while keeping the content fixed, so the only
way to satisfy a row is to actually read its system block.

An "axis" is one formatting decision — quote style, narrative person, turn
length — and its "values" are the mutually exclusive options. Which values exist
for a language is decided by the language pack: Russian has guillemet and
dash-led dialogue, English has neither, and asking an English model for
guillemets would teach it something no preset will ever request.

Two ways to apply an axis:

  tier 1   Pure regex. Rewrites markup only: quotes, emphasis, paragraph
           breaks. Free, lossless, and used on third-party data where there is
           no authorial voice to protect.
  tier 2   An LLM rewrite. Needed for anything that changes the prose itself:
           person, tense, length. Used on the source-derived data in step 07.
"""
import hashlib
import random
import re

# ---------------------------------------------------------------------------
# Axis pool
# ---------------------------------------------------------------------------


def axis_pool(pack):
    """Axes usable for this language: those with at least two defined values.

    An axis with one value cannot teach anything — there is nothing to contrast
    it with — so it is dropped rather than silently wasted.
    """
    declared = pack.get("axes", {}) or {}
    pool = {}
    for axis, values in declared.items():
        usable = [name for name, text in (values or {}).items() if str(text).strip()]
        if len(usable) >= 2:
            pool[axis] = sorted(usable)
    return pool


def directive(pack, axis, value):
    """The instruction text for one axis value, in the run's language."""
    text = pack.get(f"axes.{axis}.{value}")
    assert text, f"language pack has no directive for axis '{axis}' value '{value}'"
    return text


# Axes whose whole point is to add text that was not in the original. For these
# the overlap check is skipped: an honestly added block of inner thoughts would
# otherwise be rejected as invention. The retention check still applies, so
# softening is still caught.
ADDITIVE = {("thoughts", "appended"), ("length", "long")}


def stable_rng(*parts):
    """Deterministic RNG from arbitrary keys.

    Same inputs always produce the same variants, which is what makes a resumed
    run continue rather than start over with different data.
    """
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def pick_combo(pool, rng, max_axes, used):
    """Pick a combination of axis values, biased towards under-used ones.

    Without the bias, random sampling lets a few values dominate and the
    coverage report comes out ragged: some values appear hundreds of times,
    others never, and the never-seen ones teach nothing.
    """
    combo = {}
    if not pool:
        return combo
    count = rng.randint(1, min(max_axes, len(pool)))
    for axis in rng.sample(sorted(pool), count):
        values = pool[axis]
        weights = [1.0 / (1 + used[f"{axis}={value}"]) for value in values]
        threshold, running = rng.random() * sum(weights), 0.0
        for value, weight in zip(values, weights):
            running += weight
            if threshold <= running:
                combo[axis] = value
                break
        combo.setdefault(axis, values[-1])
    return combo


# ---------------------------------------------------------------------------
# SillyTavern macros
# ---------------------------------------------------------------------------

_MACRO_COMMENT = re.compile(r"\{\{//.*?\}\}", re.S)
_MACRO_SETGET = re.compile(r"\{\{(?:set|get|add)var::.*?\}\}", re.S)
_MACRO_IF = re.compile(r"\{\{#if.*?\}\}|\{\{/if\}\}", re.S)
_MACRO_ANY = re.compile(r"\{\{[^{}]*\}\}")


def clean_macros(text, char="", user=""):
    """Strip SillyTavern macros and substitute the names.

    This matters more than it looks. If `{{user}}` reaches the training data
    literally, the model learns to emit curly braces in its answers.
    """
    result = _MACRO_COMMENT.sub("", text or "")
    result = _MACRO_SETGET.sub("", result)
    result = _MACRO_IF.sub("", result)
    result = result.replace("{{user}}", user).replace("{{char}}", char)
    result = result.replace("<USER>", user).replace("<CHAR>", char)
    result = _MACRO_ANY.sub("", result)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


# ---------------------------------------------------------------------------
# Tier 1: regex transforms
#
# Status codes:
#   CHANGED  the markup was actually rewritten
#   OK       the turn already satisfies the target, or is neutral to it
#   NA       the target cannot be reached for this text
#
# The distinction is not cosmetic. An axis is only allowed into a row's system
# block if it produced CHANGED somewhere, otherwise the block would promise a
# format the answers do not follow — the exact mismatch this step exists to fix.
# ---------------------------------------------------------------------------

CHANGED, OK, NA = "changed", "ok", "na"

# Dialogue markers, all recognised so any of them can be converted to any other.
_GUILLEMETS = re.compile(r"«([^»]{1,600})»")
_CURLY_DOUBLE = re.compile(r"\u201c([^\u201d\n]{1,600})\u201d")
# German and Polish low opening quote, closed by either curly double.
# The closer is whichever of the three the source happens to use: German text
# converted from one editor to another mixes them freely.
_LOW_DOUBLE = re.compile("\u201e([^\u201c\u201d\"\n]{1,600})[\u201c\u201d\"]")
# Japanese and Chinese corner brackets.
_CORNER = re.compile(r"[\u300c\u300e]([^\u300d\u300f\n]{1,600})[\u300d\u300f]")
_STRAIGHT_DOUBLE = re.compile(r"\"([^\"\n]{1,600})\"")
# A dash-led line: capture the spoken part only, stopping before the ", — said
# he" attribution tag. A greedy match here used to swallow the tag and the prose
# that followed it.
_DASH_LINE = re.compile(r"^[—–-]\s+(.+?)(?=,\s*[—–]|\s*$)", re.M)

# Single-quote dialogue ('Like this,' he said) is deliberately NOT detected.
# The apostrophe is the same character, so any pattern wide enough to catch it
# also catches "it's" and "'twas" and mangles them. Sources that use single
# quotes keep them; the quotes axis simply never applies to those turns, which
# is reported as OK rather than as a change.
_ASTERISK = re.compile(r"\*([^*\n]{1,600})\*")
_UNDERSCORE = re.compile(r"(?<![\w_])_([^_\n]{1,400})_(?![\w_])")

_SENTINEL = "\x00"
_SENTINEL_SPAN = re.compile(r"\x00([^\x00]*)\x00")


def _result(new, old):
    return (new, CHANGED) if new.strip() != old.strip() else (old, OK)


def t1_quotes(text, style):
    """Convert direct speech markers to the target style.

    All markers are first replaced with a neutral sentinel in a single pass,
    then re-emitted. Doing it in two passes used to corrupt text: when one pair
    was skipped, the regex engine paired the *closing* marker of one line with
    the *opening* marker of the next and ate everything in between.
    """
    staged = _GUILLEMETS.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", text)
    staged = _LOW_DOUBLE.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", staged)
    staged = _CORNER.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", staged)
    staged = _CURLY_DOUBLE.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", staged)
    staged = _STRAIGHT_DOUBLE.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", staged)
    staged = _DASH_LINE.sub(lambda m: f"{_SENTINEL}{m.group(1)}{_SENTINEL}", staged)
    if _SENTINEL not in staged:
        return text, OK  # no direct speech at all: the rule is not violated

    if style == "guillemets":
        replace = lambda m: f"«{m.group(1)}»"  # noqa: E731
    elif style in ("double", "straight"):
        replace = lambda m: f'"{m.group(1)}"'  # noqa: E731
    elif style == "plain":
        replace = lambda m: m.group(1)  # noqa: E731
    elif style == "corner":
        replace = lambda m: f"\u300c{m.group(1)}\u300d"  # noqa: E731
    elif style == "low":
        replace = lambda m: f"\u201e{m.group(1)}\u201c"  # noqa: E731
    elif style == "dash":
        def replace(match):
            at_line_start = match.start() == 0 or match.string[match.start() - 1] == "\n"
            return ("" if at_line_start else "\n") + f"— {match.group(1)}"
    else:
        return text, NA

    out = _SENTINEL_SPAN.sub(replace, staged)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out).replace(_SENTINEL, "")
    return _result(out, text)


def t1_emphasis(text, style):
    """Normalise emphasis markup: *asterisks*, _underscores_, or neither.

    The axis is called "emphasis" and not "thoughts" on purpose. A regex cannot
    tell an action from an inner thought. The SillyTavern convention is
    asterisks for actions and underscores for thoughts, but real data mixes
    them, so promising "thoughts in underscores" in the system block while
    quietly recolouring actions too would be the very mismatch this step fixes.
    """
    if not (_ASTERISK.search(text) or _UNDERSCORE.search(text)):
        return text, OK
    if style == "asterisks":
        return _result(_UNDERSCORE.sub(r"*\1*", text), text)
    if style == "underscore":
        return _result(_ASTERISK.sub(r"_\1_", text), text)
    if style == "plain":
        return _result(_UNDERSCORE.sub(r"\1", _ASTERISK.sub(r"\1", text)), text)
    return text, NA


# Sentence boundary: end punctuation followed by whitespace and something that
# starts a sentence. Kept deliberately loose — it only decides where to break a
# paragraph, so a miss costs nothing.
_SENTENCE = re.compile(r"(?<=[.!?…»\"])\s+(?=[«\"\w—])", re.UNICODE)


def t1_paragraphs(text, style):
    """Make the text denser or airier without touching the words."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if style == "dense":
        if len(paragraphs) <= 1:
            return text, OK
        middle = len(paragraphs) // 2
        joined = "\n\n".join(["\n".join(paragraphs[:middle]), "\n".join(paragraphs[middle:])])
        return _result(joined, text)
    if style == "airy":
        out, split_any = [], False
        for paragraph in paragraphs:
            sentences = _SENTENCE.split(paragraph)
            if len(sentences) >= 4:
                middle = len(sentences) // 2
                out += [" ".join(sentences[:middle]), " ".join(sentences[middle:])]
                split_any = True
            else:
                out.append(paragraph)
        return _result("\n\n".join(out), text) if split_any else (text, OK)
    return text, NA


# Which axes can be applied by regex, and which of their values the transforms
# understand. Values the language pack declares but that are not listed here are
# simply not available for tier-1 use.
TIER1 = {
    "quotes": (t1_quotes, ["guillemets", "double", "dash", "plain", "corner", "low"]),
    "emphasis": (t1_emphasis, ["asterisks", "underscore", "plain"]),
    "paragraphs": (t1_paragraphs, ["dense", "airy"]),
}


def tier1_pool(pack):
    """Axis pool restricted to what the regex transforms can actually do."""
    available = axis_pool(pack)
    pool = {}
    for axis, (_fn, supported) in TIER1.items():
        values = [v for v in available.get(axis, []) if v in supported]
        if len(values) >= 2:
            pool[axis] = values
    return pool


def format_block(directives):
    """The <format> block appended to a system card."""
    if not directives:
        return ""
    lines = "\n".join(f"- {d}" for d in directives)
    return f"<format>\n{lines}\n</format>"
