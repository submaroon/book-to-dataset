# -*- coding: utf-8 -*-
"""Content filter for third-party data.

This runs on the diluent — roleplay logs from public datasets that you did not
write and have not read. It is a backstop, not a cleaning tool: pick sources you
trust, and treat a high drop rate as evidence that the source is wrong for this,
not as the filter doing its job.

What it looks for is the co-occurrence of two things in one session: a marker
that a participant is a minor, and sexual content. Either on its own is
unremarkable; together they are not, and such a session must never end up in
training data.

The checks are deliberately blunt and will produce false positives — a session
about a character's childhood memories can trip the age marker. That trade is
intentional. Losing a handful of usable sessions out of several hundred costs
nothing; the opposite mistake is not recoverable.

Extend the marker lists through `assemble.diluent.extra_markers` in the config
when working in a language the built-in lists do not cover.
"""
import re

# Words that indicate a minor participant. Kept short and unambiguous — a longer
# list mostly adds false positives without catching more real cases.
MINOR_MARKERS = (
    # English
    "child", "children", "kid", "kids", "toddler", "infant", "baby girl", "baby boy",
    "little girl", "little boy", "schoolgirl", "schoolboy", "preteen", "pre-teen",
    "underage", "minor", "middle school", "elementary school", "high schooler",
    "loli", "shota", "grade schooler",
    # Russian
    "ребёнок", "ребенок", "дети", "детей", "малыш", "малышка", "младенец",
    "школьник", "школьница", "школьниц", "подросток", "несовершеннолет",
    "первоклассн", "детсад", "лоли", "шота",
)

# Explicit ages. The number is checked separately, so only the surrounding
# phrasing has to be matched here.
AGE_PATTERNS = (
    re.compile(r"\b(\d{1,2})\s*(?:years?|yrs?)[\s-]*old\b", re.I),
    re.compile(r"\bage[d]?\s*(?:of\s*)?(\d{1,2})\b", re.I),
    re.compile(r"\b(\d{1,2})\s*(?:лет|года|год)\b", re.I),
)
MINOR_AGE_LIMIT = 18

# Sexual context markers. A representative set, not an exhaustive lexicon —
# enough to establish that a session is sexual in nature.
SEXUAL_MARKERS = (
    # English
    "sex", "sexual", "naked", "nude", "aroused", "arousal", "moan", "orgasm",
    "erection", "penetrat", "genital", "nipple", "thrust", "undress", "lust",
    "seduce", "seduct", "foreplay", "climax", "explicit",
    # Russian
    "секс", "сексуал", "голая", "голый", "обнаж", "возбужд", "стон",
    "оргазм", "эрекц", "проникн", "соблазн", "раздева", "похоть",
)


def _contains_any(haystack, needles):
    """First matching needle, or None."""
    for needle in needles:
        if needle in haystack:
            return needle
    return None


def _minor_age_mention(text):
    """An explicitly stated age below the limit, or None."""
    for pattern in AGE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                age = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 0 < age < MINOR_AGE_LIMIT:
                return match.group(0)
    return None


def flag_session(conversation, extra_minor=(), extra_sexual=()):
    """Return a reason string if the session must be dropped, else None.

    `conversation` is a list of {"from": ..., "value": ...} messages.
    """
    text = "\n".join(message.get("value", "") for message in conversation)
    lowered = text.lower()

    minor_hit = _contains_any(lowered, tuple(MINOR_MARKERS) + tuple(extra_minor))
    age_hit = _minor_age_mention(text)
    if not minor_hit and not age_hit:
        return None

    sexual_hit = _contains_any(lowered, tuple(SEXUAL_MARKERS) + tuple(extra_sexual))
    if not sexual_hit:
        return None

    marker = minor_hit or age_hit
    return f"minor marker '{marker}' together with sexual content '{sexual_hit}'"


def filter_sessions(sessions, extra_minor=(), extra_sexual=()):
    """Split sessions into (kept, dropped) where dropped carries its reason."""
    kept, dropped = [], []
    for session in sessions:
        reason = flag_session(session, extra_minor, extra_sexual)
        if reason:
            dropped.append((session, reason))
        else:
            kept.append(session)
    return kept, dropped
