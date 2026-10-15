This tool has been generated from Claude. With its help, a dataset for my Boulesis model was created.

# Roleplay dataset pipeline

Turns a body of prose — a novel, a collection of stories, an archive of chat
logs — into a roleplay training dataset in ShareGPT format, ready for a LoRA or
QLoRA fine-tune.

Nine numbered steps. Each one reads a file and writes a file, so you can stop
after any of them, look at the result in a text editor, fix something, and
continue from where you were. Every step that calls a model is resumable: an
interrupted run loses at most one item.

All model calls go to a single OpenAI-compatible endpoint, so the pipeline runs
against llama-server, koboldcpp, vLLM, TabbyAPI or a hosted API without any code
changes.

---

## Contents

- [What you get](#what-you-get)
- [Scope and limits](#scope-and-limits) — read this before pointing it at your data
- [The menu](#the-menu) — the easy way in
- [Requirements](#requirements)
- [Try it in two minutes](#try-it-in-two-minutes)
- [Preparing the source text](#preparing-the-source-text)
- [Configuration](#configuration)
- [The steps](#the-steps)
- [Checkpoints — where to actually look](#checkpoints--where-to-actually-look)
- [Running it overnight](#running-it-overnight)
- [Importing data that is already dialogue](#importing-data-that-is-already-dialogue)
- [More than one language](#more-than-one-language)
- [Adding a language](#adding-a-language)
- [The diluent](#the-diluent)
- [Behavioural directives from presets](#behavioural-directives-from-presets)
- [Publishing to Hugging Face](#publishing-to-hugging-face)
- [Training on the result](#training-on-the-result)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## What you get

A JSONL file where each line is one training session:

```json
{"conversations": [
  {"from": "system", "value": "You are Ivan and the scene's narrator..."},
  {"from": "human",  "value": "*Marta ends up near Ivan.*"},
  {"from": "gpt",    "value": "He turned the lantern down before answering..."}
]}
```

Three kinds of session end up in it:

1. **Converted scenes.** Real scenes from your source, restructured as a
   dialogue. The narration is carried over nearly verbatim rather than retold —
   this is what carries the source's voice into the adapter.
2. **Synthetic sessions.** Situations that are *not* in your source, written by
   the model, with the characters reasoning the way your characters reason. Without
   these, the adapter quotes the source instead of generalising from it.
3. **Format variants.** The same content rewritten under different formatting
   instructions, with those instructions written into the system block. This is
   what stops the adapter from ignoring the user's own preset at inference time.

Optionally, a share of third-party roleplay data is mixed in as a diluent, so
the adapter stays usable outside your source material.

---

## The menu

If you would rather not memorise ten commands:

```bash
python menu.py
python menu.py --config config.ru.yaml
```

It prints the ten steps as a numbered list, marks in green the ones that already
produced output, in yellow the ones that stopped partway, and runs whichever you
pick. It also shows the active config, language pack, source file and model
address at the top, so a wrong path is visible before you start rather than after.

The menu runs nothing itself. Every entry shells out to exactly the script you
would have typed, with the same `--config`, so anything started from the menu can
be stopped, resumed and debugged from the command line, and the menu can be
ignored entirely.

Letters, alongside the numbers: `A` runs steps 01-08 unattended, `S` prints the
detailed status report, `V` shows a few finished sessions, `I` imports an
already-dialogue archive, `T` runs the offline smoke test, `H` explains where to
put your files.

The banner also shows what the current prompts are tuned for. That text lives in
`prompts/<lang>.yaml` under `meta.style_note`, next to the prompts it describes —
rewrite it whenever you rewrite them, and the description cannot drift away from
the thing it describes.

## Requirements

```bash
pip install -r requirements.txt
```

That is `requests` and `PyYAML`. Two optional extras:

```bash
pip install pandas pyarrow      # only if a diluent file is .parquet
pip install huggingface_hub     # only for step 09
```

You also need a model behind an OpenAI-compatible endpoint. Anything that
handles instructions and returns valid JSON will do; a mid-sized instruct model
is plenty. A model that refuses on your material is a bad fit — every step
detects refusals, records them separately, and lets you rerun just those items
against a different model.

---

## Try it in two minutes

Before pointing this at your own text, run it against the built-in fixtures.
There is a mock server that speaks the OpenAI API and returns plausible output
for every step, so the whole pipeline runs end to end in seconds with no GPU:

```bash
# terminal 1
python tools/mock_llm_server.py --port 8999

# terminal 2
python tools/run_all.py --config tests/config.test.yaml
python 09_publish.py --config tests/config.test.yaml --dry-run
```

That exercises the plumbing: JSON parsing, resume, the quality gates, the
ShareGPT conversion, the reports. It tells you nothing about quality — the mock
returns mechanical text — but it will catch a broken config or a missing
dependency before you spend a night of compute.

Use the same trick after any change to the config or a language pack.

---

## Preparing the source text

This is the step people skip, and it is the one that decides how good the result
is. The pipeline segments on structure first and only falls back to the model
where structure is missing, so **every marker you leave in the file saves time
and improves the segmentation**.

**Format.** One plain `.txt` file, UTF-8. Other encodings are detected and
reported, but convert to UTF-8 if you can — it removes a whole class of silent
corruption. Put it in `input/` and point `input.path` at it.

**What the file should look like:**

- Paragraphs separated by a **blank line**. Not by indentation, not by a single
  newline. This is the pipeline's basic unit.
- Chapter headings **alone on their line**, under 60 characters:
  `Chapter 12`, `Part Two`, `Prologue`, a bare number, a Roman numeral.
- Scene breaks as a line of repeated separators: `* * *`, `---`, `~~~`.
  Three or more characters, nothing else on the line.
- Dialogue in whatever typography your language uses. Guillemets, straight
  quotes, curly quotes and dash-led lines are all recognised.

**What to remove before you start:**

- Page numbers, running headers and footers left over from a PDF conversion.
  These land in the middle of paragraphs and end up inside training data.
- Footnote markers and footnote bodies.
- Tables of contents, copyright pages, "about the author".
- Hard line wrapping at a fixed column. If every line breaks at 72 characters,
  join the lines within each paragraph first — otherwise every line looks like
  its own paragraph.
- Hyphenation across line breaks (`consid-\neration`).

**A quick sanity check with `sed`/`grep`,** or just open the file and scroll.
Then run the inspector, which is what it is for:

```bash
python 00_inspect.py
```

It reports block counts, the length distribution, how much direct speech there
is, and how many chapters and scene breaks it found — and then tells you what
that means for your thresholds. Read its output before running anything else.

Two numbers matter most:

- **Median block length under 200 characters** is normal — those are ordinary
  paragraphs, and the model does the segmentation.
- **Very little direct speech** is a warning. This pipeline builds dialogue
  transcripts. A source that is almost entirely narration will yield few rows;
  lower `convert.min_dialogue_share` and expect step 03 to skip a lot.

**How much text do you need?** There is no hard floor, but under roughly 200
converted transcripts the dataset is thin, and you should lean harder on step 05
(raise `synth.count`) and on the diluent. `tools/status.py` says so explicitly
when it sees a small dataset.

---

## Configuration

One config file per run. `config.yaml` in the project root is the template; every
value in it has a default in `pipeline/config.py`, so you only keep the lines you
actually change.

Every script takes `--config`:

```bash
python 03_convert.py --config config.ru.yaml
```

The settings you will touch first:

| Setting | What it does |
|---|---|
| `lang` | Which language pack to use. Selects `prompts/<lang>.yaml`. |
| `input.path` | Your source text. |
| `workdir` | Where intermediate files go. **One per language, never shared.** |
| `llm.base_url` | Your model server. |
| `llm.workers` | `1` for a local single-slot server, `6`–`8` for a hosted API. |
| `llm.max_tokens` | Must comfortably exceed your longest scene, or transcripts get truncated. |
| `convert.char` | `auto`, or a character name to build the dataset around one character. |
| `synth.count` | How many synthetic sessions to generate. |
| `assemble.diluent.enabled` | Off by default. See [The diluent](#the-diluent). |

The thresholds — `validate.lexical_overlap_min`, `augment.min_distinctive` and
friends — are worth understanding before changing. Each step's report tells you
whether its threshold is doing anything useful, and the comments in
`pipeline/config.py` explain what each one is defending against.

---

## The steps

Run them in order. Each prints what it did and what to check.

```bash
python 00_inspect.py       # diagnose the source text. No model calls.
python 01_segment.py       # text  -> scenes
python 02_attribute.py     # scenes -> who is present, who says what
python 03_convert.py       # scenes -> roleplay transcripts
python 04_validate.py      # filter the transcripts
python 05_synthesize.py    # generate new sessions anchored to real scenes
python 06_format.py        # -> ShareGPT dataset
python 07_augment.py       # vary the format across sessions
python 08_assemble.py      # + optional diluent -> final dataset
python 09_publish.py       # validate, split, card, upload
```

### 01 — segment

Two passes. The structural one is free: chapter headings, scene breaks, blank
lines. The model pass runs only inside blocks with no structural markers that are
longer than `segment.max_block_chars`, using a sliding window over numbered
paragraphs.

Afterwards, undersized scenes are glued to their neighbour and oversized ones are
split on paragraph borders — never mid-paragraph, because a scene that ends
mid-sentence produces a transcript that ends mid-sentence.

*Output:* `<workdir>/scenes.jsonl`

### 02 — attribute

Who is in each scene, and who speaks each line. This is a separate step because
attribution is the largest source of errors downstream and has to be inspectable
before it disappears into a transcript.

*Output:* `<workdir>/scenes_attr.jsonl`

### 03 — convert

Each scene becomes a transcript. `char` is the most talkative character, or the
one you named in `convert.char`. Its lines plus the surrounding narration become
one turn; the interlocutor's turns are their lines only, bare, with no narration.

That asymmetry is deliberate. It teaches the model to answer richly to thin
input, which is what a real chat session looks like.

*Output:* `<workdir>/rp_raw.jsonl`

### 04 — validate

Cheapest filter first: what share of the transcript's content words also occur in
the source scene. A low share means the converter wrote its own prose. Then,
optionally, a model judge on voice, invention and speaker assignment.

Rejections are kept with their reasons in `rp_rejected.jsonl`.

*Output:* `<workdir>/rp_clean.jsonl`, `<workdir>/rp_rejected.jsonl`

### 05 — synthesize

New situations, written by the model, anchored by two or three real scenes shown
as *a sample of how the character thinks*, not as prose to imitate. Variety comes
from a situation matrix crossed with a character rotation and a random anchor
draw, all seeded by row id so a resumed run stays deterministic.

Three filters run as rows arrive: refusal detection, an anti-copy check against
the anchors, and deduplication (pairwise n-gram similarity plus a stock-phrase
check that catches templates reused across many rows).

The situation matrix lives in the language pack under `synth.situations`. **Edit
it to match your source material** — it is the main lever on what the synthetic
part of your dataset contains.

*Output:* `<workdir>/synth_clean.jsonl`, `<workdir>/synth_rejected.jsonl`

### 06 — format

Converted transcripts plus synthetic rows become ShareGPT sessions with one
system card each. This is already a trainable dataset; if you only want a style
adapter, you can stop here.

*Output:* `format.output` (default `dataset_base.jsonl`)

### 07 — augment

The step that fixes instruction-following.

If every row carries the same system prompt, the model learns that the system
block carries no information and stops attending to it. So each transcript is
rewritten several times under different formatting directives — quote style,
narrative person, tense, turn length, paragraphing, how inner thoughts are
handled — and those directives are written into the system block. The content is
identical across variants, so reading the instruction is the only way to satisfy
the row.

The rewrite receives three things at once: the source scene as the **style**
reference, the finished transcript as the **content**, and the directive list as
the **target format**. Anchoring style to the source scene rather than to the
transcript is what keeps this from being a copy of a copy.

Three guards protect each variant:

| Guard | Catches |
|---|---|
| refusal | the model declined instead of rewriting |
| retention | softening — distinctive words replaced by vague ones |
| overlap | the rewriter substituting its own vocabulary |

Retention is checked before overlap on purpose: softening lowers both, and
reporting "low overlap" for what is actually softening sends you looking in the
wrong place.

Which axes exist is decided by the language pack. Russian has guillemets and
dash-led dialogue; English has neither, so those values simply are not declared
in `prompts/en.yaml`.

There is also a regex-only mode for third-party data, and a self-test:

```bash
python 07_augment.py --diluent source.jsonl out.jsonl
python 07_augment.py --selftest     # no config or model needed
```

*Output:* `augment.output` (default `dataset_aug.jsonl`)

### 08 — assemble

Picks the augmented dataset if step 07 produced one, otherwise the plain one from
step 06, and logs which it used. Then optionally mixes in the diluent: role names
normalised, SillyTavern macros stripped, content filter applied, sampled evenly
across files, trimmed to the session budget on turn borders, and optionally
translated into the run's language.

*Output:* `assemble.output` (default `dataset_final.jsonl`)

### 09 — publish

Validates every session against the rules a trainer will not check for you — at
least one assistant turn, no trailing user turn, no empty messages, no leftover
`{{macros}}` — then splits train/eval, generates a dataset card from the actual
numbers of your run, and uploads.

**Always run `--dry-run` first.** It does everything except the upload.

```bash
python 09_publish.py --dry-run
python 09_publish.py
```

---

## Checkpoints — where to actually look

The pipeline reports a lot. These four checks are the ones that matter, and
skipping them is how you find out at the end of a training run that step 02 was
broken.

| After | What to check |
|---|---|
| **01** | Scene length distribution. A median far below `segment.min_scene_chars` means the chunker is shredding the text — raise the thresholds. |
| **02** | Open 10–15 scenes with a non-zero `unknown_speakers` and read the attribution. Everything downstream trusts it. |
| **03** | Read 5–10 transcripts side by side with their source scenes. If the narration was **retold** rather than carried over, fix the convert prompt in the language pack. |
| **04** | Read `rp_rejected.jsonl`. A reason that repeats is a step 03 prompt problem, not a reason to throw data away. |

Two tools help:

```bash
python tools/status.py                                   # the funnel, stage by stage
python tools/inspect_dataset.py --sample 10 FILE.jsonl   # read actual sessions
python tools/inspect_dataset.py --stats FILE.jsonl       # aggregate numbers
python tools/inspect_dataset.py --health FILE.jsonl      # look for generator garbage
```

`--sample` is the one that catches real problems. The statistics look perfectly
fine on a dataset full of subtly softened rewrites; reading ten of them does not.

---

## Running it overnight

```bash
python tools/run_all.py                     # steps 01 through 08
python tools/run_all.py --from 02 --to 04   # a range
python tools/run_all.py --force             # run past a failed gate
```

Each step is retried up to three times on failure — resume makes that cheap — and
two gates sit between the steps:

- **After 02:** dialogue markers in the scene text are counted and compared with
  the number of attributed lines. If attribution is losing lines in bulk, step 03
  is not started.
- **After 03:** transcript length against source scene length. A transcript far
  shorter than its scene means the model truncated it, usually because
  `llm.max_tokens` is too low.

A gate stops the run rather than warning, because the failure it detects makes
everything downstream worthless, and noticing at 1 a.m. is much cheaper than
noticing at 8 a.m.

---

## Importing data that is already dialogue

Steps 01–04 exist to *build* transcripts out of prose. If what you already have
is a transcript — chat logs, exported roleplay sessions, a ShareGPT dataset —
running it through segmentation and conversion is expensive and lossy. Import it
instead, and pick the pipeline up at step 06:

```bash
python tools/import_transcripts.py logs/*.txt --char Ivan
python 06_format.py
python 07_augment.py
python 08_assemble.py
python 09_publish.py --dry-run
```

Two input formats, detected from the file extension:

- **ShareGPT** `.jsonl` or `.parquet`. Any of the usual role names. System
  messages are read for a character name and then dropped, because step 06 or 07
  writes a fresh card from the language pack — keeping the old one would give you
  two.
- **Plain-text log**, `Name: line` per turn. Continuation lines are joined onto
  the turn above them. Sessions are separated by a break line (`* * *`, `---`)
  or, if the file has none, by two or more blank lines.

The two most talkative speakers become char and user. Anyone else is folded into
the preceding character turn as narration, which is the same rule step 03 applies
to crowd scenes. `{{macros}}` are stripped, and `Name:` prefixes are removed from
the text unless you pass `--keep-names` — the `name_prefix` axis in step 07 is
what varies that, so baking it in would fight with it.

It writes two files into the workdir: `rp_clean.jsonl`, the transcripts, and
`scenes.jsonl`, one scene per transcript holding the same text. The second is not
decoration. Step 07 needs a corpus to measure word rarity against, and it uses
each transcript's own text as the style reference for a row with no source prose.
Building `scenes.jsonl` from the transcripts themselves is what makes that correct
here: the imported dialogue *is* the style reference.

Two things to know:

- **Use a fresh `workdir`.** The importer replaces both files wholesale rather
  than resuming, and it refuses to overwrite without `--force`. Running steps
  01–05 in the same workdir would clobber what it wrote.
- **Set `synth.enabled: false`,** or run step 05 in a separate workdir built from
  prose. Synthesis needs `scenes_attr.jsonl` from step 02, which does not exist
  on an import.

Useful options: `--char` / `--user` to force names, `--min-turns`, `--max-chars`
to trim long sessions on turn borders, `--safety-filter` to apply the content
filter, `--session-break REGEX` for a log with an unusual separator, and
`--dry-run` to see what would be imported before anything is written.

## More than one language

One run is one language. The steps never mix languages internally — a prompt in
one language against a scene in another produces nonsense.

```bash
# two configs, two workdirs, two source files
python tools/run_all.py --config config.en.yaml
python tools/run_all.py --config config.ru.yaml

# merge at the very end
python tools/merge_datasets.py dataset_bilingual.jsonl \
    out_en/dataset_final.jsonl out_ru/dataset_final.jsonl --balance
```

Each session stays monolingual. `--balance` takes an equal number from each file;
without it the larger source dominates and the adapter leans towards its
language.

---

## Adding a language

Copy a pack and translate the strings. No Python is touched.

```bash
cp prompts/en.yaml prompts/de.yaml
# translate, then set lang: de in your config
```

What to pay attention to in the new file:

- **`meta.unknown_speaker`** must be the exact word your prompt tells the model
  to use for an unidentifiable speaker. Step 03 matches against it.
- **`meta.generic_speakers`** — pronouns and role words in your language.
  Attribution returns these regularly, and a transcript whose character is
  literally "he" is useless.
- **`meta.chapter_words`** — the words that begin a chapter heading in your
  language. A short line starting with one of them becomes a free scene boundary.
  Bare numbers and Roman numerals are recognised for every language.
- **`meta.stem_length`** — prefix length for the crude stemming in the quality
  gates. Inflected languages need a short prefix (Russian uses 5) so that a
  legitimate tense change is not read as a word replacement. English uses 6.
- **`axes`** — delete values your language does not use, and add ones it does.
  An axis with fewer than two values is skipped automatically.
- **`synth.situations`** — the situation matrix. Worth rewriting for your source.

A list item containing `: ` **must be quoted**, or YAML parses it as a mapping
and the pack fails to load. The loader checks this and names the offending line,
so you will not discover it three hours into a run.

---

## The diluent

Training on one source alone produces an adapter that is excellent on that
source's rails and brittle off them: unfamiliar settings, formats and user styles
degrade sharply. A share of general roleplay data prevents that. Aim for roughly
20–30% of the final dataset; step 08 warns when it is below 20%.

It is **off by default**, because the files are yours to choose:

```yaml
assemble:
  diluent:
    enabled: true
    files: [limarp.jsonl, bluemoon.jsonl]
    count: 800
    translate: false      # true translates it into the run's language
    safety_filter: true
```

Files are read as ShareGPT `.jsonl` or `.parquet`, role names are normalised
across the common conventions, and macros are stripped.

**About the content filter.** It drops sessions where a minor marker and sexual
content co-occur. It is a backstop, not a cleaning tool — choose sources you
trust. If it drops more than 20% of a file, step 08 says so, and the right
response is to replace the source rather than to rely on the filter. Extend the
marker lists through `assemble.diluent.extra_minor_markers` and
`extra_sexual_markers` when working in a language the built-in lists do not
cover.

**About translation.** Each message is translated in a separate call. That is
slower than sending whole sessions but far more reliable: a long session
translated in one call regularly comes back with turns merged or dropped, which
silently corrupts the structure. It is cached and resumable.

---

## Behavioural directives from presets

Optional, and only worth doing if you have SillyTavern presets lying around.

Format axes come from the language pack and need nothing from you. This is about
the other kind of instruction — "never act for the user", "no purple prose",
"keep NPCs coherent". Mixing them into the system blocks of step 07 teaches the
model to follow instructions it has not seen verbatim.

```bash
python tools/extract_axes.py preset1.json preset2.json
python tools/extract_axes.py --clean          # sort the "unsorted" group
```

The output is a draft — read it and delete the skeleton tags that carry no
instruction. Only the `autonomy` and `prose` groups are used. Anything listed in
`augment.holdout_axes` is deliberately never trained on, so you can measure
whether the adapter generalises to instructions it has never seen.

Disabled preset entries are included on purpose: a preset normally enables one
value of an axis and keeps the alternatives next to it, switched off, and those
alternatives are exactly what makes the axis varied.

---

## Publishing to Hugging Face

```bash
pip install huggingface_hub
hf auth login          # older versions: huggingface-cli login
```

```yaml
publish:
  repo_id: "your-username/your-dataset"
  private: true
  eval_share: 0.05
  license: other
```

```bash
python 09_publish.py --dry-run     # read the card and the numbers first
python 09_publish.py
```

The generated card states plainly that parts of the dataset are model-generated
and were not reviewed line by line. Leave that in. It is true, and it is what
someone downloading it needs to know.

---

## Training on the result

The pipeline stops at the dataset; training is your business. Three things that
follow directly from how the data is built:

- **`train_on_inputs: false`.** The loss belongs on the `gpt` turns only.
- **Do not raise the step count because step 07 multiplied the row count.** Those
  are the same scenes in different wrappers, not more data.
- **Hold out 5–10% of sessions for eval.** Step 09 does this for you.

The test of success is not that the model reproduces your source. It is that you
give it a scenario the source does not contain, and it answers in the source's
voice without quoting it.

---

## Troubleshooting

**Everything is rejected at step 04 with low overlap.** The converter is
paraphrasing instead of carrying the narration over. Read a few rejected
transcripts — if they read like a summary, sharpen `convert.system` in the
language pack. If they look fine and the threshold is simply too tight for your
source, lower `validate.lexical_overlap_min`, but read ten of them first.

**Step 03 skips most scenes.** Either attribution found no real character names
(check `scenes_attr.jsonl` for pronouns and "the voice" in the speaker field —
add them to `meta.generic_speakers`), or your source has too little direct speech
(lower `convert.min_dialogue_share`).

**Transcripts are much shorter than their scenes.** `llm.max_tokens` is too low.
The gate after step 03 in `run_all.py` catches this.

**Step 05 rejects almost everything.** The report tells you which way to go:
*copies the anchors* in bulk means the model is rewriting your source — sharpen
`synth.system` or lower `synth.anchor_max_chars`; *duplicate* in bulk means the
situation matrix is too small; *stock phrasing* means raise `synth.temperature`.

**Step 07 rejects with "softened".** That guard is working. The model is
replacing concrete words with vague ones. Try a different model, or one that does
not moralise about your material. If it is a false positive on a heavily
inflected language, check `meta.stem_length` in the pack.

**Model refusals.** They are recorded in `refusals_NN.jsonl` and **not** marked
done. Point `llm.base_url` at a different model and rerun the step — resume
processes only the refused items.

**A step crashed halfway.** Just run it again. Resume picks up where it stopped.
To redo a step from scratch, delete its output file.

**The language pack fails to load.** The error names the key and the line. Almost
always an unquoted list item containing `: `.

---

## Coming from the earlier version of these scripts

The step numbers changed, because in the earlier version they no longer matched
the order the steps actually ran in: formatting was `05` but ran after synthesis
at `06`, augmentation was wedged in as `06b`, and assembly at `07` read its input
from the wrong place as a result.

| Old | New | Note |
|---|---|---|
| `00_stats.py` | `00_inspect.py` | |
| `01_segment.py` | `01_segment.py` | duplicate function definition removed |
| `02_attribute.py` | `02_attribute.py` | |
| `03_convert.py` | `03_convert.py` | |
| `04_validate.py` | `04_validate.py` | judge pass now optional |
| `06_synth.py` | `05_synthesize.py` | renumbered to its real position |
| `05_format.py` | `06_format.py` | renumbered to its real position |
| `06b_augment.py` | `07_augment.py` | |
| `06b_augment-both.py` | — | gone; use two configs and `--config` |
| `07_assemble_06b.py` | `08_assemble.py` | input resolution fixed and logged |
| — | `09_publish.py` | new |
| `common.py` | `pipeline/` | split into nine modules |
| `run_night.py` | `tools/run_all.py` | any range of steps, not a fixed pair |
| `check_ready.py` | `tools/status.py` | |
| `check-health.py` | `tools/inspect_dataset.py --health` | |
| `rand.py` | `tools/merge_datasets.py` | |
| `2.py` | — | encoding repair folded into `pipeline/files.py` |
| `axes-filter.py` | `tools/extract_axes.py --clean` | |

Other behavioural differences worth knowing about:

- **Prompts moved out of the scripts** into `prompts/<lang>.yaml`. The old
  `{"ru": ..., "en": ...}` dictionaries inside every file are gone.
- **The diluent is optional** and off by default, with no filenames hard-coded.
- **The content filter is implemented**, where the old `minor_check` was a stub
  that always returned `False`.
- **Word matching is script-neutral** (`[^\W\d_]{4,}` rather than a Cyrillic and
  Latin character class), and encoding detection no longer assumes Cyrillic.
- **`convert.GENERIC_SPEAKERS`** is no longer imported through
  `import_module("03_convert")` — that import silently failed and left the filter
  empty. The list now lives in the language pack.
- Old workdirs are **not** compatible field-for-field. Start a fresh `workdir`.

## Scope and limits

A standalone version of this section is in **`BEFORE_YOU_RUN.md`**, written to be
read on its own or passed to someone along with the archive.

It is worth being blunt about this, because "universal" is not the same as "eats
anything".

**What the pipeline expects.** Prose fiction with dialogue, in a language written
with spaces between words. That is the shape steps 01–04 are built around: find
scenes, find who speaks, carry the narration across. Everything below is measured
against the shipped fixtures, not assumed.

### Works out of the box

| | |
|---|---|
| English, Russian | shipped language packs |
| Any encoding | detected and reported; UTF-8 preferred |
| Chapter headings, scene breaks | several conventions, plus bare numbers and Roman numerals |
| Guillemets, straight and curly double quotes, dash-led dialogue, German low quotes, CJK corner brackets | recognised and interconvertible |
| Sources with no structural markup at all | segmented by the model instead; slower, same result |
| A file with no blank lines at all | split on sentence borders, then on whitespace, as a fallback |

### Works, but you have to write a language pack first

Any language written with spaces: German, French, Spanish, Polish, Turkish,
Greek and so on. Copy `prompts/en.yaml`, translate the strings, set the chapter
words, the generic speakers and the stem length. No Python is involved. See
[Adding a language](#adding-a-language).

### Does not work well, and here is exactly why

**Chinese, Japanese, Thai, Korean.** The quality gates tokenise on runs of
letters, which assumes spaces between words. Measured on a one-sentence sample:
English and Russian give 4 tokens, Chinese gives 2 and Japanese gives 2, where a
single "token" is an entire clause — `灯笼是冷的` comes back as one word. The
consequences are concrete:

- `overlap` and `retention` become near-binary: identical text scores 1.0, and any
  real edit drops close to 0. Step 04 and step 07 then reject almost everything.
- `ngrams` returns an empty set on short texts, so the deduplication in step 05
  silently stops working — it does not fail, it just never finds a duplicate.

Making this work needs character n-gram tokenisation for those scripts, which is
not implemented. Do not use the pipeline on them expecting the guards to mean
anything.

**Right-to-left scripts.** Arabic and Hebrew tokenise correctly, but nothing else
has been tested: the dash-dialogue transform, direction marks and punctuation
mirroring are all unknown territory.

**Single-quote dialogue** (`'Like this,' he said`) is deliberately not converted.
The apostrophe is the same character, so any pattern wide enough to catch the
dialogue also mangles `it's` and `'twas`. Such turns are left alone and the
quotes axis simply does not apply to them.

### Wrong tool entirely

**Text without dialogue.** A manual, an essay, a reference work. Step 03 needs two
speakers per scene and will skip everything. `00_inspect.py` says so explicitly
before you waste a night on it.

**Screenplays.** Speaker names sit on their own line above the line
(`ANNA` / `You said midnight.`), which the dialogue heuristics do not recognise.
The model may still attribute correctly in step 02, but the scene structure and
the narration handling are wrong for the format.

**Archives that are already dialogue** — chat logs, existing roleplay
transcripts. Steps 01–04 exist to *create* transcripts from prose; running them
over data that is already in transcript form is expensive and lossy. Do not do
that: use `tools/import_transcripts.py` and enter the pipeline at step 06. See
[Importing data that is already dialogue](#importing-data-that-is-already-dialogue).

### The one-minute check before you commit

```bash
python 00_inspect.py --config your.yaml
```

Read its last section. If it says there is very little direct speech, believe it.
If it reports no chapters and no scene breaks, the whole file goes through the
model and step 01 will be slow. Neither is fatal, but both are better known in
advance.

## Project layout

```
menu.py                  interactive launcher; start here if unsure
README.md                how to use it
BEFORE_YOU_RUN.md        what it handles and what it does not — read first
config.yaml              your settings; one file per run
prompts/
  en.yaml, ru.yaml       every prompt, directive and canned phrase
pipeline/
  config.py              config loading, defaults, CLI
  files.py               encoding detection, JSONL I/O
  llm.py                 chat client, retries, refusal detection, JSON extraction
  lang.py                language pack loading and validation
  textstats.py           overlap, retention, n-grams — the quality gates
  axes.py                format axes, regex transforms, combination picking
  sharegpt.py            ShareGPT conversion, normalisation, trimming
  safety.py              content filter for third-party data
00_inspect.py … 09_publish.py
tools/
  run_all.py             run several steps with gates between them
  status.py              the funnel, stage by stage
  inspect_dataset.py     sample / stats / health on a finished dataset
  import_transcripts.py  bring in data that is already dialogue, skipping 01-05
  extract_axes.py        harvest behavioural directives from ST presets
  merge_datasets.py      merge runs, e.g. one per language
  mock_llm_server.py     fake OpenAI server for offline smoke tests
tests/                   fixtures and smoke-test configs
```

Intermediate files, per run, in `<workdir>`:

```
scenes.jsonl             01  segments
scenes_attr.jsonl        02  + characters and line attribution
rp_raw.jsonl             03  transcripts
rp_clean.jsonl           04  transcripts that passed
rp_rejected.jsonl        04  and why the others did not
synth_clean.jsonl        05  synthetic sessions
synth_rejected.jsonl     05  and why
aug_rejected.jsonl       07  rejected format variants
diluent_translated.jsonl 08  translation cache
refusals_NN.jsonl        any step, model refusals to retry elsewhere
```
