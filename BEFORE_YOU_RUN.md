# Before you run this: what it handles and what it does not

A short, honest note to read before pointing this pipeline at your data. The
README explains how to use it; this explains where it will work well, where it
will work badly, and where it is the wrong tool. Everything below was measured
against the pipeline's own fixtures, not assumed.

---

## In one paragraph

The pipeline turns prose fiction with dialogue into a ShareGPT roleplay dataset.
It expects a language written with spaces between words, and it expects
conversations in the source. English and Russian work out of the box; any other
space-separated language works after you translate one YAML file. Chinese,
Japanese, Thai and Korean will run but the quality filters become meaningless on
them. Text without dialogue is the wrong input. Data that is *already* dialogue
should skip the first five steps entirely — there is a tool for that.

---

## Green: works as shipped

**Languages.** English and Russian, with prompt packs included.

**Encodings.** Anything decodable is detected, and the pipeline says which
encoding it used and whether the result looks suspicious. UTF-8 is still the
right thing to feed it.

**Structure.** Chapter headings in several conventions, plus bare numbers and
Roman numerals alone on a line. Scene breaks written as `* * *`, `---`, `~~~` and
similar. Blank-line paragraphs.

**Dialogue typography.** Guillemets `«»`, straight `"` and curly `“”` double
quotes, dash-led lines `— like this`, German low quotes `„…“`, and Japanese and
Chinese corner brackets `「」`. All of these are recognised and can be converted
into one another.

**Sources with no markup at all.** A file with no chapters and no scene breaks is
segmented by the model instead. It works; it is just slower, and `00_inspect.py`
tells you so in advance.

**A file with no blank lines whatsoever.** Split on sentence borders, then on
whitespace, as a fallback. A 25 000-character wall of text becomes properly sized
scenes rather than one unusable blob.

---

## Yellow: works, but you have to do something first

### Any other space-separated language

German, French, Spanish, Polish, Turkish, Greek, Portuguese, Italian, and so on.

Copy `prompts/en.yaml` to `prompts/<code>.yaml`, translate the strings, and set
`lang:` in the config. No Python is touched. Four keys need real thought rather
than a literal translation:

- `meta.chapter_words` — the words that begin a chapter heading in your language.
- `meta.generic_speakers` — pronouns and role words ("he", "the voice", "the
  stranger"). Attribution returns these constantly, and a transcript whose
  character is literally "he" is useless.
- `meta.stem_length` — prefix length for the crude stemming in the quality gates.
  Inflected languages need a short prefix so that an honest tense change is not
  read as a word replacement. Russian uses 5, English 6.
- `axes` — delete the values your language does not use, add the ones it does. An
  axis with fewer than two values is skipped automatically.

One trap: a YAML list item containing `: ` **must be quoted**, or it parses as a
mapping. The pack loader checks this and names the offending line, so you will
find out immediately rather than three hours into a run.

### Data that is already dialogue

Chat logs, exported roleplay sessions, an existing ShareGPT dataset. Steps 01–04
exist to *create* transcripts from prose. Running them over data that is already
in transcript form is expensive and lossy.

Use `tools/import_transcripts.py` instead and enter at step 06. It reads ShareGPT
files and `Name: line` logs, folds extra speakers into narration, strips
`{{macros}}`, and writes what step 04 would have written. Then run 06, 07 and 08
normally.

Set `synth.enabled: false` when you do this, and use a fresh `workdir`.

---

## Red: it will run, and the output will be worse than it looks

### Chinese, Japanese, Thai, Korean

This is not a prompt problem. It is the measurement layer.

The quality gates tokenise on runs of letters, which assumes spaces between
words. Measured on a single sentence: English and Russian give 4 tokens, Chinese
gives 2, Japanese gives 2 — and a "token" there is an entire clause. `灯笼是冷的`
comes back as one word.

Two consequences, both silent:

- **`overlap` and `retention` become near-binary.** Identical text scores 1.0;
  any real edit drops close to 0. Step 04 and step 07 will reject nearly
  everything, and the reasons in the reject files will look like model failures
  when they are measurement failures.
- **Deduplication stops working.** `ngrams` returns an empty set on short CJK
  texts, so step 05 never finds a duplicate. It does not error. It just never
  rejects anything, and you get a synthetic set full of near-copies.

Fixing this properly needs character n-gram tokenisation for those scripts. It is
not implemented. If you run on these languages anyway, treat every threshold in
the config as decorative and read a much larger sample of the output by hand.

### Right-to-left scripts

Arabic and Hebrew tokenise correctly, so the gates are meaningful. Nothing else
has been tested: the dash-dialogue transform, direction marks, and punctuation
mirroring are unknown. Expect to check the output carefully.

### Single-quote dialogue

`'Like this,' he said` is deliberately left alone. The apostrophe is the same
character, so any pattern wide enough to catch the dialogue also mangles `it's`
and `'twas`. Such turns pass through untouched and the quote axis simply does not
apply to them. Nothing breaks; you just get less format variety.

---

## Wrong tool entirely

**Text without dialogue.** A manual, an essay, a reference work, a wiki dump.
Step 03 needs two identified speakers per scene and will skip everything.
`00_inspect.py` reports this before you spend a night on it — the line to watch
for is the one about there being very little direct speech.

**Screenplays and stage plays.** The speaker name sits on its own line above the
line it belongs to:

```
ANNA
You said midnight.
```

The dialogue heuristics do not recognise that shape. The model may still
attribute correctly in step 02, but the scene structure and the narration
handling are built for prose, not for a format where there is almost no
narration. Convert to `Name: line` first and use the importer.

**Very short sources.** There is no hard floor, but under roughly 200 accepted
transcripts the dataset is thin. Lean on step 05 (raise `synth.count`) and on the
diluent. `tools/status.py` says so explicitly when it sees a small dataset.

---

## Two habits that will save you a night

**Run the offline smoke test after any config change.** There is a mock server
that speaks the OpenAI API and returns plausible output for every step, so the
whole pipeline runs in seconds with no GPU:

```bash
python tools/mock_llm_server.py --port 8999 &
python tools/run_all.py --config tests/config.test.yaml
```

It tests the plumbing, not the quality. It will still catch a broken config, a
malformed language pack, or a missing dependency before you commit real compute.

**Start from the menu if you are unsure.** `python menu.py` lists the ten steps,
marks in green what has already run, and launches them for you. It also shows the
active config, source file and model address at the top, so a wrong path is
visible before you start.

**Run `00_inspect.py` on your actual file and read its last section.** It reports
block counts, length distribution, how much direct speech there is, and how many
chapters and scene breaks it found — and then says what that means for your
thresholds. Most of the failures described in this note announce themselves
there, for free, in one second.

---

## What quality means here

The pipeline's filters catch mechanical failures: the converter paraphrasing
instead of carrying prose across, a rewrite that softens the source, a generator
that repeats itself, a model refusing. They cannot tell you whether the dataset
is any good.

The statistics look perfectly healthy on a dataset full of subtly flattened
rewrites. Reading ten sessions does not:

```bash
python tools/inspect_dataset.py --sample 10 dataset_final.jsonl
```

Do that before you train, every time. And keep the checkpoints in the README —
especially reading a handful of transcripts side by side with their source scenes
after step 03. That single check catches more real problems than every threshold
in the config put together.
