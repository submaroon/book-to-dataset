Fixtures for the offline smoke test. Nothing here is real data.

    sample_source.txt      generated English source with chapters, scene breaks
                           and one deliberately marker-free block, so the LLM
                           segmentation path is exercised
    sample_source_ru.txt   the same idea in Russian, for the ru language pack
    config.test.yaml       tiny thresholds, mock server, everything enabled
    config.test.ru.yaml    the same for the Russian run
    diluent.jsonl          a small third-party-looking file, deliberately mixing
                           the two role-name conventions and containing
                           {{macros}} that step 08 has to strip
    sample_log.txt         a `Name: line` chat log with a third speaker to fold
                           into narration, for tools/import_transcripts.py
    config.test.import.yaml  workdir for the import path, synthesis disabled
    axes.test.yaml         behavioural directives, as tools/extract_axes.py
                           would produce them, including a held-out group
    run_smoke.sh           start the mock server, run the named steps, stop it
    run_smoke_ru.sh        the same against the Russian config
    run_all_smoke.sh       the same, driving tools/run_all.py

Usage:

    bash tests/run_all_smoke.sh
    bash tests/run_import_smoke.sh 06_format.py 07_augment.py 08_assemble.py
    python 09_publish.py --config tests/config.test.yaml --dry-run
