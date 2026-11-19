# Bugs found by the evaluation suite

Real defects the eval suite found in the bot, and how each was fixed.
Logged from Phase 2 onward.

## Bot defects

None logged yet: the bot has not been run against the golden dataset (no SEC data / API key in
the development sandbox so far). Entries will be added from the first real eval run.

| ID | Category | Symptom | Detected by | Root cause | Fix | Regression test |
|----|----------|---------|-------------|------------|-----|-----------------|
| - | - | - | - | - | - | - |

## Defects in the eval suite itself

Found while building the suite. Logged because a wrong eval produces false bug reports.

| ID | Symptom | Detected by | Root cause | Fix | Regression test |
|----|---------|-------------|------------|-----|-----------------|
| EV-1 | A hand-written injection item reused a qualitative item's question verbatim | `dataset check-handwritten` (duplicate-question check) | Copy-paste when writing adversarial items | Reworded the injection question | `check_dataset` duplicate check (`test_duplicate_question_is_an_error`) |
| EV-2 | `hallucination_free` was 0.0 while DeepEval marked the case as passed | Probing DeepEval with a scripted fake judge | DeepEval 4.x flipped `HallucinationMetric` to higher-is-better; our code inverted the score and used a max-style threshold (0.3) | Removed inversion, threshold 0.7 as a minimum | `test_hallucination_direction_is_higher_is_better` |
