# Bugs found by the evaluation suite

Real defects the eval suite found in the bot, and how each was fixed.
Logged from Phase 2 onward.

## Bot defects

None logged yet: the bot has not been run against the golden dataset (no SEC data / API key in
the development sandbox so far). Entries will be added from the first real eval run.

| ID | Category | Symptom | Detected by | Root cause | Fix | Regression test |
|----|----------|---------|-------------|------------|-----|-----------------|
| - | - | - | - | - | - | - |

### How to log a bot defect
1. Find it in `report.md` of an eval run (failures section, unit mix-ups, gate failures).
2. Confirm it is not a dataset or ingestion problem: `filingsqa dataset validate` is clean for the
   item and `context_has_numbers` / `recall_at_k` show whether retrieval found the evidence.
3. Classify the root cause: retrieval, prompt, parsing/chunking, metadata, or model behavior.
4. Fix it, add a regression test (unit test or golden item), re-run the affected category, and
   record before -> after scores in the table above.

## Ingestion defects found on real data

| ID | Symptom | Detected by | Root cause | Fix | Regression test |
|----|---------|-------------|------------|-----|-----------------|
| BUG-1 | Ingestion of MSFT 10-Q `0001193125-26-191507` stopped: DEI `DocumentType` read as `'10- Q'` | Fail-loudly consistency check in `build_filing_meta` (DEI form vs manifest form) | The filing writes the value as `<span>10-</span><span>Q</span>`; `get_text(" ")` put a space between every pair of elements, unlike a browser, which joins inline elements. The same function was used for table cells | `_rendered_text()`: inline children joined as-is, block children and `<br>` separated by a space; used for DEI facts and table cells | `test_dei_value_split_across_inline_spans`, `test_table_cell_inline_spans_are_not_split`, `test_table_cell_block_children_stay_separated` |

## Defects in the eval suite itself

Found while building the suite. Logged because a wrong eval produces false bug reports.

| ID | Symptom | Detected by | Root cause | Fix | Regression test |
|----|---------|-------------|------------|-----|-----------------|
| EV-1 | A hand-written injection item reused a qualitative item's question verbatim | `dataset check-handwritten` (duplicate-question check) | Copy-paste when writing adversarial items | Reworded the injection question | `check_dataset` duplicate check (`test_duplicate_question_is_an_error`) |
| EV-2 | `hallucination_free` was 0.0 while DeepEval marked the case as passed | Probing DeepEval with a scripted fake judge | DeepEval 4.x flipped `HallucinationMetric` to higher-is-better; our code inverted the score and used a max-style threshold (0.3) | Removed inversion, threshold 0.7 as a minimum | `test_hallucination_direction_is_higher_is_better` |
