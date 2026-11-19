# Judge reliability

Hand labels used to measure how often the LLM judge agrees with a human (ADR-025).

1. Run a full eval with the judge: `make eval`.
2. Export a sample to label (judge verdicts are hidden on purpose):
   `uv run filingsqa eval label-sample --run eval/reports/<run>/results.json --metric fiscal_period_correct --n 30`
3. Open `eval/judge_agreement/labels_<metric>.jsonl`, set `"human_pass": true/false` on each row,
   optionally add a `note`. Commit the file.
4. `uv run filingsqa eval judge-agreement --run eval/reports/<run>/results.json --labels eval/judge_agreement/labels_<metric>.jsonl`

Report accuracy, Cohen's kappa, and where the judge disagrees (too lenient = false passes,
too strict = false fails). Kappa below ~0.6 means the metric should not gate CI.

No labels are committed yet: they require a real run.
