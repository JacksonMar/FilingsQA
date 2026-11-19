"""Evaluation suite for FilingsQA (the main product of the project)."""

import os

# DeepEval sends telemetry by default; the eval suite must not phone home.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
