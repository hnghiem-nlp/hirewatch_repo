"""Violation-category loader.

Single source of truth for mapping illegal question IDs to their
protected-class categories. The underlying data lives in
``data/question_bank/illegal_questions.csv`` — the same file the
inference code reads. Notebooks should call :func:`load_categories`
rather than hardcoding the grouping.
"""
from pathlib import Path
import csv


# CSV category key → paper-facing display name. Order is the plot
# column order used in 4.1 (violation-category bar chart) and 4.2
# (dimension × category heatmap): PHYSICAL/HEALTH on the left,
# SCHEDULE in the middle, PUBLIC on the right.
CATEGORY_LABELS = [
    ("disability", "Disability"),
    ("genetic",    "Genetic"),
    ("pregnancy",  "Pregnancy"),
    ("sex",        "Sex/Marital"),
    ("religion",   "Religion"),
    ("nat_origin", "Nat Origin"),
    ("race",       "Race"),
    ("color",      "Color"),
]


def load_categories(bank_path):
    """Return ``[(display_name, [question_ids]), ...]`` from the CSV.

    IDs are stripped of whitespace so downstream comparisons against
    ``trial["illegal_ids"]`` work without surprises (one row in the
    bank, ``P03``, has a trailing space in the CSV).
    """
    by_key = {key: [] for key, _ in CATEGORY_LABELS}
    with open(bank_path) as f:
        for row in csv.DictReader(f):
            qid = row["question_id"].strip()
            cat = (row.get("category") or "").strip()
            if cat in by_key:
                by_key[cat].append(qid)
            elif cat:
                raise ValueError(f"Unknown category {cat!r} for {qid}")
    return [(label, by_key[key]) for key, label in CATEGORY_LABELS]


def default_bank_path(project_root):
    """Return the canonical path to the illegal-questions CSV."""
    return Path(project_root) / "data" / "question_bank" / "illegal_questions.csv"
