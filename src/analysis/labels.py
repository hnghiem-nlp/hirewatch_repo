"""Canonical label mappings for V4 analysis notebooks.

All paper-facing names live in ``config/paper_labels.yaml``. This module
loads that file and exposes the same constants notebooks have always
imported — ``MODELS``, ``TIER_LADDER``, ``FAMILY_COLORS``, etc. The
import interface is unchanged; only the data source moved to YAML.
"""
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Load the YAML config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "paper_labels.yaml"

with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Tier ladder
# ---------------------------------------------------------------------------
# (tier_number, condition_slug, short_label, full_label)
TIER_LADDER = [
    (t["number"], t["slug"], t["short"], t.get("full", t["short"]))
    for t in _CFG["tiers"]
]

# Convenience dicts
SLUG_TO_LABEL = {slug: label for _, slug, label, _ in TIER_LADDER}
SLUG_TO_SHORT = {slug: short for _, slug, short, _ in TIER_LADDER}
SLUG_TO_FULL  = {slug: full  for _, slug, _, full in TIER_LADDER}
SLUG_TO_TIER  = {slug: n for n, slug, _, _ in TIER_LADDER}
TIER_SLUGS    = [slug for _, slug, _, _ in TIER_LADDER]

# Tier metadata (descriptions, groups) for table generators
TIER_META = {t["slug"]: t for t in _CFG["tiers"]}


def tier_subset(include_title_only: bool = True):
    """Return list of (slug, short_label) for plotting.

    By default includes all 7 tiers. Pass ``include_title_only=False``
    to drop tier 1 (Baseline / title-only).
    """
    items = [(slug, label) for _, slug, label, _ in TIER_LADDER]
    if not include_title_only:
        items = items[1:]
    return items


# ---------------------------------------------------------------------------
# Model roster
# ---------------------------------------------------------------------------
# (filesystem_slug, short_name, family)
MODELS = [
    (m["slug"], m["name"], m["family"])
    for m in _CFG["models"]
]

MODEL_SLUGS       = [slug  for slug, _, _ in MODELS]
MODEL_SHORT_NAMES = [name  for _, name, _ in MODELS]
SLUG_TO_NAME      = {slug: name for slug, name, _ in MODELS}
SLUG_TO_FAMILY    = {slug: fam  for slug, _,  fam in MODELS}

# Provider family colors (hex)
FAMILY_COLORS = dict(_CFG["family_colors"])


# ---------------------------------------------------------------------------
# Safety conditions
# ---------------------------------------------------------------------------
SAFETY_LEVELS = [
    (s["suffix"], s["column"], s["label"])
    for s in _CFG["safety_levels"]
]
SAFETY_SUFFIXES     = [s for s, _, _ in SAFETY_LEVELS]
SAFETY_COLUMN_NAMES = [c for _, c, _ in SAFETY_LEVELS]
SAFETY_LABELS       = [l for _, _, l in SAFETY_LEVELS]


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
PERSONAS = _CFG.get("personas", {})


# ---------------------------------------------------------------------------
# Escalation ladder table colors (for the table-generator notebook)
# ---------------------------------------------------------------------------
LADDER_COLORS = _CFG.get("ladder_colors", {})
