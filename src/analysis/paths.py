"""Shared result-path helpers for V4 analysis notebooks.

Single source of truth for where trial / aggregate files live. Use
``cell_dir`` to build the path to one (condition, model, temp, persona)
cell; notebooks should never hardcode `persona{K}` in their loaders.
"""
from pathlib import Path


def cell_dir(root: Path, cond: str, model_slug: str,
             temp: float, persona: str, version: str = "v4") -> Path:
    """Return the directory holding trial JSONs for one cell.

    Example::

        cell_dir(ROOT, "onet_baseline", "gpt-4o-mini", 0.0, "B")
        # -> .../results/interview_v4/onet_baseline/gpt-4o-mini/temp_0.0/personaB
    """
    return (
        root / "results" / f"interview_{version}" / cond / model_slug
        / f"temp_{temp}" / f"persona{persona}"
    )


def all_results_path(root: Path, cond: str, model_slug: str,
                     temp: float, persona: str, version: str = "v4") -> Path:
    """Return the per-persona ``all_results.json`` path for one cell."""
    return cell_dir(root, cond, model_slug, temp, persona, version) / "all_results.json"
