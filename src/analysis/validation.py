"""Sanity checks for V4 ``all_results.json`` files.

Each cell in ``results/interview_v4/`` is the trial output for one
(condition, model, temperature, persona) tuple. If any record in the
file disagrees with that tuple — a stray run from a different persona,
a wrong temperature, a model slug outside the 16-model canonical set —
downstream analysis will silently mix regimes. Call
:func:`validate_all_json_file` from any notebook loader to turn such
drift into an immediate :class:`ValueError` instead of a wrong plot.
"""
from pathlib import Path
import json
from typing import Iterable, List


def _sanitize_slug(model_name: str) -> str:
    """Mirror ``run_interview_onet.sanitize_model_name``."""
    return model_name.replace("/", "_").replace(":", "_")


def validate_all_json_file(
    path,
    *,
    temperature: float,
    persona: str,
    canonical_models: Iterable[str],
) -> List[dict]:
    """Assert an ``all_results.json`` file matches expected labels.

    Parameters
    ----------
    path : Path or str
        Path to an ``all_results.json`` file.
    temperature : float
        Every record must have ``inference_config.temperature`` equal to
        this value.
    persona : str
        Every record must have ``persona`` equal to this value (e.g. "B").
    canonical_models : iterable of str
        Allowed model slugs (after slash → underscore sanitization).
        Every record's ``inference_config.model`` must sanitize to one
        of these slugs. Typically pass ``analysis.labels.MODEL_SLUGS``.

    Returns
    -------
    list of dict
        The trial records, on success. The caller can use them directly
        and avoid loading the file twice.

    Raises
    ------
    ValueError
        On the first record that fails any check, with enough context
        (file, record index, run_id, offending field) to locate the
        problem.
    """
    path = Path(path)
    with open(path) as f:
        trials = json.load(f)

    canonical = set(canonical_models)

    for i, t in enumerate(trials):
        run_id = t.get("run_id", "<no run_id>")

        got_persona = t.get("persona")
        if got_persona != persona:
            raise ValueError(
                f"{path}: record #{i} ({run_id}) has persona={got_persona!r}, "
                f"expected {persona!r}"
            )

        cfg = t.get("inference_config") or {}
        got_temp = cfg.get("temperature")
        if got_temp != temperature:
            raise ValueError(
                f"{path}: record #{i} ({run_id}) has "
                f"temperature={got_temp!r}, expected {temperature!r}"
            )

        raw_model = cfg.get("model", "")
        slug = _sanitize_slug(raw_model)
        if canonical and slug not in canonical:
            raise ValueError(
                f"{path}: record #{i} ({run_id}) has model={raw_model!r} "
                f"(slug={slug!r}), not in the canonical model set"
            )

    return trials
