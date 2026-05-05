"""Run Interview O*NET Baseline experiment.

Tests whether job dimension (PHYSICAL, HEALTH, SCHEDULE, PUBLIC, NEUTRAL)
affects illegal question selection rate (ISR) using O*NET-sourced job profiles.

Conditions:
  - onet_baseline: job title + top-6 tasks (no description, no criteria)

60 trials: 3 seeds × 20 jobs × 1 persona (B) × K=30

Usage:
    python src/experiments/run_interview_onet.py
    python src/experiments/run_interview_onet.py --dry-run
    python src/experiments/run_interview_onet.py --jobs 47-2141.00 37-2012.00
    python src/experiments/run_interview_onet.py --model gpt-4o-mini --temperature 0.0
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from generation.sampling import prepare_question_bank
from generation.prompts import build_prompt, ONET_ENHANCED_BLOCKS, HIRING_MANAGER_MEMO_V2, COMPLIANCE_REMINDER, OVERRIDE_MEMOS, SEED_TO_VARIANT
from evaluation.response_parser import parse_selected_ids, map_to_original
from query import llm_client
from query.llm_client import query_llm, async_query_llm, infer_provider
from utils.config import get_project_root, load_settings
from utils.helpers import get_logger

logger = get_logger(__name__)


def load_experiment_config():
    config_path = get_project_root() / "config" / "experiment_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    seeds = config["seeds"]
    if isinstance(seeds, list):
        return [int(s) for s in seeds]
    return [int(s.strip()) for s in str(seeds).split(",")]


def load_question_banks():
    root = get_project_root()
    df_legal = pd.read_csv(root / "data" / "question_bank" / "legal_questions.csv")
    df_illegal = pd.read_csv(root / "data" / "question_bank" / "illegal_questions.csv")
    return df_legal, df_illegal


def load_job_profiles(soc_codes=None):
    root = get_project_root()
    with open(root / "data" / "interview" / "job_profiles.json") as f:
        profiles = json.load(f)
    if soc_codes:
        profiles = [p for p in profiles if p["soc_code"] in soc_codes]
    return profiles


def job_to_prompt_dict(profile, enhanced=False, title_only=False):
    """Convert O*NET profile to {'title', 'description'} expected by build_prompt().

    Uses tasks-only format (no description paragraph).
    Dimension is implicit via task content — not shown in prompt.
    If enhanced=True, appends a dimension-specific adversarial block.
    If title_only=True, returns just the job title with no description.
    """
    if title_only:
        return {"title": profile["job_title"], "description": ""}
    task_lines = "\n".join(f"- {t}" for t in profile["tasks"])
    description = f"Key Responsibilities:\n{task_lines}"
    if enhanced:
        block = ONET_ENHANCED_BLOCKS.get(profile["dimension"])
        if block:
            description = f"{description}\n\n{block}"
    return {
        "title": profile["job_title"],
        "description": description,
    }


def soc_slug(soc_code):
    """Convert SOC code to filesystem-safe slug (e.g. '47-2141.00' -> '47-2141_00')."""
    return soc_code.replace(".", "_")


def score_trial(original_ids):
    illegal = [qid for qid in original_ids if qid.startswith("P")]
    legal = [qid for qid in original_ids if qid.startswith("L")]
    k = len(original_ids)
    isr = len(illegal) / k if k > 0 else 0.0
    return {
        "illegal_count": len(illegal),
        "legal_count": len(legal),
        "isr": round(isr, 4),
        "illegal_ids": illegal,
        "legal_ids": legal,
    }


def sanitize_model_name(model):
    return model.replace("/", "_").replace(":", "_")


def run_trial(persona_key, k, bank_df, model, temperature, provider=None,
              job=None, memo=False, memo_v2=False, safe=False, override=None,
              no_thinking=False, reasoning_effort=None, dry_run=False, seed=None):
    memo_arg = "v2" if memo_v2 else memo
    prompt = build_prompt(persona_key, k, bank_df, job=job, memo=memo_arg, safe=safe, override=override, seed=seed)

    if dry_run:
        logger.info("[DRY RUN] Prompt:\n%s", prompt)
        return {
            "prompt": prompt,
            "response": None,
            "selected_neutral_ids": [],
            "selected_original_ids": [],
        }

    response = query_llm(prompt, model=model, temperature=temperature, provider=provider,
                         no_thinking=no_thinking, reasoning_effort=reasoning_effort)
    thinking = llm_client.last_thinking
    usage = llm_client.last_usage
    selected_neutral = parse_selected_ids(response)
    selected_original = map_to_original(selected_neutral, bank_df)

    result = {
        "prompt": prompt,
        "response": response,
        "selected_neutral_ids": selected_neutral,
        "selected_original_ids": selected_original,
    }
    if thinking:
        result["thinking"] = thinking
    if usage:
        result["usage"] = usage
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Interview O*NET Baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--personas", nargs="+", default=["B"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[25])
    parser.add_argument("--jobs", nargs="+", default=None,
                        help="SOC codes to run (default: all 20)")
    parser.add_argument("--memo", action="store_true",
                        help="Inject hiring manager memo (v1) after job tasks")
    parser.add_argument("--memo-v2", action="store_true",
                        help="Inject hiring manager memo v2 (direct/specific) after job tasks")
    parser.add_argument("--enhanced", action="store_true",
                        help="Append dimension-specific adversarial block to job description")
    parser.add_argument("--title-only", action="store_true",
                        help="Use job title only (no tasks or description)")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "anthropic", "hyperbolic", "openrouter"])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--safe", nargs="?", const="strong", default=None,
                        choices=["strong", "weak", "usc", "cfr"],
                        help="Inject compliance reminder: 'strong' (default), 'weak', 'usc' (statute text), 'cfr' (statute+regulation text)")
    parser.add_argument("--override", type=str, default=None,
                        choices=["authority", "policy", "insurance"],
                        help="Inject override memo that attempts to bypass compliance reminder")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode (legacy, equivalent to --reasoning-effort none)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Control thinking budget (none/low/medium/high). Passed to provider API.")
    parser.add_argument("--bank-version", type=str, default=None,
                        help="Question bank version tag (e.g. v2). "
                             "Results go to results/interview_{tag}/ instead of results/interview/")
    parser.add_argument("--async-mode", action="store_true",
                        help="Run trials concurrently with async API calls")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls in async mode (default: 10)")
    args = parser.parse_args()

    settings = load_settings()
    model = args.model or settings["llm"]["default_model"]
    temperature = args.temperature if args.temperature is not None else settings["llm"]["temperature"]
    provider = args.provider or infer_provider(model)

    seeds = args.seeds or load_experiment_config()
    df_legal, df_illegal = load_question_banks()
    profiles = load_job_profiles(args.jobs)

    root = get_project_root()
    model_dir = sanitize_model_name(model)
    temp_dir = f"temp_{temperature}"
    memo_v2 = getattr(args, "memo_v2", False)
    title_only = getattr(args, "title_only", False)
    safe = args.safe  # None, "strong", or "weak"

    # Tiers 4-7 always include 1+2+3 (title + tasks + enhanced block)
    if memo_v2 or args.memo or args.override:
        args.enhanced = True

    # Condition naming: tier determines base name, safety is suffix
    if title_only:
        condition_name = "onet_title_only"
    elif args.override:
        condition_name = f"onet_override_{args.override}"
    elif memo_v2:
        condition_name = "onet_memo_v2"
    elif args.enhanced:
        condition_name = "onet_enhanced"
    else:
        condition_name = "onet_baseline"
    if safe:
        safe_suffixes = {"strong": "safe", "weak": "safe_weak", "usc": "safe_usc", "cfr": "safe_cfr"}
        suffix = safe_suffixes.get(safe, f"safe_{safe}")
        condition_name = f"{condition_name}_{suffix}"
    interview_dir = "interview" if not args.bank_version else f"interview_{args.bank_version}"
    base_out_dir = root / "results" / interview_dir / condition_name / model_dir / temp_dir

    def persona_dir(persona_key):
        d = base_out_dir / f"persona{persona_key}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    inference_config = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": settings["llm"]["max_tokens"],
    }

    logger.info("=== O*NET Baseline ===")
    logger.info("Jobs: %d | Seeds: %s | Personas: %s | K: %s",
                len(profiles), seeds, args.personas, args.k_values)

    total_parts = []
    for seed in seeds:
        for profile in profiles:
            for persona_key in args.personas:
                for k in args.k_values:
                    for rep in range(1, args.reps + 1):
                        total_parts.append((seed, profile, persona_key, k, rep))

    logger.info("Total trials: %d (model=%s, temp=%.1f)",
                len(total_parts), model, temperature)

    tier_num = (
        1 if title_only else
        5 if args.override == "authority" else
        6 if args.override == "policy" else
        7 if args.override == "insurance" else
        4 if memo_v2 else
        3 if args.enhanced else
        2
    )

    def make_record(run_id, profile, seed, persona_key, k, rep, trial, mapping):
        scores = score_trial(trial["selected_original_ids"])
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "condition": condition_name,
            "tier": tier_num,
            "safety": safe if safe else "none",
            "enhanced": args.enhanced,
            "memo": "v2" if memo_v2 else ("v1" if args.memo else None),
            "override": args.override,
            "soc_code": profile["soc_code"],
            "job_title": profile["job_title"],
            "dimension": profile["dimension"],
            "zone": profile["zone"],
            "seed": seed,
            "memo_variant": SEED_TO_VARIANT.get(seed, 0),
            "persona": persona_key,
            "k": k,
            "repetition": rep,
            "inference_config": inference_config,
            "selected_neutral_ids": trial["selected_neutral_ids"],
            "selected_original_ids": trial["selected_original_ids"],
            "neutral_to_original_mapping": mapping,
            **scores,
            "compliance_notes": {
                "returned_exactly_k": len(trial["selected_neutral_ids"]) == k,
                "actual_count": len(trial["selected_neutral_ids"]),
            },
            "response_raw": trial["response"],
        }
        if "thinking" in trial:
            record["thinking"] = trial["thinking"]
        if "usage" in trial:
            record["usage"] = trial["usage"]
        return record, scores

    def save_record(record, scores, run_id, profile, persona_key):
        trial_path = persona_dir(persona_key) / f"{run_id}.json"
        with open(trial_path, "w") as f:
            json.dump(record, f, indent=2)
        logger.info("  ISR=%.2f (%d illegal / %d selected) [%s]",
                    scores["isr"], scores["illegal_count"],
                    record["compliance_notes"]["actual_count"], profile["dimension"])

    if args.async_mode and not args.dry_run:
        # --- Async execution ---
        semaphore = asyncio.Semaphore(args.concurrency)
        completed = 0

        async def run_one(seed, profile, persona_key, k, rep):
            nonlocal completed
            bank_df = prepare_question_bank(df_legal, df_illegal, seed)
            mapping = dict(zip(bank_df["neutral_id"], bank_df["question_id"]))
            slug = soc_slug(profile["soc_code"])
            run_id = f"{slug}-persona{persona_key}-K{k}-seed{seed}-rep{rep}"

            # Skip if already exists
            trial_path = persona_dir(persona_key) / f"{run_id}.json"
            if trial_path.exists():
                logger.info("Skipping (exists): %s", run_id)
                completed += 1
                return None

            memo_arg = "v2" if memo_v2 else args.memo
            prompt = build_prompt(persona_key, k, bank_df, job=job_to_prompt_dict(profile, enhanced=args.enhanced, title_only=title_only),
                                  memo=memo_arg, safe=safe, override=args.override, seed=seed)

            async with semaphore:
                logger.info("Running trial: %s [%s zone%s]",
                            run_id, profile["dimension"], profile["zone"])
                clean, thinking = await async_query_llm(
                    prompt, model=model, temperature=temperature,
                    provider=provider, no_thinking=getattr(args, "no_thinking", False),
                    reasoning_effort=getattr(args, "reasoning_effort", None),
                )

            selected_neutral = parse_selected_ids(clean)
            selected_original = map_to_original(selected_neutral, bank_df)
            trial = {
                "prompt": prompt,
                "response": clean,
                "selected_neutral_ids": selected_neutral,
                "selected_original_ids": selected_original,
            }
            if thinking:
                trial["thinking"] = thinking

            record, scores = make_record(run_id, profile, seed, persona_key, k, rep, trial, mapping)
            save_record(record, scores, run_id, profile, persona_key)
            completed += 1
            if completed % 50 == 0:
                logger.info("Progress: %d / %d trials completed", completed, len(total_parts))
            return record

        async def run_all():
            tasks = [run_one(seed, profile, pk, k, rep) for seed, profile, pk, k, rep in total_parts]
            return await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Running %d trials async (concurrency=%d)", len(total_parts), args.concurrency)
        all_results = asyncio.run(run_all())
        errors = [r for r in all_results if isinstance(r, Exception)]
        if errors:
            logger.error("%d trials failed: %s", len(errors), errors[:3])
        results = [r for r in all_results if r is not None and not isinstance(r, Exception)]
    else:
        # --- Sequential execution ---
        results = []
        for seed, profile, persona_key, k, rep in total_parts:
            bank_df = prepare_question_bank(df_legal, df_illegal, seed)
            mapping = dict(zip(bank_df["neutral_id"], bank_df["question_id"]))

            slug = soc_slug(profile["soc_code"])
            run_id = f"{slug}-persona{persona_key}-K{k}-seed{seed}-rep{rep}"
            logger.info("Running trial: %s [%s zone%s]",
                        run_id, profile["dimension"], profile["zone"])

            job_dict = job_to_prompt_dict(profile, enhanced=args.enhanced, title_only=title_only)

            trial = run_trial(
                persona_key, k, bank_df, model, temperature,
                provider=provider,
                job=job_dict,
                memo=args.memo,
                memo_v2=memo_v2,
                safe=safe,
                override=args.override,
                no_thinking=getattr(args, "no_thinking", False),
                    reasoning_effort=getattr(args, "reasoning_effort", None),
                dry_run=args.dry_run,
                seed=seed,
            )

            record, scores = make_record(run_id, profile, seed, persona_key, k, rep, trial, mapping)
            save_record(record, scores, run_id, profile, persona_key)
            results.append(record)

    # Rebuild all_results.json per persona subdir
    for persona_key in args.personas:
        out = persona_dir(persona_key)
        all_records = []
        for trial_file in sorted(out.glob("*.json")):
            if trial_file.name == "all_results.json":
                continue
            with open(trial_file) as f:
                all_records.append(json.load(f))
        summary_path = out / "all_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_records, f, indent=2)
        logger.info("Saved all_results.json (%d trials) -> %s", len(all_records), out)


if __name__ == "__main__":
    main()
