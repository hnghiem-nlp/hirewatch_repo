"""Question bank sampling and neutral ID assignment for interview experiments."""

import pandas as pd


def prepare_question_bank(df_legal, df_illegal, seed):
    """Shuffle combined question bank and assign neutral IDs.

    Combines legal and illegal questions, shuffles with a fixed seed,
    and assigns neutral IDs (Q001, Q002, ...) to prevent P/L prefix
    leakage while preserving original IDs for grading.

    Args:
        df_legal: DataFrame with columns 'question_id', 'question'.
        df_illegal: DataFrame with columns 'question_id', 'question'.
        seed: Random seed for reproducible shuffling.

    Returns:
        DataFrame with columns: neutral_id, question, question_id.
    """
    combined = pd.concat(
        [df_legal[["question_id", "question"]], df_illegal[["question_id", "question"]]],
        ignore_index=True,
    )

    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled.insert(0, "neutral_id", [f"Q{i + 1:03d}" for i in range(len(shuffled))])

    return shuffled
