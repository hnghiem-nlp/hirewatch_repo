"""Parse LLM responses and map neutral IDs back to original question IDs."""

import re


def parse_selected_ids(response_text):
    """Extract question IDs (Q001, Q002, etc.) from model response.

    Handles numbered lists, comma-separated, whitespace-separated formats.

    Args:
        response_text: Raw text returned by the LLM.

    Returns:
        List of neutral ID strings (e.g., ['Q001', 'Q017', 'Q045']).
    """
    return re.findall(r"\bQ\d{3}\b", response_text)


def map_to_original(selected_neutral_ids, bank_df):
    """Map neutral IDs back to original question IDs for grading.

    Args:
        selected_neutral_ids: List of neutral IDs (e.g., ['Q001', 'Q017']).
        bank_df: DataFrame from prepare_question_bank() with neutral_id, question_id.

    Returns:
        List of original question IDs (e.g., ['P53', 'L01']).
    """
    id_map = dict(zip(bank_df["neutral_id"], bank_df["question_id"]))
    return [id_map[nid] for nid in selected_neutral_ids if nid in id_map]
