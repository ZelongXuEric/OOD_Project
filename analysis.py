from collections import defaultdict
import torch
import numpy as np
import re
import config
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from typing import Tuple, Dict, List, Optional, DefaultDict, Any
import utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_word_token_spans_from_offsets(
    token_ids_segment: np.ndarray,      # (T_tok_seg) - token IDs for the relevant segment
    token_offsets_segment: np.ndarray,  # (T_tok_seg, 2) - corresponding RELATIVE offsets for the relevant segment
    original_text_segment: str, # The ground truth text for this segment (e.g., item['input_seq'] or target_no_marker_str)
    tokenizer: AutoTokenizer,
    segment_name: str, 
    sample_index: int, 
    layer_idx_str: str
) -> Tuple[List[str], Dict[int, List[int]], List[Optional[int]]]:

    ground_truth_words = [w for w in original_text_segment.split(' ') if w]
    if not ground_truth_words:
        logger.debug(f"    {segment_name} (Sample {sample_index} layer {layer_idx_str}): original_text_segment '{original_text_segment}' is empty or has no valid word.。")
        return [], defaultdict(list), [None] * len(token_ids_segment)

    aligned_word_to_token_indices: DefaultDict[int, List[int]] = defaultdict(list)
    
    current_char_search_start_in_segment = 0

    gt_word_char_spans_relative: List[Optional[Tuple[int, int]]] = [None] * len(ground_truth_words)

    for gt_word_idx, gt_word_str in enumerate(ground_truth_words):
        try:
            start_char_rel = original_text_segment.index(gt_word_str, current_char_search_start_in_segment)
            end_char_rel = start_char_rel + len(gt_word_str)
            gt_word_char_spans_relative[gt_word_idx] = (start_char_rel, end_char_rel)

            current_char_search_start_in_segment = end_char_rel
        except ValueError:
            logger.warning(f"    {segment_name} (sample {sample_index} layer {layer_idx_str}): GT word '{gt_word_str}' (idx {gt_word_idx}) not found in "
                           f"'{original_text_segment}' (starting from char {current_char_search_start_in_segment}).")

            continue
    
    for token_idx_in_segment, (tok_start_rel, tok_end_rel) in enumerate(token_offsets_segment):
        if tok_start_rel == tok_end_rel == 0 and tok_start_rel == 0 : # Skip padding/special tokens (often (0,0))
            continue
        try:

            current_token_id = token_ids_segment[token_idx_in_segment]
            token_text_debug = tokenizer.decode([current_token_id], skip_special_tokens=False)
        except Exception as e_decode_token_debug:
            logger.warning(f"    {segment_name} (sample {sample_index} layer {layer_idx_str}): "
                           f"Failed to decode token ID {token_ids_segment[token_idx_in_segment]} (index {token_idx_in_segment}): {e_decode_token_debug}")
            token_text_debug = "[decoding error]"

        for gt_word_idx, word_span_rel in enumerate(gt_word_char_spans_relative):
            if word_span_rel is None:
                continue

            word_start_rel, word_end_rel = word_span_rel

            overlap_start = max(word_start_rel, tok_start_rel)
            overlap_end = min(word_end_rel, tok_end_rel)

            if overlap_start < overlap_end:
                aligned_word_to_token_indices[gt_word_idx].append(token_idx_in_segment)
                logger.debug(f"      Token {token_idx_in_segment} ('{token_text_debug}', ID: {token_ids_segment[token_idx_in_segment]}) "
                             f"span ({tok_start_rel}, {tok_end_rel}) "
                             f"matched GT word {gt_word_idx} ('{ground_truth_words[gt_word_idx]}') "
                             f"span ({word_start_rel}, {word_end_rel})")
                break
            else:
                if gt_word_idx == len(ground_truth_words) - 1:
                    logger.debug(f"      Token {token_idx_in_segment} ('{token_text_debug}', ID: {token_ids_segment[token_idx_in_segment]}) "
                                 f"span ({tok_start_rel}, {tok_end_rel}) did not match any GT word.")
    
    num_gt_words_with_tokens = len(aligned_word_to_token_indices)

    aligned_token_to_word_idx: List[Optional[int]] = [None] * len(token_ids_segment)
    for gt_word_idx, token_indices_for_gt_word in aligned_word_to_token_indices.items():
        for tok_idx_in_segment in token_indices_for_gt_word:
            if 0 <= tok_idx_in_segment < len(aligned_token_to_word_idx):
                 aligned_token_to_word_idx[tok_idx_in_segment] = gt_word_idx

    return ground_truth_words, aligned_word_to_token_indices, aligned_token_to_word_idx

def _extract_word_states_from_pretokenized(
    states_tensor: torch.Tensor,
    input_ids_full: torch.Tensor,
    offset_mapping_full: torch.Tensor,
    item: Dict[str, Any], 
    tokenizer: AutoTokenizer,
    sample_index: int,
    layer_idx_str: str,
    text_for_offsets_reference: str
) -> Optional[Dict[str, Any]]:
    if torch is None: raise RuntimeError("Torch is not loaded.")

# 1. Data preparation and validation
    try:
        states_np = states_tensor.cpu().float().numpy()
        ids_full_np = input_ids_full.cpu().numpy()
        offsets_full_np = offset_mapping_full.cpu().numpy()
    except Exception as e_conv:
        logger.error(f"Sample {sample_index} layer {layer_idx_str}: failed converting Tensors to NumPy: {e_conv}")
        return None

    T_state_actual = states_np.shape[0]
    T_tok_full_from_ids = ids_full_np.shape[0]

    if T_state_actual != T_tok_full_from_ids:
        logger.warning(f"Sample {sample_index} layer {layer_idx_str}: state length ({T_state_actual}) does not match token length ({T_tok_full_from_ids}) for text_for_offsets_reference.")
        if T_state_actual == T_tok_full_from_ids - 1 and ids_full_np[0] == tokenizer.bos_token_id:
            logger.info("  Detected states shorter by 1 and first token is BOS; assuming states correspond to tokens[1:], adjusting offsets and ids.")
            ids_full_np = ids_full_np[1:]
            offsets_full_np = offsets_full_np[1:]
            if T_state_actual != ids_full_np.shape[0]:
                logger.error(f"  Adjusted lengths ({T_state_actual} vs {ids_full_np.shape[0]}) still do not match. Skipping sample.")
                return None
        else:
            logger.error("  Lengths do not match and cannot auto-correct. Skipping sample.")
            return None

    input_seq_gt_str = item['input_seq'].strip()
    target_gt_with_marker_str = item['target']
    target_gt_no_marker_str = target_gt_with_marker_str.replace(config.END_MARKER, "").strip()

    logger.debug(f"\nProcessing sample {sample_index} layer {layer_idx_str} (_extract_word_states_from_pretokenized v8)")
    logger.debug(f"  Text for Offsets Reference (len {len(text_for_offsets_reference)}): '{text_for_offsets_reference[:250]}...'")
    logger.debug(f"  Input Seq GT (item['input_seq'].strip()): '{input_seq_gt_str}'")
    logger.debug(f"  Target No Marker GT (item['target_no_marker'].strip()): '{target_gt_no_marker_str}'")

    # 2. Locate character spans of GT sequences in text_for_offsets_reference
    input_seq_char_start_in_full = -1
    input_seq_char_end_in_full = -1
    output_seq_char_start_in_full = -1
    output_seq_no_marker_char_end_in_full = -1

    content_separator = "\n\n"  # Llama3 chat template uses blank line to separate header and content

    if input_seq_gt_str:
        user_header_tag = "<|start_header_id|>user<|end_header_id|>"
        last_user_header_pos = text_for_offsets_reference.rfind(user_header_tag)
        if last_user_header_pos != -1:
            start_of_user_content_search = last_user_header_pos + len(user_header_tag)
            actual_user_content_start_marker = text_for_offsets_reference.find(content_separator, start_of_user_content_search)
            if actual_user_content_start_marker != -1:
                actual_user_content_start_offset = actual_user_content_start_marker + len(content_separator)
                temp_input_start = text_for_offsets_reference.find(input_seq_gt_str, actual_user_content_start_offset)
                next_header_after_user = text_for_offsets_reference.find("<|start_header_id|>", actual_user_content_start_offset)
                if temp_input_start != -1 and temp_input_start >= actual_user_content_start_offset and \
                   (next_header_after_user == -1 or temp_input_start < next_header_after_user):
                    input_seq_char_start_in_full = temp_input_start
                    input_seq_char_end_in_full = input_seq_char_start_in_full + len(input_seq_gt_str)
                    logger.debug("  GT Input Seq found in last user message block via template structure.")
                else:  # Fallback
                    logger.warning(f"Sample {sample_index} layer {layer_idx_str}: GT Input Seq '{input_seq_gt_str}' not found in last user message block; attempting global rfind.")
                    input_seq_char_start_in_full = text_for_offsets_reference.rfind(input_seq_gt_str)
                    if input_seq_char_start_in_full != -1:
                        input_seq_char_end_in_full = input_seq_char_start_in_full + len(input_seq_gt_str)
                        logger.warning("  Fallback: GT Input Seq found in text_for_offsets_reference via rfind.")
                    else:
                        logger.error(f"Sample {sample_index} layer {layer_idx_str}: GT Input Seq '{input_seq_gt_str}' not found at all. Skipping.")
                        return None
            else:
                logger.error(f"Sample {sample_index} layer {layer_idx_str}: content separator '{content_separator}' not found after last user header. Skipping.")
                return None
        else:
            logger.error(f"Sample {sample_index} layer {layer_idx_str}: last user header '{user_header_tag}' not found. Skipping.")
            return None
    else:
        logger.error(f"Sample {sample_index} layer {layer_idx_str}: GT Input Seq is empty. Skipping.")
        return None

    if target_gt_no_marker_str:
        assistant_header_tag = "<|start_header_id|>assistant<|end_header_id|>"
        search_context_for_assistant_header = text_for_offsets_reference[input_seq_char_end_in_full if input_seq_char_end_in_full != -1 else 0:]
        relative_assistant_header_pos = search_context_for_assistant_header.rfind(assistant_header_tag)
        if relative_assistant_header_pos != -1:
            absolute_assistant_header_pos = (input_seq_char_end_in_full if input_seq_char_end_in_full != -1 else 0) + relative_assistant_header_pos
            start_of_assistant_content_search = absolute_assistant_header_pos + len(assistant_header_tag)
            actual_assistant_content_start_marker = text_for_offsets_reference.find(content_separator, start_of_assistant_content_search)
            if actual_assistant_content_start_marker != -1:
                actual_assistant_content_start_offset = actual_assistant_content_start_marker + len(content_separator)
                temp_target_start = text_for_offsets_reference.find(target_gt_no_marker_str, actual_assistant_content_start_offset)
                next_eot_after_assistant = text_for_offsets_reference.find("<|eot_id|>", actual_assistant_content_start_offset)
                if temp_target_start != -1 and temp_target_start >= actual_assistant_content_start_offset and \
                   (next_eot_after_assistant == -1 or temp_target_start + len(target_gt_no_marker_str) <= next_eot_after_assistant):
                    output_seq_char_start_in_full = temp_target_start
                    output_seq_no_marker_char_end_in_full = output_seq_char_start_in_full + len(target_gt_no_marker_str)
                    logger.debug("  GT Target Seq found in last assistant message block via template structure.")
                else:
                    logger.warning(f"Sample {sample_index} layer {layer_idx_str}: GT Target Seq '{target_gt_no_marker_str}' not properly found in last assistant block.")
                    output_seq_char_start_in_full = -1
            else:
                logger.warning(f"Sample {sample_index} layer {layer_idx_str}: content separator '{content_separator}' not found after last assistant header.")
                output_seq_char_start_in_full = -1
        else:
            logger.warning(f"Sample {sample_index} layer {layer_idx_str}: assistant header '{assistant_header_tag}' not found after GT input.")
            output_seq_char_start_in_full = -1
    else:
        logger.debug(f"Sample {sample_index} layer {layer_idx_str}: GT Target Seq is empty.")
        output_seq_char_start_in_full = -1

    logger.debug(f"  Final character span of input_seq in full_text: [{input_seq_char_start_in_full}, {input_seq_char_end_in_full})")
    if target_gt_no_marker_str:
        if output_seq_char_start_in_full != -1:
            logger.debug(f"  Final character span of target_seq (no marker) in full_text: [{output_seq_char_start_in_full}, {output_seq_no_marker_char_end_in_full})")
        else:
            logger.debug(f"  Target_seq (no marker) '{target_gt_no_marker_str}' could not be clearly located in full_text.")

    # 3. Find token spans for input and output segments (using offsets_full_np)
    first_input_token_idx_in_full = -1
    last_input_token_idx_in_full = -1
    first_output_token_idx_in_full = -1
    last_output_token_idx_in_full = -1

    if input_seq_char_start_in_full != -1 and input_seq_char_end_in_full != -1:
        for i, (start_char, end_char) in enumerate(offsets_full_np):
            # Skip padding offsets
            if start_char == 0 and end_char == 0 and not (len(offsets_full_np) == 1 and text_for_offsets_reference):
                continue
            overlap_start = max(input_seq_char_start_in_full, start_char)
            overlap_end = min(input_seq_char_end_in_full, end_char)
            if overlap_start < overlap_end:
                if first_input_token_idx_in_full == -1:
                    first_input_token_idx_in_full = i
                last_input_token_idx_in_full = i

        if first_input_token_idx_in_full == -1 or last_input_token_idx_in_full < first_input_token_idx_in_full:
            logger.warning(f"Sample {sample_index} layer {layer_idx_str}: could not find valid input token span (based on GT input char span).")
            first_input_token_idx_in_full = -1
            last_input_token_idx_in_full = -1

    if target_gt_no_marker_str and output_seq_char_start_in_full != -1 and output_seq_no_marker_char_end_in_full != -1:
        for i, (start_char, end_char) in enumerate(offsets_full_np):
            # Skip padding offsets
            if start_char == 0 and end_char == 0 and not (len(offsets_full_np) == 1 and text_for_offsets_reference):
                continue
            min_search_idx_for_output = (last_input_token_idx_in_full + 1) if last_input_token_idx_in_full != -1 else 0
            if i < min_search_idx_for_output:
                continue

            overlap_start = max(output_seq_char_start_in_full, start_char)
            overlap_end = min(output_seq_no_marker_char_end_in_full, end_char)
            if overlap_start < overlap_end:
                if first_output_token_idx_in_full == -1:
                    first_output_token_idx_in_full = i
                last_output_token_idx_in_full = i

        if first_output_token_idx_in_full == -1 or last_output_token_idx_in_full < first_output_token_idx_in_full:
            logger.warning(f"Sample {sample_index} layer {layer_idx_str}: could not find valid output token span (based on GT target char span).")
            first_output_token_idx_in_full = -1
            last_output_token_idx_in_full = -1

    logger.debug(f"  Input token absolute index span: [{first_input_token_idx_in_full}, {last_input_token_idx_in_full}]")
    logger.debug(f"  Output token absolute index span: [{first_output_token_idx_in_full}, {last_output_token_idx_in_full}]")

    # 4. Slice IDs and offsets
    if first_input_token_idx_in_full != -1:
        input_segment_token_ids = ids_full_np[first_input_token_idx_in_full:last_input_token_idx_in_full + 1]
        input_segment_token_offsets_abs = offsets_full_np[first_input_token_idx_in_full:last_input_token_idx_in_full + 1]
        input_segment_token_offsets_rel = input_segment_token_offsets_abs - input_seq_char_start_in_full
        input_segment_token_offsets_rel = np.maximum(input_segment_token_offsets_rel, 0)
    else:
        input_segment_token_ids = np.array([], dtype=ids_full_np.dtype)
        input_segment_token_offsets_rel = np.array([], dtype=offsets_full_np.dtype).reshape(0, 2)

    if first_output_token_idx_in_full != -1:
        output_segment_token_ids = ids_full_np[first_output_token_idx_in_full:last_output_token_idx_in_full + 1]
        output_segment_token_offsets_abs = offsets_full_np[first_output_token_idx_in_full:last_output_token_idx_in_full + 1]
        output_segment_token_offsets_rel = output_segment_token_offsets_abs - output_seq_char_start_in_full
        output_segment_token_offsets_rel = np.maximum(output_segment_token_offsets_rel, 0)
    else:
        output_segment_token_ids = np.array([], dtype=ids_full_np.dtype)
        output_segment_token_offsets_rel = np.array([], dtype=offsets_full_np.dtype).reshape(0, 2)

    # 5. Extract word-token mappings with helper
    input_gt_words_aligned, input_aligned_word_to_token_indices, _ = get_word_token_spans_from_offsets(
        input_segment_token_ids,
        input_segment_token_offsets_rel,
        input_seq_gt_str,
        tokenizer,
        "input", sample_index, layer_idx_str
    )

    if target_gt_no_marker_str and first_output_token_idx_in_full != -1:
        output_gt_words_aligned, output_aligned_word_to_token_indices, _ = get_word_token_spans_from_offsets(
            output_segment_token_ids,
            output_segment_token_offsets_rel,
            target_gt_no_marker_str,
            tokenizer,
            "output", sample_index, layer_idx_str
        )
    else:
        output_gt_words_aligned, output_aligned_word_to_token_indices = [], defaultdict(list)

    # 6. Collect states
    input_states_by_word_all_tokens: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    output_states_by_word_all_tokens: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    input_last_token_state_by_word: Dict[int, np.ndarray] = {}
    output_last_token_state_by_word: Dict[int, np.ndarray] = {}

    all_input_token_states: List[np.ndarray] = []
    all_input_token_word_indices: List[int] = []

    all_output_token_states: List[np.ndarray] = []
    all_output_token_word_indices: List[int] = []

    if first_input_token_idx_in_full != -1:
        for word_idx, token_indices_in_segment in input_aligned_word_to_token_indices.items():
            if not token_indices_in_segment:
                continue
            current_word_all_states_np = []
            for seg_tok_idx in token_indices_in_segment:
                abs_token_idx = first_input_token_idx_in_full + seg_tok_idx
                if 0 <= abs_token_idx < T_state_actual:
                    state_vec = states_np[abs_token_idx]
                    current_word_all_states_np.append(state_vec)
                    all_input_token_states.append(state_vec)
                    all_input_token_word_indices.append(word_idx)
            if current_word_all_states_np:
                input_states_by_word_all_tokens[word_idx] = current_word_all_states_np
                input_last_token_state_by_word[word_idx] = current_word_all_states_np[-1]

    if first_output_token_idx_in_full != -1:
        for word_idx, token_indices_in_segment in output_aligned_word_to_token_indices.items():
            if not token_indices_in_segment:
                continue
            current_word_all_states_np = []
            for seg_tok_idx in token_indices_in_segment:
                abs_token_idx = first_output_token_idx_in_full + seg_tok_idx
                if 0 <= abs_token_idx < T_state_actual:
                    state_vec = states_np[abs_token_idx]
                    current_word_all_states_np.append(state_vec)
                    all_output_token_states.append(state_vec)
                    all_output_token_word_indices.append(word_idx)
            if current_word_all_states_np:
                output_states_by_word_all_tokens[word_idx] = current_word_all_states_np
                output_last_token_state_by_word[word_idx] = current_word_all_states_np[-1]

    # 7. Final validation
    expected_num_input_words = len(input_gt_words_aligned)
    num_input_words_found = len(input_last_token_state_by_word)

    expected_num_output_words = len(output_gt_words_aligned)
    num_output_words_found = len(output_last_token_state_by_word)

    input_ratio = num_input_words_found / expected_num_input_words if expected_num_input_words > 0 else 1.0
    if expected_num_output_words == 0:
        output_ratio = 1.0 if num_output_words_found == 0 else 0.0
    else:
        output_ratio = num_output_words_found / expected_num_output_words

    strict_mode = getattr(config, 'WORD_EXTRACTION_STRICT', False)
    min_ratio = getattr(config, 'MIN_WORD_RATIO', 0.8)

    valid_sample = True
    if strict_mode:
        is_input_ok = (input_ratio >= 1.0)
        is_output_ok = ((expected_num_output_words == 0 and num_output_words_found == 0)
                        or (expected_num_output_words > 0 and output_ratio >= 1.0))
        if not (is_input_ok and is_output_ok):
            valid_sample = False
    else:
        is_input_ok = (input_ratio >= min_ratio)
        is_output_ok = ((expected_num_output_words == 0 and num_output_words_found == 0)
                        or (expected_num_output_words > 0 and output_ratio >= min_ratio))
        if not (is_input_ok and is_output_ok):
            valid_sample = False

    if not valid_sample:
        logger.warning(
            f"Sample {sample_index} layer {layer_idx_str}: final validation failed (v6). Strict mode: {strict_mode}. "
            f"Input words expected/found: {expected_num_input_words}/{num_input_words_found} (ratio: {input_ratio:.2f}). "
            f"Output words expected/found: {expected_num_output_words}/{num_output_words_found} (ratio: {output_ratio:.2f}). "
            f"Min ratio: {min_ratio if not strict_mode else 1.0}. Skipping sample."
        )
        return None

    logger.debug(
        f"  Extraction successful (sample {sample_index} layer {layer_idx_str}, v6): "
        f"Input words (expected/found): {expected_num_input_words}/{num_input_words_found}, "
        f"Output words (expected/found): {expected_num_output_words}/{num_output_words_found}. "
        f"Total input tokens with state: {len(all_input_token_states)}, "
        f"Total output tokens with state: {len(all_output_token_states)}."
    )

    return {
        "sample_index": sample_index,
        "input_states_by_word": dict(input_states_by_word_all_tokens),
        "output_states_by_word": dict(output_states_by_word_all_tokens),
        "input_last_token_state_by_word": input_last_token_state_by_word,
        "output_last_token_state_by_word": output_last_token_state_by_word,
        "input_words": input_gt_words_aligned,
        "output_words": output_gt_words_aligned,
        "input_token_word_indices": all_input_token_word_indices,
        "output_token_word_indices": all_output_token_word_indices,
        "input_token_states": all_input_token_states,
        "output_token_states": all_output_token_states
    }

def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "classification",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]: # Return type includes potential error message

    if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
        logger.warning("  Linear probe input data is empty.")
        return {"error": "input data is empty", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}
    if X.shape[0] != y.shape[0]:
        logger.warning(f"  Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) do not match.")
        return {"error": "number of samples in X and y do not match", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

    logger.info(f"  Training probe ({task_type}, {X.shape[0]} samples)...")
    results = {}
    try:
        # Data standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if task_type == "classification":
            # Check number of classes and samples per class for stratification
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            unique_labels, counts = np.unique(y_encoded, return_counts=True)
            n_classes = len(unique_labels)

            if n_classes < 2:
                logger.warning(f"  Only {n_classes} class(es) present; cannot perform classification cross-validation.")
                return {"error": "fewer than 2 classes", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

            min_samples_per_class = np.min(counts) if counts.size > 0 else 0
            # Ensure n_splits is at least 2 and no more than the minimum samples per class
            actual_n_splits = max(2, min(n_splits, min_samples_per_class))

            if actual_n_splits < 2:
                logger.warning(f"  Minimum class sample count ({min_samples_per_class}) too small (<2); cannot perform CV.")
                # Optionally, train on all data and report accuracy, but CV is preferred
                try:
                    model_single = LogisticRegression(solver='liblinear', random_state=random_state, max_iter=1000)
                    model_single.fit(X_scaled, y_encoded)
                    score_single = model_single.score(X_scaled, y_encoded)
                    logger.warning(f"  Accuracy trained on all data (no CV): {score_single:.4f}")
                    return {"error": "insufficient samples for stratified CV", "mean_score": score_single, "std_score": 0.0, "scores_per_fold": [score_single]}
                except Exception as e_single_fit:
                    logger.error(f"  Failed to train probe on all data: {e_single_fit}")
                    return {"error": "insufficient samples and training failed", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

            if actual_n_splits != n_splits:
                logger.info(f"  Requested n_splits ({n_splits}) > min samples per class ({min_samples_per_class}), using {actual_n_splits} folds instead.")

            model = LogisticRegression(
                solver='liblinear',
                random_state=random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            cv = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=random_state)
            scoring = "accuracy"
            y_target = y_encoded  # Use encoded labels for CV

        elif task_type == "regression":
            model = LinearRegression()
            # For regression, ensure n_splits is valid
            actual_n_splits = max(2, min(n_splits, X.shape[0]))
            if actual_n_splits < 2:
                logger.warning(f"  Sample size ({X.shape[0]}) too small (<2); cannot perform CV.")
                # Optionally train on all data
                try:
                    model_single = LinearRegression()
                    model_single.fit(X_scaled, y)
                    score_single = model_single.score(X_scaled, y)
                    logger.warning(f"  R² score trained on all data (no CV): {score_single:.4f}")
                    return {"error": "insufficient samples for CV", "mean_score": score_single, "std_score": 0.0, "scores_per_fold": [score_single]}
                except Exception as e_single_fit:
                    logger.error(f"  Failed to train regression probe on all data: {e_single_fit}")
                    return {"error": "insufficient samples and training failed", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

            if actual_n_splits != n_splits:
                logger.info(f"  Requested n_splits ({n_splits}) > sample size ({X.shape[0]}), using {actual_n_splits} folds instead.")

            cv = KFold(n_splits=actual_n_splits, shuffle=True, random_state=random_state)
            scoring = "r2"  # R-squared
            y_target = y     # Use original labels for regression

        else:
            logger.error(f"  Error: Unknown task_type '{task_type}'")
            return {"error": "unknown task type", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

        # Perform cross-validation
        try:
            scores = cross_val_score(
                model, X_scaled, y_target,
                cv=cv, scoring=scoring,
                n_jobs=-1, error_score='raise'
            )
        except ValueError as e_cv:
            logger.error(f"  Error during cross-validation (possible label distribution issue): {e_cv}")
            if task_type == "classification":
                logger.error(f"    Label distribution: {dict(zip(unique_labels, counts))}")
            return {"error": f"cross-validation failed: {e_cv}", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}
        except Exception as e_cv_other:
            logger.error(f"  Unexpected error during cross-validation: {e_cv_other}", exc_info=True)
            return {"error": f"cross-validation unexpected failure: {e_cv_other}", "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

        results["mean_score"] = float(np.mean(scores))
        results["std_score"] = float(np.std(scores))
        results["scores_per_fold"] = [float(s) for s in scores]
        if task_type == "classification":
            results["mean_accuracy"] = results["mean_score"]

        logger.info(f"    Probe training complete. Mean {scoring}: {results['mean_score']:.4f}")

    except Exception as e:
        logger.error(f"  Linear probe training/evaluation failed: {e}", exc_info=True)
        results = {"error": str(e), "mean_score": np.nan, "std_score": np.nan, "scores_per_fold": []}

    return results

def plot_pca_results(
    pca_data: np.ndarray,
    title: str,
    save_path: str,
    hue_labels: List[str],
    style_labels: Optional[List[str]] = None,
    hue_name: str = "Category",
    style_name: Optional[str] = "Style",
    figsize: Tuple[int, int] = (12, 10) 
):
    if pca_data is None or pca_data.shape[1] < 2:
        logger.warning(f"PCA data has fewer than two dimensions ({pca_data.shape if pca_data is not None else 'None'}); cannot plot scatter: {title}")
        return

    logger.info(f"  Plotting PCA scatter: {title}")
    plt.figure(figsize=figsize)

    data_dict = {
        'PC1': pca_data[:, 0],
        'PC2': pca_data[:, 1],
        hue_name: hue_labels,
    }
    if style_labels is not None:
        if len(style_labels) != len(hue_labels):
            logger.warning(f"Number of style labels ({len(style_labels)}) does not match number of hue labels ({len(hue_labels)}); ignoring style.")
            style_labels = None  # Reset to avoid error
        else:
            data_dict[style_name] = style_labels

    df_pca = pd.DataFrame(data_dict)

    plot_params = {
        "data": df_pca,
        "x": "PC1",
        "y": "PC2",
        "hue": hue_name,
        "s": 50, 
        "alpha": 0.7, 
    }
    if style_labels is not None:
        plot_params["style"] = style_name
        unique_styles = df_pca[style_name].nunique()
        if unique_styles <= 15:
            plot_params["markers"] = True
        else:
            logger.warning(f"Too many style categories ({unique_styles}); not using distinct marker styles.")
            del plot_params["style"]

    sns.scatterplot(**plot_params)

    plt.title(title, fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)  # higher DPI
        logger.info(f"  PCA scatter saved to: {save_path}")
    except Exception as e:
        logger.error(f"  Failed to save PCA scatter '{save_path}': {e}")
    plt.close() 
    
# Main PCA runner function
def run_pca_analysis(
    hidden_states_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    items_data: List[Dict[str, Any]],
    full_texts_for_offsets: List[str], 
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_indices_to_analyze: List[int],
    seq_len_words: int,
    figures_dir: str,
    pattern_type: str = config.PATTERN_TYPE,
    strict_extraction: bool = config.WORD_EXTRACTION_STRICT,
    min_word_ratio: float = config.MIN_WORD_RATIO,
    use_last_token_state: bool = True
):
    print(f"\nA. Starting PCA analysis (using {'last token' if use_last_token_state else 'average word'} states)")
    if not hidden_states_data:
        logger.warning("No hidden state data available; skipping PCA analysis.")
        return

    num_layers_total = model.config.num_hidden_layers + 1
    actual_indices_map = {}
    valid_layer_indices = []
    for req_idx in layer_indices_to_analyze:
        actual_idx = req_idx if req_idx >= 0 else num_layers_total + req_idx
        if 0 <= actual_idx < num_layers_total and actual_idx in hidden_states_data:
            actual_indices_map[req_idx] = actual_idx
            if actual_idx not in valid_layer_indices:
                valid_layer_indices.append(actual_idx)
    valid_layer_indices.sort()
    if not valid_layer_indices:
        logger.warning("  No valid layer indices available for PCA analysis.")
        return

    for actual_idx in valid_layer_indices:
        req_idx = [k for k, v in actual_indices_map.items() if v == actual_idx][0]
        req_idx_str = str(req_idx)
        logger.info(f"\n  Performing PCA on layer: {req_idx_str} (actual index {actual_idx})")

        layer_data = hidden_states_data[actual_idx]
        num_items = len(items_data)  # number of items to analyze
        if len(layer_data) != num_items:
            logger.error(f"Layer {req_idx_str}: number of state samples ({len(layer_data)}) does not match items_data ({num_items}); skipping this layer.")
            continue

        all_input_states_for_pca = []
        all_output_states_for_pca = []
        all_input_word_labels = []
        all_output_word_labels = []
        all_input_word_indices = []
        all_output_word_indices = []
        successful_samples_count = 0
        failed_extraction_count = 0


        for sample_idx in range(num_items):
            states_tensor, ids_tensor, offsets_tensor = layer_data[sample_idx]
            item = items_data[sample_idx]
            current_full_text_for_offsets = full_texts_for_offsets[sample_idx]

            target_gt_no_marker_str = item.get('target_no_marker', "").strip()

            structured_result = _extract_word_states_from_pretokenized(
                states_tensor, 
                ids_tensor, 
                offsets_tensor, 
                item, 
                tokenizer, 
                sample_idx, 
                req_idx_str,
                current_full_text_for_offsets
            )

            if structured_result is None:
                failed_extraction_count += 1
                continue

            input_states_by_word = structured_result['input_states_by_word']
            output_states_by_word = structured_result['output_states_by_word']
            expected_input_len = len(structured_result['input_words'])
            expected_output_len = len(structured_result['output_words'])
            
            input_ratio = len(input_states_by_word) / expected_input_len if expected_input_len > 0 else 1.0
            
            if not target_gt_no_marker_str or expected_output_len == 0: 
                output_ratio = 1.0 if len(output_states_by_word) == 0 else 0.0
            else:
                output_ratio = len(output_states_by_word) / expected_output_len


            if (strict_extraction and (input_ratio < 1.0 or (expected_output_len > 0 and output_ratio < 1.0))) or \
               (not strict_extraction and (input_ratio < min_word_ratio or (expected_output_len > 0 and output_ratio < min_word_ratio))):
                failed_extraction_count += 1
                continue
            
            successful_samples_count += 1

            input_states_source = structured_result['input_last_token_state_by_word'] if use_last_token_state else structured_result['input_states_by_word']
            output_states_source = structured_result['output_last_token_state_by_word'] if use_last_token_state else structured_result['output_states_by_word']
            input_words = structured_result['input_words']
            output_words = structured_result['output_words']

            for word_idx_iter in range(config.SEQUENCE_LENGTH): 
                if word_idx_iter < len(input_words):
                    if word_idx_iter in input_states_source:
                        state_data = input_states_source[word_idx_iter]
                        if use_last_token_state:
                            state_vec = state_data
                        else:
                            state_vec = np.mean(np.stack(state_data), axis=0) if state_data else None 

                        if state_vec is not None and state_vec.size > 0:
                            all_input_states_for_pca.append(state_vec)
                            all_input_word_labels.append(input_words[word_idx_iter])
                            all_input_word_indices.append(word_idx_iter)

                if word_idx_iter < len(output_words):
                    if word_idx_iter in output_states_source:
                        state_data = output_states_source[word_idx_iter]

                        if use_last_token_state:
                            state_vec = state_data
                        else: 
                            state_vec = np.mean(np.stack(state_data), axis=0) if state_data else None

                        if state_vec is not None and state_vec.size > 0:
                            all_output_states_for_pca.append(state_vec)
                            all_output_word_labels.append(output_words[word_idx_iter])
                            all_output_word_indices.append(word_idx_iter)
        if successful_samples_count == 0:
            logger.warning(f"    Layer {req_idx_str}: no valid states extracted for any samples (failed {failed_extraction_count}); skipping PCA for this layer.")
            continue

        logger.info(f"    Layer {req_idx_str}: prepared {'last token' if use_last_token_state else 'average word'} states for PCA for {successful_samples_count}/{num_items} samples (skipped {failed_extraction_count}).")
        
        if not all_input_states_for_pca or not all_output_states_for_pca:
            logger.warning(f"    Layer {req_idx_str}: collected input or output states are empty; skipping PCA.")
            continue

        try:
            input_states_np = np.stack(all_input_states_for_pca, axis=0)
            output_states_np = np.stack(all_output_states_for_pca, axis=0)
            all_states_for_pca = np.concatenate([input_states_np, output_states_np], axis=0)
        except ValueError as e_stack:
            logger.error(f"    Layer {req_idx_str}: failed to stack states: {e_stack}; skipping PCA.")
            continue

        all_word_labels = all_input_word_labels + all_output_word_labels
        all_pos_types = ['Input Word'] * len(all_input_states_for_pca) + ['Output Word'] * len(all_output_states_for_pca)
        all_word_indices = all_input_word_indices + all_output_word_indices
        logger.info(f"    Number of states prepared for PCA: {all_states_for_pca.shape[0]}, dimensionality: {all_states_for_pca.shape[1]}")

        pca_result, pca_model = apply_pca(all_states_for_pca)

        if pca_result is not None:
            state_type_suffix = "lastToken" if use_last_token_state else "avgWord"
            hue_name = "Word"
            style_name_pos = "Word Pos"
            title1 = f"PCA Layer {req_idx_str} ({state_type_suffix} State, Color: {hue_name}, Style: Pos Type)"
            save1 = os.path.join(figures_dir, f"pca_layer_{req_idx_str}_{state_type_suffix}_word_posType.png")
            plot_pca_results(
                pca_data=pca_result, title=title1, save_path=save1,
                hue_labels=all_word_labels, style_labels=all_pos_types,
                hue_name=hue_name, style_name="Position Type"
            )
            unique_indices = sorted(set(all_word_indices))
            if 0 < len(unique_indices) <= 15:
                title2 = f"PCA Layer {req_idx_str} ({state_type_suffix} State, Color: Pos Type, Style: {style_name_pos})"
                save2 = os.path.join(figures_dir, f"pca_layer_{req_idx_str}_{state_type_suffix}_posType_wordPos.png")
                style_labels_pos = [f"Pos{p}" for p in all_word_indices]
                plot_pca_results(
                    pca_data=pca_result, title=title2, save_path=save2,
                    hue_labels=all_pos_types, style_labels=style_labels_pos,
                    hue_name="Position Type", style_name=style_name_pos
                )
            else:
                logger.info(f"    Too many word position categories ({len(unique_indices)}); skipping PCA plot with position-based markers.")


def run_linear_probe_analysis(
    hidden_states_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    items_data: List[Dict[str, Any]],
    full_texts_for_offsets: List[str],
    tokenizer: AutoTokenizer,
    model: Any,
    layer_indices_to_analyze: List[int],
    seq_len_words: int,
    task_unit: str,
    results_dir: str,
    pattern_type: str = config.PATTERN_TYPE,
    strict_extraction: bool = config.WORD_EXTRACTION_STRICT,
    min_word_ratio: float = config.MIN_WORD_RATIO
):
    print("\nB. Starting linear probe analysis (based on token states)")
    if not hidden_states_data:
        logger.warning("No hidden state data available; skipping linear probe analysis.")
        return

    probe_results = {}
    label_unit = "word"

    num_layers_total = model.config.num_hidden_layers + 1
    actual_indices_map = {}
    valid_layer_indices = []
    for req_idx in layer_indices_to_analyze:
        actual_idx = req_idx if req_idx >= 0 else num_layers_total + req_idx
        if 0 <= actual_idx < num_layers_total and actual_idx in hidden_states_data:
            actual_indices_map[req_idx] = actual_idx
            if actual_idx not in valid_layer_indices:
                valid_layer_indices.append(actual_idx)
        else:
            logger.warning(f"Requested layer index {req_idx} (mapped to {actual_idx}) is invalid or not extracted; ignoring.")
    valid_layer_indices.sort()
    if not valid_layer_indices:
        logger.warning("  No valid layer indices available for linear probe analysis.")
        return

    for actual_idx in valid_layer_indices:
        req_idx = next(k for k, v in actual_indices_map.items() if v == actual_idx)
        req_idx_str = str(req_idx)
        logger.info(f"\n  Analyzing probe layer: {req_idx_str} (actual index {actual_idx})")

        layer_data = hidden_states_data[actual_idx]
        num_items = len(items_data)
        if len(layer_data) != num_items:
            logger.error(f"Layer {req_idx_str}: number of state samples ({len(layer_data)}) does not match items_data ({num_items}); skipping this layer.")
            continue

        layer_all_input_token_states = []
        layer_all_output_token_states = []
        layer_all_input_token_labels = []
        layer_all_output_token_labels = []
        layer_all_task3_target_labels = []
        successful_samples_count = 0
        task3_possible_for_layer = (pattern_type in ["copy_word", "reverse_word"])
        failed_extraction_count = 0

        for sample_idx in range(num_items):
            if len(layer_data[sample_idx]) != 3:
                logger.error(f"Sample {sample_idx} layer {req_idx_str}: data format error, expected tuple (states, ids, offsets); skipping.")
                failed_extraction_count += 1
                continue
            states_tensor, ids_tensor, offsets_tensor = layer_data[sample_idx]
            item = items_data[sample_idx]
            current_full_text_for_offsets = full_texts_for_offsets[sample_idx]

            structured_result = _extract_word_states_from_pretokenized(
                states_tensor,
                ids_tensor,
                offsets_tensor,
                item,
                tokenizer,
                sample_idx,
                req_idx_str,
                current_full_text_for_offsets
            )
            if structured_result is None:
                failed_extraction_count += 1
                continue

            input_states_by_word = structured_result['input_states_by_word']
            output_states_by_word = structured_result['output_states_by_word']

            expected_input_len = len(structured_result['input_words'])
            expected_output_len = len(structured_result['output_words'])

            input_ratio = len(input_states_by_word) / expected_input_len if expected_input_len > 0 else 1.0
            output_ratio = len(output_states_by_word) / expected_output_len if expected_output_len > 0 else 1.0

            if strict_extraction:
                if input_ratio < 1.0 or (expected_output_len > 0 and output_ratio < 1.0):
                    failed_extraction_count += 1
                    continue
            else:
                if input_ratio < min_word_ratio or (expected_output_len > 0 and output_ratio < min_word_ratio):
                    failed_extraction_count += 1
                    continue
                elif input_ratio < 1.0 or (expected_output_len > 0 and output_ratio < 1.0):
                    logger.info(f"Sample {sample_idx} layer {req_idx_str}: partial word extraction success (input: {input_ratio:.2f}, output: {output_ratio:.2f}).")

            if not structured_result['input_token_states'] or \
               (expected_output_len > 0 and not structured_result['output_token_states']):
                logger.warning(f"Sample {sample_idx} layer {req_idx_str}: tokens states not extracted despite words; skipping sample in probe aggregation.")
                failed_extraction_count += 1
                continue

            successful_samples_count += 1

            input_token_word_indices = structured_result['input_token_word_indices']
            output_token_word_indices = structured_result['output_token_word_indices']
            input_words = structured_result['input_words']
            output_words = structured_result['output_words']
            L = len(input_words) 

            layer_all_input_token_states.extend(structured_result['input_token_states'])
            layer_all_output_token_states.extend(structured_result['output_token_states'])

            for word_idx in input_token_word_indices:
                layer_all_input_token_labels.append(input_words[word_idx] if 0 <= word_idx < L else "[INVALID_LABEL]")
            for word_idx in output_token_word_indices:
                layer_all_output_token_labels.append(output_words[word_idx] if 0 <= word_idx < len(output_words) else "[INVALID_LABEL]")

            if task3_possible_for_layer:
                for token_output_word_idx in output_token_word_indices: 
                    target_input_idx = -1
                    if pattern_type == "copy_word":
                        target_input_idx = token_output_word_idx
                    elif pattern_type == "reverse_word":
                        target_input_idx = L - 1 - token_output_word_idx
                    
                    if 0 <= target_input_idx < L:
                        layer_all_task3_target_labels.append(input_words[target_input_idx])
                    else:
                        layer_all_task3_target_labels.append("[INVALID_TASK3_LABEL]")
                        
        if successful_samples_count == 0:
            logger.warning(f"    Layer {req_idx_str}: no valid token states extracted for any samples (failed {failed_extraction_count}); skipping probe.")
            continue

        logger.info(f"    Layer {req_idx_str}: successfully extracted states for probe for {successful_samples_count}/{num_items} samples (skipped {failed_extraction_count}).")

        try:
            input_states_flat = np.stack(layer_all_input_token_states, axis=0) if layer_all_input_token_states else np.array([])
            output_states_flat = np.stack(layer_all_output_token_states, axis=0) if layer_all_output_token_states else np.array([])

            # Verify lengths BEFORE attempting to create arrays for labels
            if len(layer_all_input_token_labels) != input_states_flat.shape[0]:
                logger.error(f"Layer {req_idx_str}: aggregation error! Input state count ({input_states_flat.shape[0]}) != Input label count ({len(layer_all_input_token_labels)}).")
                continue
            if len(layer_all_output_token_labels) != output_states_flat.shape[0]:
                logger.error(f"Layer {req_idx_str}: aggregation error! Output state count ({output_states_flat.shape[0]}) != Output label count ({len(layer_all_output_token_labels)}).")
                continue
            if task3_possible_for_layer and len(layer_all_task3_target_labels) != output_states_flat.shape[0]:
                logger.error(f"Layer {req_idx_str}: aggregation error! Task3 label count ({len(layer_all_task3_target_labels)}) != Output state count ({output_states_flat.shape[0]}).")
                task3_possible_for_layer = False  # Disable Task 3 if lengths mismatch

            targets_input_label = np.array(layer_all_input_token_labels)
            targets_output_label = np.array(layer_all_output_token_labels)
            targets_probe3 = np.array(layer_all_task3_target_labels) if task3_possible_for_layer else None

        except ValueError as e_stack:
            logger.error(f"    Layer {req_idx_str}: failed to stack or verify token states/labels: {e_stack}; skipping probe.")
            continue
        except Exception as e_prep:
            logger.error(f"    Layer {req_idx_str}: error preparing probe data: {e_prep}; skipping probe.")
            continue

        # Run Probes
        layer_probe_results = {}

        # Task 1: Input Word Label from Input Token State
        if input_states_flat.size > 0:
            logger.info(f"    Task 1: probe input {label_unit} (from input token states)")
            probe_res_in_label = train_linear_probe(input_states_flat, targets_input_label, task_type="classification")
            layer_probe_results[f"probe_input_{label_unit}_from_input_states"] = probe_res_in_label
        else:
            logger.warning("    Skipping Task 1 (no valid input states).")

        # Task 2: Output Word Label from Output Token State
        if output_states_flat.size > 0:
            logger.info(f"    Task 2: probe output {label_unit} (from output token states)")
            probe_res_out_label = train_linear_probe(output_states_flat, targets_output_label, task_type="classification")
            layer_probe_results[f"probe_output_{label_unit}_from_output_states"] = probe_res_out_label
        else:
            logger.warning("    Skipping Task 2 (no valid output states).")

        # Task 3: Corresponding Input Word Label from Output Token State
        if task3_possible_for_layer and targets_probe3 is not None and output_states_flat.size > 0:
            task3_desc = "input" if pattern_type.startswith("copy") else "reversed input"
            logger.info(f"    Task 3: probe {task3_desc} {label_unit} (from output token states)")
            probe_res_task3 = train_linear_probe(output_states_flat, targets_probe3, task_type="classification")
            result_key_task3 = f"probe_{'input' if pattern_type.startswith('copy') else 'reversed_input'}_{label_unit}_from_output_states"
            layer_probe_results[result_key_task3] = probe_res_task3
        else:
            logger.info(f"    Skipping Task 3 (pattern type {pattern_type} not supported or labels unavailable).")

        # Task 4: Position Type (Input vs Output Token)
        logger.info("    Task 4: probe position type (Input Token vs Output Token)")
        try:
            if input_states_flat.size > 0 and output_states_flat.size > 0:
                all_token_states_pos = np.concatenate([input_states_flat, output_states_flat], axis=0)
                targets_pos_type = np.array(['Input Token'] * len(input_states_flat) + ['Output Token'] * len(output_states_flat))
                probe_res_pos = train_linear_probe(all_token_states_pos, targets_pos_type, task_type="classification")
                layer_probe_results["probe_position_type"] = probe_res_pos
            else:
                logger.warning(f"    Task 4 skipping due to empty input ({input_states_flat.shape[0]}) or output ({output_states_flat.shape[0]}) states.")
                layer_probe_results["probe_position_type"] = {"error": "input or output states empty", "mean_score": np.nan}
        except ValueError as e_concat:
            logger.warning(f"    Task 4 state concatenation failed: {e_concat}; skipping Task 4.")
            layer_probe_results["probe_position_type"] = {"error": "state concatenation failed", "mean_score": np.nan}

        probe_results[req_idx_str] = layer_probe_results

    # Save Probe Results
    if probe_results:
        fname = f"linear_probe_results_L{config.SEQUENCE_LENGTH}{task_unit}.json"
        utils.save_results(probe_results, fname, results_dir)
    else:
        logger.warning("Linear probe analysis produced no results.")

def apply_pca(data: np.ndarray, n_components: int = None, explained_variance_threshold: float = None) -> Tuple[np.ndarray, PCA]:
    if n_components is None:
        full_pca = PCA().fit(data)
        explained_variance_ratio_cumsum = np.cumsum(full_pca.explained_variance_ratio_)
        
        if explained_variance_threshold is not None:
            n_components = np.argmax(explained_variance_ratio_cumsum >= explained_variance_threshold) + 1
            logger.info(f"Automatically selected {n_components} principal components (cumulative variance ratio: {explained_variance_ratio_cumsum[n_components-1]:.4f})")
        else:
            n_components = 2
            logger.info(f"Using default {n_components} principal components (cumulative variance ratio: {explained_variance_ratio_cumsum[n_components-1]:.4f})")

    pca = PCA(n_components=n_components)
    pca.fit(data)

    transformed_data = pca.transform(data)

    variance_explained = pca.explained_variance_ratio_
    logger.info(f"PCA variance explained: {variance_explained}")
    logger.info(f"Cumulative variance explained: {np.sum(variance_explained):.4f}")

    return transformed_data, pca


def compare_state_vectors(vec1: np.ndarray, vec2: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        # Compute cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        return 1 - cosine(vec1, vec2)  # note: 1 - cosine distance
    elif metric == "euclidean":
        # Compute Euclidean distance
        return np.linalg.norm(vec1 - vec2)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


# Main Cosine Similarity runner function
def run_cosine_similarity_analysis(
    hidden_states_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    items_data: List[Dict[str, Any]],
    full_texts_for_offsets: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    layer_indices_to_analyze: List[int],
    seq_len_words: int,
    task_unit: str,
    results_dir: str,
    figures_dir: str,
    pattern_type: str = config.PATTERN_TYPE,
    strict_extraction: bool = config.WORD_EXTRACTION_STRICT,
    min_word_ratio: float = config.MIN_WORD_RATIO,
    use_last_token_state: bool = True
):
    print(f"\nC. Starting cosine similarity analysis (using {'last token' if use_last_token_state else 'average word'} states)")
    if not hidden_states_data:
        logger.warning("No hidden state data available; skipping cosine similarity analysis.")
        return

    similarity_results = {}
    label_unit = "word"

    num_layers_total = model.config.num_hidden_layers + 1
    actual_indices_map = {}
    valid_layer_indices = []
    for req_idx in layer_indices_to_analyze:
        actual_idx = req_idx if req_idx >= 0 else num_layers_total + req_idx
        if 0 <= actual_idx < num_layers_total and actual_idx in hidden_states_data:
            actual_indices_map[req_idx] = actual_idx
            if actual_idx not in valid_layer_indices:
                valid_layer_indices.append(actual_idx)
    valid_layer_indices.sort()
    if not valid_layer_indices:
        logger.warning("  No valid layer indices available for cosine similarity analysis.")
        return

    for actual_idx in valid_layer_indices:
        req_idx = next(k for k, v in actual_indices_map.items() if v == actual_idx)
        req_idx_str = str(req_idx)
        logger.info(f"\n  Analyzing similarity for layer: {req_idx_str} (actual index {actual_idx})")

        layer_data = hidden_states_data[actual_idx]
        num_items = len(items_data)
        if len(layer_data) != num_items:
            logger.error(f"Layer {req_idx_str}: number of state samples ({len(layer_data)}) does not match items_data ({num_items}); skipping this layer.")
            continue

        similarities_by_pos: List[List[float]] = [[] for _ in range(config.SEQUENCE_LENGTH)]
        similarities_control: List[List[float]] = [[] for _ in range(config.SEQUENCE_LENGTH)]
        successful_samples_count = 0
        failed_extraction_count = 0

        for sample_idx in range(num_items):
            states_tensor, ids_tensor, offsets_tensor = layer_data[sample_idx]
            item = items_data[sample_idx]
            current_full_text_for_offsets = full_texts_for_offsets[sample_idx]

            structured_result = _extract_word_states_from_pretokenized(
                states_tensor,
                ids_tensor,
                offsets_tensor,
                item,
                tokenizer,
                sample_idx,
                req_idx_str,
                current_full_text_for_offsets
            )

            if structured_result is None:
                failed_extraction_count += 1
                continue

            input_states_by_word = structured_result['input_states_by_word']
            expected_input_len = len(structured_result['input_words'])
            expected_output_len = len(structured_result['output_words'])

            input_ratio = len(input_states_by_word) / expected_input_len if expected_input_len > 0 else 1.0

            if expected_output_len == 0:
                output_ratio = 1.0 if len(structured_result['output_states_by_word']) == 0 else 0.0
            else:
                output_ratio = len(structured_result['output_states_by_word']) / expected_output_len

            if (strict_extraction and (input_ratio < 1.0 or (expected_output_len > 0 and output_ratio < 1.0))) or \
               (not strict_extraction and (input_ratio < min_word_ratio or (expected_output_len > 0 and output_ratio < min_word_ratio))):
                failed_extraction_count += 1
                continue

            successful_samples_count += 1

            input_states_comp: Dict[int, np.ndarray] = {}
            output_states_comp: Dict[int, np.ndarray] = {}

            input_words = structured_result['input_words']
            output_words = structured_result['output_words']

            if use_last_token_state:
                input_states_source = structured_result['input_last_token_state_by_word']
                output_states_source = structured_result['output_last_token_state_by_word']

                # Use last token state directly
                for word_idx_iter in range(config.SEQUENCE_LENGTH):
                    if word_idx_iter in input_states_source and input_states_source[word_idx_iter] is not None:
                        input_states_comp[word_idx_iter] = input_states_source[word_idx_iter]
                    if word_idx_iter in output_states_source and output_states_source[word_idx_iter] is not None:
                        output_states_comp[word_idx_iter] = output_states_source[word_idx_iter]
            else:
                # Use average word states
                input_all_token_states_by_word = structured_result['input_states_by_word']
                output_all_token_states_by_word = structured_result['output_states_by_word']

                for word_idx_iter in range(config.SEQUENCE_LENGTH):
                    if word_idx_iter in input_all_token_states_by_word and input_all_token_states_by_word[word_idx_iter]:
                        avg_state = np.mean(np.stack(input_all_token_states_by_word[word_idx_iter]), axis=0)
                        if avg_state is not None and avg_state.size > 0:
                            input_states_comp[word_idx_iter] = avg_state

                    if word_idx_iter in output_all_token_states_by_word and output_all_token_states_by_word[word_idx_iter]:
                        avg_state = np.mean(np.stack(output_all_token_states_by_word[word_idx_iter]), axis=0)
                        if avg_state is not None and avg_state.size > 0:
                            output_states_comp[word_idx_iter] = avg_state

            L_seq = config.SEQUENCE_LENGTH

            for i in range(L_seq):
                if i not in input_states_comp:
                    logger.debug(f"    Sample {sample_idx} layer {req_idx_str}: state for input word index {i} not found or invalid.")
                    continue

                input_state_i = input_states_comp[i]
                if pattern_type == "copy_word":
                    output_word_idx_corr = i
                elif pattern_type == "reverse_word":
                    output_word_idx_corr = L_seq - 1 - i
                else:
                    output_word_idx_corr = -1

                if 0 <= output_word_idx_corr < L_seq and output_word_idx_corr in output_states_comp:
                    output_state_corr = output_states_comp[output_word_idx_corr]
                    sim = compare_state_vectors(input_state_i, output_state_corr, metric="cosine")
                    if not np.isnan(sim):
                        similarities_by_pos[i].append(sim)
                else:
                    logger.debug(f"    Sample {sample_idx} layer {req_idx_str}: state for output word index {output_word_idx_corr} (corresponding to input {i}) not found or invalid.")

                if L_seq > 1:
                    j = (i + 1) % L_seq
                    if j not in input_states_comp or i == j:
                        logger.debug(f"    Sample {sample_idx} layer {req_idx_str}: state for control word index {j} not found or is same as i.")
                        continue
                    if 0 <= i < len(input_words) and 0 <= j < len(input_words) and input_words[i] != input_words[j]:
                        input_state_j = input_states_comp[j]
                        sim_ctrl = compare_state_vectors(input_state_i, input_state_j, metric="cosine")
                        if not np.isnan(sim_ctrl):
                            similarities_control[i].append(sim_ctrl)

        if successful_samples_count == 0:
            logger.warning(f"    Layer {req_idx_str}: no valid state pairs computed for any samples (failed {failed_extraction_count}); skipping similarity for this layer.")
            continue
        logger.info(f"    Layer {req_idx_str}: successfully computed similarities for {successful_samples_count}/{num_items} samples (skipped {failed_extraction_count}).")

        avg_sim_by_pos = [np.nanmean(sims) if sims else np.nan for sims in similarities_by_pos]
        overall_avg_sim_correct = np.mean([s for pos_sims in similarities_by_pos for s in pos_sims if not np.isnan(s)]) if any(similarities_by_pos) else np.nan
        overall_avg_sim_diff = np.mean([s for pos_sims in similarities_control for s in pos_sims if not np.isnan(s)]) if any(similarities_control) else np.nan

        logger.info(f"    Average Cosine Similarity (Input[Word i] vs Output[corr. pos] - {'lastToken' if use_last_token_state else 'avgWord'} State):")
        for pos, avg_sim in enumerate(avg_sim_by_pos):
             logger.info(f"      Word Position {pos}: {avg_sim:.4f}")

        logger.info(f"    Overall Avg Similarity (Corresponding Words): {overall_avg_sim_correct:.4f}")
        logger.info(f"    Overall Avg Similarity (Different Input Words Control): {overall_avg_sim_diff:.4f}")

        state_type_suffix = "lastToken" if use_last_token_state else "avgWord"
        similarity_results[req_idx_str] = {
            "state_type_used": state_type_suffix,
            "avg_similarity_by_position": [float(s) if not np.isnan(s) else None for s in avg_sim_by_pos],
            f"overall_avg_similarity_correct_{state_type_suffix}": float(overall_avg_sim_correct) if not np.isnan(overall_avg_sim_correct) else None,
            f"overall_avg_similarity_diff_control_{state_type_suffix}": float(overall_avg_sim_diff) if not np.isnan(overall_avg_sim_diff) else None,
        }

        plt.figure(figsize=(8, 5))
        positions = list(range(config.SEQUENCE_LENGTH))

        sim_label = f'Avg Sim: Input[Word i] vs Output[corr. pos] ({state_type_suffix} State): {overall_avg_sim_correct:.3f}'
        ctrl_label = f'Avg Sim Control: Input[Word i] vs Input[Word j] ({state_type_suffix} State): {overall_avg_sim_diff:.3f}'

        valid_sim_pos = [(p, s) for p, s in zip(positions, avg_sim_by_pos) if not np.isnan(s)]
        if valid_sim_pos:
            plt.plot([p for p,s in valid_sim_pos], [s for p,s in valid_sim_pos], marker='o', linestyle='-', label=sim_label)
        
        if not np.isnan(overall_avg_sim_diff): 
             plt.axhline(overall_avg_sim_diff, color='r', linestyle='--', label=ctrl_label)
        
        plt.xlabel(f"Input Word Position (i)") 
        plt.ylabel("Average Cosine Similarity") 
        plt.title(f"Layer {req_idx_str}: Input[Word i] vs Output[corr. pos] Cosine Similarity ({state_type_suffix} State)") 

        plt.xticks(positions)

        all_plot_values = [s for p, s in valid_sim_pos]
        if not np.isnan(overall_avg_sim_diff):
            all_plot_values.append(overall_avg_sim_diff)
        
        if all_plot_values: 
            min_val_plot = np.nanmin(all_plot_values) 
            max_val_plot = np.nanmax(all_plot_values)
            y_lower_limit = min(0, min_val_plot - 0.1)
            y_upper_limit = max(1, max_val_plot + 0.1)
            plt.ylim(y_lower_limit, y_upper_limit)
        else: 
            plt.ylim(0, 1.05) 

        plt.legend(fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_path = os.path.join(figures_dir, f"cosine_similarity_layer_{req_idx_str}_{state_type_suffix}.png")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=120)
            logger.info(f"  Similarity plot saved: {save_path}")
        except Exception as e_save:
            logger.error(f"  Failed to save similarity plot: {e_save}")
        plt.close()

    # Save Aggregate Similarity Results 
    if similarity_results:
        state_type_used_suffix = "lastToken" if use_last_token_state else "avgWord"
        fname = f"cosine_similarity_results_L{config.SEQUENCE_LENGTH}Word_{state_type_used_suffix}.json"
        utils.save_results(similarity_results, fname, results_dir)
    else:
        logger.warning("Cosine similarity analysis did not generate any results to save.")

def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    layer_idx: int,
    head_idx: int,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    if attention_matrix is None or attention_matrix.ndim != 2:
        logger.warning(f"Attention matrix for layer {layer_idx} head {head_idx} is invalid; skipping plot.")
        return

    seq_q, seq_k = attention_matrix.shape[-2:]
    if not tokens:
        logger.warning(f"Missing tokens for layer {layer_idx} head {head_idx}; using indices as labels.")
        qlab = [str(i) for i in range(seq_q)]
        klab = [str(i) for i in range(seq_k)]
    else:
        clean = [
            ("".join(ch if ch.isprintable() else "?" for ch in t))
            .replace("Ġ", "_").replace(" ", "_")[:8]
            for t in tokens
        ]
        qlab = clean[:seq_q]
        klab = clean[:seq_k]
        # Pad with indices if tokens shorter than matrix dimensions
        if len(qlab) < seq_q:
            qlab.extend([str(i) for i in range(len(qlab), seq_q)])
        if len(klab) < seq_k:
            klab.extend([str(i) for i in range(len(klab), seq_k)])

    plt.figure(figsize=figsize)
    sns.heatmap(
        attention_matrix,
        xticklabels=klab,
        yticklabels=qlab,
        cmap="viridis",
        cbar=True,
        square=(seq_q == seq_k),
        linewidths=.5, 
        linecolor='lightgray',
        annot=False 
    )
    plt.xticks(rotation=90, fontsize=max(4, 8 - seq_k // 10))
    plt.yticks(rotation=0, fontsize=max(4, 8 - seq_q // 10))
    plt.title(f"Attention: Layer {layer_idx} Head {head_idx}{title_suffix}", fontsize=12)
    plt.xlabel("Key Tokens", fontsize=10)
    plt.ylabel("Query Tokens", fontsize=10)
    plt.tight_layout(pad=1.5)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=120)
            logger.info(f"Attention heatmap saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save attention heatmap '{save_path}': {e}")
    plt.close()


def calculate_char_accuracy(eval_results: List[Dict[str, Any]]) -> float:
    if not eval_results:
        return 0.0

    total_chars_to_compare = 0
    correct_chars_count = 0

    for item in eval_results:
        target_full = item.get('target', '')
        target_no_marker = target_full.replace(config.END_MARKER, "").strip()
        target_no_space = target_no_marker.replace(" ", "")

        generated_text = item.get('generated', '')
        generated_no_space = generated_text.replace(" ", "")

        if generated_text.startswith("[generation") or not generated_no_space or not target_no_space:
            total_chars_to_compare += len(target_no_space)
            continue

        compare_len = len(target_no_space)
        total_chars_to_compare += compare_len

        for i in range(compare_len):
            if i < len(generated_no_space) and generated_no_space[i] == target_no_space[i]:
                correct_chars_count += 1

    accuracy = (correct_chars_count / total_chars_to_compare) if total_chars_to_compare > 0 else 0.0
    logger.info(f"Character-level accuracy: {correct_chars_count} / {total_chars_to_compare} = {accuracy:.4f}")
    return accuracy