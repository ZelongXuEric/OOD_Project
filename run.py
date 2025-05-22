import os
import time
import torch
import json
import logging
import traceback
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)


import config
import tasks
import models
import analysis
import utils
from tqdm import tqdm

logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main() -> Dict[str, Any]:
    start_time = time.time()
    logger.info(f"Start experiment ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    utils.set_random_seed(42)

    analysis_results: Dict[str, Any] = {}

    logger.info("\nExperiment Configuration")
    is_word_task = "word" in config.PATTERN_TYPE
    task_unit_name = "word" if is_word_task else "char"

    config_summary = (
        f"Model: {config.MODEL_NAME}\n"
        f"Task: {config.PATTERN_TYPE} (Mode: {'Chat Template' if config.USE_CHAT_TEMPLATE else 'Raw Prompt'}, "
        f"{'Few-Shot' if config.NUM_SHOTS > 0 else 'Zero-Shot'}, EndMarker: '{config.END_MARKER}')\n"
        f"Sequence Length: {config.SEQUENCE_LENGTH} {task_unit_name}(s)"
        f"{f', Few-Shot Examples: {config.NUM_SHOTS}' if config.NUM_SHOTS > 0 else ''}\n"
        f"Number of Samples: {config.NUM_EXAMPLES}\n"
        f"Layers for Extraction: {config.LAYER_INDICES}\n"
        f"Device: {config.DEVICE}, Batch Size: {config.BATCH_SIZE}\n"
        f"Word Extraction Strict Mode: {config.WORD_EXTRACTION_STRICT}"
        f"{f', Min Word Extraction Ratio: {config.MIN_WORD_RATIO*100:.0f}%' if not config.WORD_EXTRACTION_STRICT else ''}"
    )
    logger.info(config_summary)

    output_dir = config.OUTPUT_DIR
    figures_dir = config.FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Figures Directory: {figures_dir}")
    logger.info("-" * 30)

    # 1. Load Model and Tokenizer
    logger.info("\n1. Loading Model and Tokenizer")
    try:
        model, tokenizer = models.load_model(
            model_name=config.MODEL_NAME,
            device=config.DEVICE,
            force_float32=True
        )
        # Determine terminators for generation
        terminators = []
        if tokenizer.eos_token_id is not None:
            terminators.append(tokenizer.eos_token_id)
        
        # Try to get <|eot_id|> if it's Llama 3 specific and not the main EOS
        # config.EOS_TOKEN_ID should be the ID for config.END_MARKER
        if config.EOS_TOKEN_ID is not None and config.EOS_TOKEN_ID not in terminators:
            terminators.append(config.EOS_TOKEN_ID)
        
        # Check model's generation config for any other specified EOS tokens
        if hasattr(model.generation_config, "eos_token_id"):
            gen_eos_ids = model.generation_config.eos_token_id
            if isinstance(gen_eos_ids, int) and gen_eos_ids not in terminators:
                terminators.append(gen_eos_ids)
            elif isinstance(gen_eos_ids, list):
                for tid in gen_eos_ids:
                    if tid not in terminators:
                        terminators.append(tid)
        
        terminators = list(set(terminators))
        if not terminators:
            # This should ideally not happen if tokenizer.eos_token_id or config.EOS_TOKEN_ID is set
            raise ValueError("ERROR: Could not determine any valid EOS token IDs for generation!")
        logger.info(f"Using Token IDs as terminators for generation: {terminators}")

    except Exception as e:
        logger.error(f"ERROR: Failed to load model '{config.MODEL_NAME}': {e}", exc_info=True)
        return {"error": f"Model loading failed: {str(e)}"}

    # 2. Generate Data
    logger.info("\n2. Generating Data")
    try:
        pattern_data = tasks.generate_patterned_data(
            num_examples=config.NUM_EXAMPLES,
            seq_len=config.SEQUENCE_LENGTH,
            num_shots=config.NUM_SHOTS,
            pattern_type=config.PATTERN_TYPE,
            end_marker=config.END_MARKER
        )
        if not pattern_data:
            raise ValueError("No samples were generated.")
        logger.info(f"Generated {len(pattern_data)} samples for task: {config.PATTERN_TYPE}.")

        targets_with_marker = [item['target'] for item in pattern_data]

        if config.USE_CHAT_TEMPLATE:
            messages_for_generation = [item['messages'] for item in pattern_data]
            prompts_for_log_and_raw_analysis = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_for_generation
            ]
            prompts_for_model_input = messages_for_generation
        else:
            prompts_for_generation = [item['prompt'] for item in pattern_data]
            prompts_for_log_and_raw_analysis = prompts_for_generation
            prompts_for_model_input = prompts_for_generation

    except Exception as e:
        logger.error(f"ERROR: Data generation failed: {e}", exc_info=True)
        return {"error": f"Data generation failed: {str(e)}"}

    # Prepare full text sequences for state extraction and analysis
    logger.info("\nPreparing full text sequences for state extraction/analysis")
    full_sequences_for_state_extraction_str: List[str] = []
    if pattern_data and tokenizer: # Ensure dependencies are ready
        if config.USE_CHAT_TEMPLATE:
            for item_idx, item_chat in enumerate(pattern_data):
                messages_with_gt_reply = list(item_chat['messages'])
                messages_with_gt_reply.append({"role": "assistant", "content": item_chat['target_no_marker']})
                try:
                    full_dialog_str = tokenizer.apply_chat_template(
                        messages_with_gt_reply,
                        tokenize=False,
                        add_generation_prompt=False # Correct for full dialogue
                    )
                    full_sequences_for_state_extraction_str.append(full_dialog_str)
                except Exception as e_template_apply:
                    logger.error(f"ERROR: Failed to apply chat template for sample {item_idx} to build full sequence: {e_template_apply}", exc_info=True)
                    full_sequences_for_state_extraction_str.append("") # Placeholder
        else: # Non-chat template mode
            for item_raw in pattern_data:
                if 'prompt' in item_raw and 'target' in item_raw:
                    full_sequences_for_state_extraction_str.append(item_raw['prompt'] + item_raw['target'])
                else:
                    logger.error(f"ERROR: Item in pattern_data (raw mode) missing 'prompt' or 'target' key.")
                    full_sequences_for_state_extraction_str.append("") # Placeholder
        
        if len(full_sequences_for_state_extraction_str) == len(pattern_data):
            logger.info(f"Constructed {len(full_sequences_for_state_extraction_str)} full text sequences for state extraction/analysis.")
        else:
            logger.error(f"ERROR: Mismatch in constructed full text sequence count ({len(full_sequences_for_state_extraction_str)}) "
                         f"and sample data count ({len(pattern_data)}). Subsequent analysis may be flawed.")
            # Decide error handling: return {} or raise
    else:
        logger.error("ERROR: pattern_data or tokenizer not initialized. Cannot construct full text sequences for state extraction.")
        return {"error": "Dependencies for full_sequences_for_state_extraction_str not met."}

    # 3. Tokenizer Debug ---
    logger.info(f"\n3. Tokenizer Debug Info ({config.PATTERN_TYPE} Task)")
    try:
        if not pattern_data:
            logger.info("  No samples generated for Tokenizer Debug.")
        else:
            debug_item = pattern_data[0]
            # Determine the actual prompt string fed to the tokenizer for encoding
            if config.USE_CHAT_TEMPLATE:
                # For chat template, the "prompt" is the result of apply_chat_template
                # (including add_generation_prompt=True for the *generation* step,
                # but for *logging/analysis* of this input part, it's what was in messages_for_generation[0])
                prompt_text_for_debug = prompts_for_log_and_raw_analysis[0]

            else: # Raw prompt mode
                prompt_text_for_debug = debug_item["prompt"]

            enc = tokenizer(prompt_text_for_debug, add_special_tokens=False) # Use the derived prompt_text_for_debug
    
            raw_ids = enc["input_ids"]
            if isinstance(raw_ids, int): prompt_ids = [raw_ids]
            elif isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list): prompt_ids = raw_ids[0]
            elif hasattr(raw_ids, "tolist"): prompt_ids = raw_ids.tolist()
            else: prompt_ids = raw_ids
    
            prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
            logger.info(f"  Prompt Tokens (from input to model): {prompt_tokens}")
            logger.info(f"  Total Prompt Tokens: {len(prompt_ids)}")
    
            gold_target_with_marker_str = debug_item["target"]
            target_ids = tokenizer(gold_target_with_marker_str, add_special_tokens=False)["input_ids"]
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
            logger.info(f"\n  Gold Target (with Marker): '{gold_target_with_marker_str}'")
            logger.info(f"  Target Tokens:   {target_tokens}")
            logger.info(f"  Target Token Count: {len(target_ids)}")

    except Exception as e:
        logger.error(f"ERROR: Tokenizer Debug failed: {e}", exc_info=True)


    # 4. Extract/Load Hidden States 
    logger.info("4. Extracting/Loading Hidden States")
    hidden_states_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    hs_file_suffix = "_chat" if config.USE_CHAT_TEMPLATE else "_rawprompt"
    hidden_states_file = f"hidden_states_data{hs_file_suffix}_{config.MODEL_NAME.split('/')[-1]}_{config.PATTERN_TYPE}_L{config.SEQUENCE_LENGTH}words_shots{config.NUM_SHOTS}.pkl"
    hidden_states_path = os.path.join(output_dir, hidden_states_file)

    if os.path.exists(hidden_states_path):
        logger.info(f"Found cached hidden states file: {hidden_states_path}")
        loaded_data = utils.load_results(hidden_states_file, output_dir)
        if isinstance(loaded_data, dict) and all(isinstance(k, int) for k in loaded_data.keys()):
            is_valid_format = True
            if loaded_data:
                first_layer_data = next(iter(loaded_data.values()))
                if not (isinstance(first_layer_data, list) and
                        (not first_layer_data or (isinstance(first_layer_data[0], tuple) and len(first_layer_data[0]) == 3))):
                    is_valid_format = False
            if is_valid_format:
                hidden_states_data = loaded_data
                logger.info(" Cache loaded successfully.")
            else:
                logger.warning(f" Error: Loaded hidden states file format is incorrect. Re-extracting...")
                hidden_states_data = {}
        else:
            logger.warning(f" Error: Root type of loaded hidden states file is incorrect. Re-extracting...")
            hidden_states_data = {}

    if not hidden_states_data and config.LAYER_INDICES:
        logger.info("Needs to extract hidden states (including IDs and Offsets)...")
        try:
            num_layers_total = model.config.num_hidden_layers + 1
            actual_indices_map = {}
            actual_layer_indices = []
            for idx in config.LAYER_INDICES:
                real = idx if idx >= 0 else num_layers_total + idx
                if 0 <= real < num_layers_total:
                    actual_indices_map[idx] = real
                    if real not in actual_layer_indices:
                        actual_layer_indices.append(real)
                else:
                    logger.warning(f"Requested layer {idx} (actual {real}) out of range, ignoring.")
            actual_layer_indices.sort()
            if not actual_layer_indices:
                raise ValueError("No valid layer indices.")
            logger.info(f"Actual layer indices to be extracted: {actual_layer_indices}")

            full_sequences_for_state_extraction_str: List[str] = []
            if config.USE_CHAT_TEMPLATE:
                for idx, item in enumerate(pattern_data):
                    # item['messages'] already contains the test sample's user query
                    # item['target_no_marker'] is the ground-truth assistant reply
                    messages_with_gt_reply = list(item['messages'])
                    messages_with_gt_reply.append({"role": "assistant", "content": item['target_no_marker']})
                    
                    # Apply template, but without adding generation prompt
                    full_dialog_str = tokenizer.apply_chat_template(
                        messages_with_gt_reply,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    full_sequences_for_state_extraction_str.append(full_dialog_str)
            else:
                full_sequences_for_state_extraction_str = [
                    p + t for p, t in zip(prompts_for_generation, targets_with_marker)
                ]
            
            if full_sequences_for_state_extraction_str and \
               len(full_sequences_for_state_extraction_str) == len(pattern_data):
                hidden_states_data = models.extract_hidden_states_with_details(
                    model, tokenizer, full_sequences_for_state_extraction_str,
                    actual_layer_indices,
                    config.BATCH_SIZE, config.DEVICE
                )
                if hidden_states_data:
                    logger.info("\nExtracted hidden states details:")
                    for req_idx in config.LAYER_INDICES:
                        actual_idx_val = actual_indices_map.get(req_idx)
                        if actual_idx_val is not None and actual_idx_val in hidden_states_data:
                            data_list = hidden_states_data[actual_idx_val]
                            if data_list:
                                s, i_ids, o = data_list[0]
                                logger.info(f" Layer {req_idx} (mapped to {actual_idx_val}): {len(data_list)} samples, e.g., shapes S:{s.shape}, ID:{i_ids.shape}, Off:{o.shape}")
                    utils.save_results(hidden_states_data, hidden_states_file, output_dir)
                else:
                    logger.warning(" No hidden states were extracted.")
            else:
                logger.warning(" Failed to construct sequences for state extraction.")

        except Exception as e:
            logger.error(f"Error: Hidden state extraction failed: {e}", exc_info=True)
            hidden_states_data = {}
    elif not config.LAYER_INDICES:
        logger.info(" Skipping hidden state extraction/loading because config.LAYER_INDICES is empty.")
    else:
        logger.info(" Successfully loaded cached hidden states (including IDs and Offsets).")


    # 5. Task Evaluation
    logger.info(f"\n5. {config.PATTERN_TYPE.capitalize()} task accuracy test ({'Few-Shot' if config.NUM_SHOTS > 0 else 'Zero-Shot'})")
    task_correct_token_exact = 0
    task_correct_word_sequence = 0
    task_eval_results = []
    mismatch_print_count = 0
    max_mismatch_prints = 10
    tokenizer_analysis_count = 0
    max_samples_for_tokenizer_analysis = 2
    task_max_new_tokens = 80

    # Update GenerationConfig
    try:
        if hasattr(model, "generation_config"):
            generation_config = model.generation_config
        else:
            # Attempt to load from pretrained model config if the model has no bound config
            generation_config = GenerationConfig.from_pretrained(
                config.MODEL_NAME, trust_remote_code=True
            )

        # Combine terminators
        terminators_set = set()
        if tokenizer.eos_token_id is not None:
            terminators_set.add(tokenizer.eos_token_id)
        if config.EOS_TOKEN_ID is not None:  # The <|eot_id|> you defined
            terminators_set.add(config.EOS_TOKEN_ID)
        # Llama 3 Instruct may implicitly use <|end_of_text|> (128001) as a final stop
        # Usually <|eot_id|> is sufficient. If issues persist, consider adding 128001.
        # eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Already in config.EOS_TOKEN_ID
        # end_of_text_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        # if end_of_text_id is not None: terminators_set.add(end_of_text_id)

        terminators = list(terminators_set)
        if not terminators:
            logger.error("Error: Unable to determine any valid termination token IDs.")
            if tokenizer.eos_token_id is not None:
                terminators = [tokenizer.eos_token_id]
                logger.warning(f"Falling back to tokenizer.eos_token_id ({tokenizer.eos_token_id}) as terminator.")
            else:
                raise ValueError("Cannot set valid termination token IDs.")

        generation_config.eos_token_id = terminators
        generation_config.max_new_tokens = task_max_new_tokens
        generation_config.do_sample = False
        generation_config.num_beams = 1
        # Ensure pad_token_id is set correctly
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        elif terminators:  # If tokenizer lacks pad_token_id, fall back to eos
            generation_config.pad_token_id = terminators[0]
            logger.warning(f"Tokenizer missing pad_token_id. Set GenerationConfig.pad_token_id to first eos_token_id: {generation_config.pad_token_id}")
        else:
            logger.error("Tokenizer missing pad_token_id and cannot fall back from eos_token_id.")

        model.generation_config = generation_config
        logger.info(" Updated/applied GenerationConfig to model.")
        logger.info(f"   eos_token_id for generation: {model.generation_config.eos_token_id}")
        logger.info(f"   pad_token_id for generation: {model.generation_config.pad_token_id}")
        logger.info(f"   max_new_tokens for generation: {model.generation_config.max_new_tokens}")

    except Exception as e_gen_cfg_outer:
        logger.error(f"Error: Failed to set GenerationConfig: {e_gen_cfg_outer}", exc_info=True)
        return {"error": "GenerationConfig setup failed"}



    all_generated_sequences_ids: List[List[int]] = []
    all_prompt_token_lens: List[int] = []
    all_target_token_ids_gt: List[List[int]] = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, config.NUM_EXAMPLES, config.BATCH_SIZE), desc="Model generation batch"):
            batch_items_for_gen = pattern_data[i : i + config.BATCH_SIZE]
            
            current_batch_input_ids_for_gen: Optional[torch.Tensor] = None
            current_batch_attention_mask: Optional[torch.Tensor] = None
            current_batch_prompt_lengths_list: List[int] = []

            if config.USE_CHAT_TEMPLATE:
                batch_messages = [item['messages'] for item in batch_items_for_gen]
                try:
                    tokenized_prompts_list = []
                    for single_sample_messages in batch_messages:
                        ids = tokenizer.apply_chat_template(single_sample_messages, add_generation_prompt=True, return_tensors="pt")
                        tokenized_prompts_list.append(ids.squeeze(0))
                    
                    max_len_in_batch = max(len(ids_tensor) for ids_tensor in tokenized_prompts_list)
                    padded_ids_list = []
                    attention_mask_list = []
                    
                    for ids_tensor in tokenized_prompts_list:
                        prompt_len = ids_tensor.size(0)
                        current_batch_prompt_lengths_list.append(prompt_len)
                        padding_needed = max_len_in_batch - prompt_len
                        if padding_needed > 0:
                            pad_tensor = torch.full((padding_needed,), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.EOS_TOKEN_ID, dtype=torch.long)
                            padded_ids = torch.cat([pad_tensor, ids_tensor], dim=0)
                            attn_mask = torch.cat([torch.zeros(padding_needed, dtype=torch.long), torch.ones(prompt_len, dtype=torch.long)], dim=0)
                        else:
                            padded_ids = ids_tensor
                            attn_mask = torch.ones(prompt_len, dtype=torch.long)
                        padded_ids_list.append(padded_ids)
                        attention_mask_list.append(attn_mask)

                    current_batch_input_ids_for_gen = torch.stack(padded_ids_list).to(config.DEVICE)
                    current_batch_attention_mask = torch.stack(attention_mask_list).to(config.DEVICE)

                except Exception as e_chat_tok:
                    logger.error(f"Chat template tokenization/padding 失败 (批次 {i//config.BATCH_SIZE}): {e_chat_tok}", exc_info=True)
                    for _ in batch_items_for_gen: all_prompt_token_lens.append(0)
                    continue
            else: # Raw prompt mode
                batch_raw_prompts = [item['prompt'] for item in batch_items_for_gen]
                inputs_tokenized = tokenizer(
                    batch_raw_prompts, return_tensors="pt", padding=True, truncation=True, 
                    max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 8000 else 2048,
                    return_attention_mask=True
                ).to(config.DEVICE)
                current_batch_input_ids_for_gen = inputs_tokenized['input_ids']
                current_batch_attention_mask = inputs_tokenized['attention_mask']
                current_batch_prompt_lengths_list = current_batch_attention_mask.sum(dim=1).cpu().tolist()

            all_prompt_token_lens.extend(current_batch_prompt_lengths_list)

            batch_gt_targets_with_marker_str = [item['target'] for item in batch_items_for_gen]
            target_encodings_gt = tokenizer(batch_gt_targets_with_marker_str, add_special_tokens=False, padding=False, truncation=False)
            all_target_token_ids_gt.extend(target_encodings_gt['input_ids'])

            outputs = None
            try:
                outputs = model.generate(
                    input_ids=current_batch_input_ids_for_gen,
                    attention_mask=current_batch_attention_mask,
                    max_new_tokens=model.generation_config.max_new_tokens, 
                    eos_token_id=model.generation_config.eos_token_id, 
                    pad_token_id=model.generation_config.pad_token_id, 
                    do_sample=False, 
                    num_beams=1, 
                    return_dict_in_generate=True 
                )
                all_generated_sequences_ids.extend(outputs.sequences.cpu().tolist())
            except Exception as e_gen:
                logger.error(f"\nError: Model generation failed in batch {i//config.BATCH_SIZE}: {e_gen}", exc_info=True)
                for k_idx_err in range(current_batch_input_ids_for_gen.size(0)):
                     all_generated_sequences_ids.append(current_batch_input_ids_for_gen[k_idx_err].cpu().tolist()) # Store prompt as gen on error
            
            del outputs, current_batch_input_ids_for_gen, current_batch_attention_mask
            if 'inputs_tokenized' in locals(): del inputs_tokenized
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        # Second loop: evaluation
        logger.info(f"\n5. {config.PATTERN_TYPE.capitalize()} task accuracy test ({'Few-Shot' if config.NUM_SHOTS > 0 else 'Zero-Shot'})")
        task_correct_token_exact = 0
        task_correct_word_sequence = 0
        task_eval_results = []
        mismatch_print_count = 0
        max_mismatch_prints = 10 
        tokenizer_analysis_count = 0
        max_samples_for_tokenizer_analysis = 2 
    
        if len(pattern_data) != len(all_generated_sequences_ids) or \
           len(pattern_data) != len(all_prompt_token_lens) or \
           len(pattern_data) != len(all_target_token_ids_gt):
            logger.error("Error: Inconsistent lengths of data lists required for evaluation! Cannot proceed.")
            logger.error(f"  pattern_data: {len(pattern_data)}, all_generated_sequences_ids: {len(all_generated_sequences_ids)}, "
                         f"all_prompt_token_lens: {len(all_prompt_token_lens)}, all_target_token_ids_gt: {len(all_target_token_ids_gt)}")
            return {"error": "Inconsistent evaluation data list lengths"}
    
    
        for global_idx in tqdm(range(len(pattern_data)), desc="Evaluating generated results"):
            item = pattern_data[global_idx]
            prompt_for_log = prompts_for_log_and_raw_analysis[global_idx]
    
            gold_target_with_marker_str = item.get('target', '[TARGET_MISSING]')
            input_seq_gt_str = item.get('input_seq', '[INPUT_SEQ_MISSING]')
            target_no_marker_gt_str = item.get('target_no_marker', '[TARGET_NO_MARKER_MISSING]')
    
            target_token_ids_gt_sample = all_target_token_ids_gt[global_idx]  # List of ints
            full_generated_ids_sample = all_generated_sequences_ids[global_idx]  # List of ints
            prompt_len_tokens_sample = all_prompt_token_lens[global_idx]  # int
    
            generated_ids_only = []
            processed_gen_ids_for_token_match = []
            decoded_gen_text_for_word_match = "[generation error or too short]"
            is_correct_token_exact_sample = False
            is_correct_word_sequence_sample = False
            generated_words_for_eval_sample = []
    
            gold_target_words_gt = [w for w in target_no_marker_gt_str.split(' ') if w]
    
            if prompt_len_tokens_sample == 0 and full_generated_ids_sample:
                 logger.warning(f"Sample {global_idx+1}: Prompt token length is 0 but generation exists.")
                 generated_ids_only = full_generated_ids_sample
            elif len(full_generated_ids_sample) > prompt_len_tokens_sample:
                generated_ids_only = full_generated_ids_sample[prompt_len_tokens_sample:]
            else:  # Generated sequence not longer than prompt
                logger.warning(f"Sample {global_idx+1}: Generated sequence length ({len(full_generated_ids_sample)}) "
                               f"is not greater than prompt length ({prompt_len_tokens_sample}). No generated portion.")
            # generated_ids_only remains empty
    
            first_stop_idx = -1
            actual_stop_token_id = -1
            if generated_ids_only:  # Only search if there is content
                for stop_idx, token_id in enumerate(generated_ids_only):
                    if token_id in terminators:  # Using the previously defined terminators list
                        first_stop_idx = stop_idx
                        actual_stop_token_id = token_id  # Record the actual stop token encountered
                        break
            
            if first_stop_idx != -1:
                # If the GT target itself ends with config.EOS_TOKEN_ID and the model also generated it, keep it
                # Otherwise exclude the stop token itself
                if target_token_ids_gt_sample and \
                   target_token_ids_gt_sample[-1] == actual_stop_token_id and \
                   actual_stop_token_id == config.EOS_TOKEN_ID:
                    processed_gen_ids_for_token_match = generated_ids_only[:first_stop_idx + 1]
                else:
                    processed_gen_ids_for_token_match = generated_ids_only[:first_stop_idx]
            else:
                processed_gen_ids_for_token_match = list(generated_ids_only)
    
            # Decode text for word matching
            decoded_gen_text_for_word_match = tokenizer.decode(
                processed_gen_ids_for_token_match,
                skip_special_tokens=True  # Skip all special tokens, including possible EOS
            ).strip()
    
            temp_decoded_text = decoded_gen_text_for_word_match
            if not config.USE_CHAT_TEMPLATE:  # Only strip prefixes in raw-prompt mode
                common_prefixes_to_strip = [": ", ":"]
                for prefix in common_prefixes_to_strip:
                    if temp_decoded_text.startswith(prefix):
                        temp_decoded_text = temp_decoded_text[len(prefix):].strip()
            
            generated_words_for_eval_sample = [w for w in temp_decoded_text.split(' ') if w]
    
            # Word-sequence match evaluation
            if gold_target_words_gt and generated_words_for_eval_sample:
                if len(generated_words_for_eval_sample) == len(gold_target_words_gt):
                    is_correct_word_sequence_sample = all(
                        gen_w.lower() == gt_w.lower()
                        for gen_w, gt_w in zip(generated_words_for_eval_sample, gold_target_words_gt)
                    )
            if is_correct_word_sequence_sample:
                task_correct_word_sequence += 1
    
            # Token exact-match evaluation
            if processed_gen_ids_for_token_match and target_token_ids_gt_sample:
                is_correct_token_exact_sample = (processed_gen_ids_for_token_match == target_token_ids_gt_sample)
                if is_correct_token_exact_sample:
                    task_correct_token_exact += 1
    
            # Detailed logging (on mismatch or for first few samples)
            if (not is_correct_token_exact_sample and mismatch_print_count < max_mismatch_prints) or \
               (tokenizer_analysis_count < max_samples_for_tokenizer_analysis):
                if not is_correct_token_exact_sample:
                     logger.info(f"\nToken MISMATCH (Sample {global_idx+1})")
                     mismatch_print_count += 1
                else:  # For tokenizer_analysis_count
                     logger.info(f"\nTokenizer/Generation Analysis (Sample {global_idx+1})")
                
                logger.info(f" Prompt (for generation, may be long, showing last 250 chars): ...{prompt_for_log[-250:]}")
                logger.info(f" GT Input Seq (for analysis): '{input_seq_gt_str}'")
                logger.info(f" GT Target (with Marker): '{gold_target_with_marker_str}'")
                logger.info(f" GT Target (no Marker): '{target_no_marker_gt_str}' (GT words: {gold_target_words_gt})")
    
                logger.info(f" Model generation details:")
                logger.info(f"   Prompt token length: {prompt_len_tokens_sample}")
                logger.info(f"   Full generated token IDs ({len(full_generated_ids_sample)}): {full_generated_ids_sample}")
                logger.info(f"   Generated-only token IDs ({len(generated_ids_only)}): {generated_ids_only}")
                
                found_stop_token_in_gen_debug = False
                gen_stop_token_id_debug = -1
                gen_stop_token_text_debug = ""
                if generated_ids_only:
                    logger.info(f"  Generated-only token IDs (raw, after prompt) ({len(generated_ids_only)}): {generated_ids_only}")
                    try:
                        # Print each generated token ID and its text until a terminator or end
                        decoded_gen_only_tokens_debug = []
                        found_stop_in_raw_gen = False
                        for idx, token_id_gen_raw in enumerate(generated_ids_only):
                            tok_text_gen_raw = tokenizer.decode([token_id_gen_raw], skip_special_tokens=False)
                            decoded_gen_only_tokens_debug.append(f"{idx}:{tok_text_gen_raw}({token_id_gen_raw})")
                            if token_id_gen_raw in terminators:
                                decoded_gen_only_tokens_debug.append("<-STOP")
                                found_stop_in_raw_gen = True
                                break
                        logger.info(f"       Decoded generated-only tokens (one by one, including specials, until stop): [{' '.join(decoded_gen_only_tokens_debug)}]")
                        if not found_stop_in_raw_gen:
                            logger.info(f"       No defined stop token found early in generated-only tokens.")
                
                    except Exception as e_dec_raw:
                        logger.error(f"       Failed to decode generated-only (raw): {e_dec_raw}")
                else:
                    logger.info(f"     Generated-only token IDs (raw, after prompt): [empty]")
    
                logger.info(f"   Processed gen IDs (for token match, {len(processed_gen_ids_for_token_match)}): {processed_gen_ids_for_token_match}")
                logger.info(f"  Decoded: '{tokenizer.decode(processed_gen_ids_for_token_match, skip_special_tokens=False)}'")
                logger.info(f"  Decoded (skip_special_tokens): '{tokenizer.decode(processed_gen_ids_for_token_match, skip_special_tokens=True)}'")
    
                logger.info(f"   Decoded gen (for word match, stripped): '{decoded_gen_text_for_word_match}' (words: {generated_words_for_eval_sample})")
                
                logger.info(f" Comparison:")
                logger.info(f"   GT target token IDs ({len(target_token_ids_gt_sample)}): {target_token_ids_gt_sample} ({tokenizer.convert_ids_to_tokens(target_token_ids_gt_sample)})")
                logger.info(f"   Token exact-match result: {is_correct_token_exact_sample}")
                logger.info(f"   Decoded word-sequence match result: {is_correct_word_sequence_sample}")
    
            if tokenizer_analysis_count < max_samples_for_tokenizer_analysis:
                tokenizer_analysis_count += 1
    
    
            task_eval_results.append({
                "index": global_idx,
                "prompt_for_log": prompt_for_log, # The prompt string actually fed to the model (or messages for chat)
                "input_seq_gt": input_seq_gt_str, # Ground truth input word sequence
                "target_gt_with_marker": gold_target_with_marker_str,
                "target_gt_no_marker": target_no_marker_gt_str,
                "target_gt_words": gold_target_words_gt,
                "generated_decoded_for_word_match": decoded_gen_text_for_word_match,
                "generated_words_for_eval": generated_words_for_eval_sample,
                "target_token_ids_gt": target_token_ids_gt_sample,
                "processed_generated_ids_for_token_match": processed_gen_ids_for_token_match,
                "full_generated_ids_incl_prompt": full_generated_ids_sample, 
                "prompt_token_len_for_gen": prompt_len_tokens_sample,
                "is_correct_token_exact": is_correct_token_exact_sample,
                "is_correct_word_sequence": is_correct_word_sequence_sample,
            })
    
        # Calculate and print overall accuracy
        total_evaluated = len(task_eval_results)
        task_accuracy_token_exact = task_correct_token_exact / total_evaluated if total_evaluated > 0 else 0.0
        task_accuracy_word_sequence = task_correct_word_sequence / total_evaluated if total_evaluated > 0 else 0.0

        eval_results_for_char_acc = [
            {"target": r["target_gt_no_marker"], "generated": " ".join(r["generated_words_for_eval"])}
            for r in task_eval_results
        ]
        task_char_accuracy = analysis.calculate_char_accuracy(eval_results_for_char_acc)

        logger.info(f"\n{config.PATTERN_TYPE.capitalize()} (Token Exact Match) Accuracy: {task_accuracy_token_exact:.3%}")
        logger.info(f"{config.PATTERN_TYPE.capitalize()} (Decoded Word Sequence Match) Accuracy: {task_accuracy_word_sequence:.3%}")
        logger.info(f"{config.PATTERN_TYPE.capitalize()} (Character-Level, ignoring spaces/Markers) Accuracy: {task_char_accuracy:.3%}")

        task_results_summary = {
            "accuracy_token_exact_match": task_accuracy_token_exact,
            "accuracy_word_sequence_match": task_accuracy_word_sequence,
            "char_accuracy_no_marker": task_char_accuracy,
            "num_samples_total": config.NUM_EXAMPLES,
            "num_samples_evaluated": total_evaluated,
            "pattern_type": config.PATTERN_TYPE,
            "sequence_length_words": config.SEQUENCE_LENGTH,
            "num_shots": config.NUM_SHOTS,
            "model_name": config.MODEL_NAME,
            "end_marker_used": config.END_MARKER,
            "use_chat_template": config.USE_CHAT_TEMPLATE,
            "word_extraction_strict": config.WORD_EXTRACTION_STRICT,
            "min_word_ratio": getattr(config, "MIN_WORD_RATIO", 0.8) if not config.WORD_EXTRACTION_STRICT else 1.0,
            "max_new_tokens_setting_for_task": task_max_new_tokens,
            "terminators_used_for_generation": terminators,
        }
        utils.save_results(task_results_summary, f"summary_{config.PATTERN_TYPE}.json", output_dir)
        utils.save_results(task_eval_results, f"details_{config.PATTERN_TYPE}.json", output_dir)

        analysis_results["task_accuracy_token_exact_match"] = task_accuracy_token_exact
        analysis_results["task_accuracy_word_sequence_match"] = task_accuracy_word_sequence
        analysis_results["char_accuracy_no_marker"] = task_char_accuracy
        analysis_results.update(task_results_summary)

    # 6. Hidden State Analysis
    logger.info("\n6. Starting hidden state analysis")
    if hidden_states_data and config.LAYER_INDICES:
        analysis_successful = True

        # Ensure full_sequences_for_state_extraction_str matches pattern_data length and is non-empty
        if not full_sequences_for_state_extraction_str or \
           len(full_sequences_for_state_extraction_str) != len(pattern_data):
            logger.error(f"Error: full_sequences_for_state_extraction_str length ({len(full_sequences_for_state_extraction_str)}) "
                         f"does not match pattern_data length ({len(pattern_data)}) or is empty. Cannot proceed with analysis.")
            analysis_successful = False

        if analysis_successful:
            # 6.A PCA Analysis
            try:
                logger.info("Running PCA analysis (using last token state)...")
                analysis.run_pca_analysis(
                    hidden_states_data,
                    pattern_data,
                    full_sequences_for_state_extraction_str,
                    tokenizer, model,
                    config.LAYER_INDICES, config.SEQUENCE_LENGTH, figures_dir,
                    pattern_type=config.PATTERN_TYPE,
                    strict_extraction=config.WORD_EXTRACTION_STRICT,
                    min_word_ratio=config.MIN_WORD_RATIO,
                    use_last_token_state=True
                )
                logger.info("Running PCA analysis (using average word state)...")
                analysis.run_pca_analysis(
                    hidden_states_data,
                    pattern_data,
                    full_sequences_for_state_extraction_str,
                    tokenizer, model,
                    config.LAYER_INDICES, config.SEQUENCE_LENGTH, figures_dir,
                    pattern_type=config.PATTERN_TYPE,
                    strict_extraction=config.WORD_EXTRACTION_STRICT,
                    min_word_ratio=config.MIN_WORD_RATIO,
                    use_last_token_state=False
                )
            except Exception as e_pca:
                logger.error(f"Error: PCA analysis failed: {e_pca}", exc_info=True)
                analysis_successful = False

            # 6.B Linear Probe Analysis
            try:
                logger.info("Running linear probe analysis...")
                analysis.run_linear_probe_analysis(
                    hidden_states_data,
                    pattern_data,
                    full_sequences_for_state_extraction_str,
                    tokenizer,
                    model,
                    config.LAYER_INDICES,
                    config.SEQUENCE_LENGTH,
                    task_unit,
                    output_dir,
                    pattern_type=config.PATTERN_TYPE,
                    strict_extraction=config.WORD_EXTRACTION_STRICT,
                    min_word_ratio=config.MIN_WORD_RATIO
                )
            except Exception as e_probe:
                logger.error(f"Error: Linear probe analysis failed: {e_probe}", exc_info=True)
                analysis_successful = False

            # 6.C Cosine Similarity Analysis
            try:
                logger.info("Running cosine similarity analysis (using last token state)...")
                analysis.run_cosine_similarity_analysis(
                    hidden_states_data,
                    pattern_data,
                    full_sequences_for_state_extraction_str,
                    tokenizer, model,
                    config.LAYER_INDICES,
                    config.SEQUENCE_LENGTH,
                    task_unit,
                    output_dir, figures_dir,
                    pattern_type=config.PATTERN_TYPE,
                    strict_extraction=config.WORD_EXTRACTION_STRICT,
                    min_word_ratio=config.MIN_WORD_RATIO,
                    use_last_token_state=True
                )
                logger.info("Running cosine similarity analysis (using average word state)...")
                analysis.run_cosine_similarity_analysis(
                    hidden_states_data,
                    pattern_data,
                    full_sequences_for_state_extraction_str,
                    tokenizer, model,
                    config.LAYER_INDICES,
                    config.SEQUENCE_LENGTH,
                    task_unit,
                    output_dir, figures_dir,
                    pattern_type=config.PATTERN_TYPE,
                    strict_extraction=config.WORD_EXTRACTION_STRICT,
                    min_word_ratio=config.MIN_WORD_RATIO,
                    use_last_token_state=False
                )
            except Exception as e_cos:
                logger.error(f"Error: Cosine similarity analysis failed: {e_cos}", exc_info=True)
                analysis_successful = False

            if analysis_successful:
                logger.info("Hidden state analysis complete.")
            else:
                logger.warning("Warning: Some or all hidden state analysis steps failed.")

    # 7. Summarize and save all results
    probe_fname = f"linear_probe_results_L{config.SEQUENCE_LENGTH}words.json"
    probe_results = utils.load_results(probe_fname, output_dir)
    if probe_results:
        analysis_results["linear_probe_summary"] = probe_results

    for state_type_suffix_cos in ["lastToken", "avgWord"]:
        sim_fname = f"cosine_similarity_results_L{config.SEQUENCE_LENGTH}{task_unit}_{state_type_suffix_cos}.json"
        similarity_res_type = utils.load_results(sim_fname, output_dir)
        if similarity_res_type:
            analysis_results[f"cosine_similarity_summary_{state_type_suffix_cos}"] = similarity_res_type
        else:
            logger.warning(f"Warning: Could not find or load cosine similarity results file: {sim_fname}")

    utils.save_results(analysis_results, f"all_analysis_results_{config.PATTERN_TYPE}.json", output_dir)

    # 8. Print final summary
    end_time = time.time()
    logger.info(f"\nExperiment complete (total time: {end_time - start_time:.2f} seconds)")
    logger.info("\nExperiment Summary")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Task: {config.PATTERN_TYPE.capitalize()} (USE_CHAT_TEMPLATE: {config.USE_CHAT_TEMPLATE}, {'Few-Shot' if config.NUM_SHOTS > 0 else 'Zero-Shot'}), Sequence length: {config.SEQUENCE_LENGTH} {task_unit}, Shots: {config.NUM_SHOTS}")
    if not config.WORD_EXTRACTION_STRICT:
        logger.info(f"Word extraction: relaxed mode (min ratio: {config.MIN_WORD_RATIO*100:.0f}%)")
    else:
        logger.info("Word extraction: strict mode (100% required)")

    logger.info(f"  Token exact-match accuracy: {analysis_results.get('task_accuracy_token_exact_match', 0.0):.3%}")
    logger.info(f"  Decoded word-sequence match accuracy: {analysis_results.get('task_accuracy_word_sequence_match', 0.0):.3%}")
    logger.info(f"  Character-level accuracy (ignoring spaces/Markers): {analysis_results.get('char_accuracy_no_marker', 0.0):.3%}")

    if probe_results:
        logger.info("\nPartial linear probe results (mean accuracy):")
        for layer_idx_str_probe, results_probe in probe_results.items():
            logger.info(f"  Layer {layer_idx_str_probe}:")
            for probe_name, res_p in results_probe.items():
                acc_p = res_p.get('mean_accuracy', res_p.get('mean_score', 'N/A'))
                if isinstance(acc_p, float) and not np.isnan(acc_p):
                    logger.info(f"    {probe_name}: {acc_p:.4f}")
                else:
                    logger.info(f"    {probe_name}: {acc_p} (insufficient data or error)")

    for state_type_suffix_cos_sum in ["lastToken", "avgWord"]:
        sim_summary_key = f"cosine_similarity_summary_{state_type_suffix_cos_sum}"
        if sim_summary_key in analysis_results and analysis_results[sim_summary_key] is not None:
            logger.info(f"\nPartial cosine similarity results (overall mean, using {state_type_suffix_cos_sum} state):")
            for layer_idx_str_cos, results_cos in analysis_results[sim_summary_key].items():
                logger.info(f"  Layer {layer_idx_str_cos}:")
                sim_correct_val = results_cos.get(f"overall_avg_similarity_correct_{state_type_suffix_cos_sum}")
                sim_control_val = results_cos.get(f"overall_avg_similarity_diff_control_{state_type_suffix_cos_sum}")
                sim_label = f"Corresponding {label_unit} (Input[i] vs Output[corr. pos])"

                if sim_correct_val is not None and not np.isnan(sim_correct_val):
                    logger.info(f"    {sim_label}: {sim_correct_val:.4f}")
                else:
                    logger.info(f"    {sim_label}: N/A (insufficient data or error)")

                if sim_control_val is not None and not np.isnan(sim_control_val):
                    logger.info(f"    Different {label_unit} control (Input[i] vs Input[j]): {sim_control_val:.4f}")
                else:
                    logger.info(f"    Different {label_unit} control (Input[i] vs Input[j]): N/A (insufficient data or error)")

    logger.info(f"\nPlease check the {figures_dir} directory for visualizations.")
    logger.info(f"Detailed results are saved in the {output_dir} directory.")

    return analysis_results

if __name__ == "__main__":
    main()