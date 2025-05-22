import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import traceback
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
import config

# 1.  Model Loading
def load_model(
    model_name: str = config.MODEL_NAME,
    device: str = config.DEVICE,
    quantize: bool = False,
    force_float32: bool = True, 
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading: {model_name} to: {device} ...")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer，try trust_remote_code=True: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set Padding Side and Pad Token ID
    tokenizer.padding_side = "left"

    pad_token_set_source = "tokenizer's default"
    if tokenizer.pad_token_id is None:
        if config.EOS_TOKEN_ID is not None:
            tokenizer.pad_token_id = config.EOS_TOKEN_ID
            pad_token_set_source = f"config.EOS_TOKEN_ID ({config.END_MARKER})"
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            pad_token_set_source = "tokenizer.eos_token_id"
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            pad_token_set_source = "tokenizer.unk_token_id"
        else:
            print("CRITICAL WARNING: No pad_token_id found in tokenizer, and could not be set "
                  "from config.EOS_TOKEN_ID, tokenizer.eos_token_id, or tokenizer.unk_token_id. "
                  "Padding-dependent operations may fail. Please check tokenizer configuration "
                  "or manually add a pad token to the tokenizer.")

    print(f"Tokenizer Configuration:")
    print(f"Padding Side: {tokenizer.padding_side}")
    if tokenizer.pad_token_id is not None:
        pad_token_text = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False)
        print(f"Pad Token ID: {tokenizer.pad_token_id} (corresponds to: '{pad_token_text}', set from: {pad_token_set_source})")
    else:
        print(f"Pad Token ID: Not set (CRITICAL ISSUE)")
    print(f"Using END_MARKER for logic: '{config.END_MARKER}' (ID: {config.EOS_TOKEN_ID})")


    #Determine Model Data Type
    if force_float32:
        model_dtype = torch.float32
        print("Force float32 to load the model.。")
        if quantize:
            print("Warning: When force_float32=True, quantization will be disabled.")
            quantize = False
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print("Load the model using bfloat16.")
    elif device == "cuda":
        model_dtype = torch.float16
        print("BF16 does not support it. Use float16 to load the model.")
    else: # CPU
        model_dtype = torch.float32
        print("Load model with float32 on CPU.")

    # Prepare Model Loading Keyword Arguments
    model_kwargs = dict(
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=model_dtype,
    )

    # Quantization configuration (enabled only when it is not force_float32 and is on CUDA)
    if quantize and device == "cuda" and not force_float32:
        print("Using 4-bit quantization loading...")
        # Determine compute dtype based on model_dtype
        compute_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    else:
        # Use 'auto' for device mapping if on CUDA, otherwise use the CPU
        model_kwargs["device_map"] = "auto" if device == "cuda" else device

    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
         print(f"Error loading model, try trust_remote_code=True: {e}")
         model_kwargs["trust_remote_code"] = True
         try:
             model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
         except Exception as e_trust:
             print(f"Loading the model with trust_remote_code=True still failed: {e_trust}")
             raise

    # Set Default Generation Config
    try:
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            generation_config = model.generation_config
        else:
            generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True) # Add trust_remote_code here too just in case

        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        # Ensure EOS is set correctly if possible
        if hasattr(generation_config, 'eos_token_id'):
             eos_to_set = getattr(config, 'EOS_TOKEN_ID', None) # Prefer config ID
             if eos_to_set is None:
                 eos_to_set = tokenizer.eos_token_id # Fallback to tokenizer default
             if eos_to_set is not None:
                # Check if model expects a list (like Llama3 generation_config.json)
                if isinstance(generation_config.eos_token_id, list):
                    if eos_to_set not in generation_config.eos_token_id:
                        # Decide whether to append or overwrite. Appending seems safer.
                        generation_config.eos_token_id.append(eos_to_set)
                        generation_config.eos_token_id = list(set(generation_config.eos_token_id)) # De-duplicate
                else: # If it's a single ID, overwrite
                    generation_config.eos_token_id = eos_to_set

        # Set other configs to match model loading flags
        generation_config.return_dict_in_generate = True
        generation_config.output_attentions = getattr(model.config, 'output_attentions', False)
        generation_config.output_hidden_states = getattr(model.config, 'output_hidden_states', False)

        model.generation_config = generation_config
        print("Model generation_config has been set/updated.")

    except Exception as e_gencfg:
        print(f"Warning: Error setting model generation_config: {e_gencfg}")

    # Device Check Log
    if model_kwargs.get("device_map") == "auto":
        print(f"Model loaded with device_map='auto', distributed across: {model.hf_device_map}")
    elif device != 'cpu' and hasattr(model, 'hf_device_map') and not model.hf_device_map:  # Check if not mapped
        print(f"Model appears to be loaded on: {model.device} (expected: {device})")
    elif device != 'cpu':
        print(f"Model loaded on device: {model.device}")
    else: 
        print("Model loaded on CPU.")

    model.eval()
    print("Model and tokenizer loaded successfully.\n")
    return model, tokenizer

# 2.  Hidden State Extraction
def extract_hidden_states_with_details(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sequences: List[str],
    layer_indices: List[int],
    batch_size: int = config.BATCH_SIZE,
    device: str = config.DEVICE,
) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:

    if torch is None:
        raise RuntimeError("Torch is not installed; cannot extract hidden states.")
    if not layer_indices:
        print("Warning: layer_indices is empty; no hidden states will be extracted.")
        return {}
        
    # Verify that the tier index is valid
    num_layers_total = model.config.num_hidden_layers + 1  # Embedding + N transformer blocks
    valid_layer_indices = sorted([idx for idx in layer_indices if 0 <= idx < num_layers_total])
    if not valid_layer_indices:
        print(f"ERROR: provided layer indices {layer_indices} are all invalid (valid range [0, {num_layers_total-1}]).")
        return {}
    if len(valid_layer_indices) < len(layer_indices):
        print(f"Warning: some requested layer indices are invalid; actually extracting layers: {valid_layer_indices}")

    hidden_details_by_layer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = \
        {idx: [] for idx in valid_layer_indices}
    print(f"Beginning extraction of hidden states, IDs, and offsets; actual layers: {valid_layer_indices}")

    # Device handling
    target_device = (
        model.device
        if not hasattr(model, 'hf_device_map') or not model.hf_device_map
        else next(iter(model.hf_device_map.values()))
    )
    print(f"Inputs will be sent to device: {target_device}")

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting hidden states/IDs/Offsets per batch"):
            batch_sequences = sequences[i : i + batch_size]
            try:
                inputs = tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=(
                        tokenizer.model_max_length
                        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 8000
                        else 2048
                    ),
                    return_attention_mask=True,
                    return_offsets_mapping=True,
                ).to(target_device)
            except Exception as e_tok:
                print(f"\nERROR: tokenizing batch {i//batch_size} failed: {e_tok}")
                traceback.print_exc() 
                continue

            try:
                outputs = model(**inputs)
            except Exception as e_fwd:
                print(f"\nERROR: model forward pass failed on batch {i//batch_size}: {e_fwd}")
                traceback.print_exc()
                continue 

            # Validate outputs
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                print(f"\nERROR: model output is missing 'hidden_states' (batch {i//batch_size}). Ensure output_hidden_states=True when loading the model.")
                continue
            all_states_tuple = outputs.hidden_states  # Tuple of (L+1, B, T_padded, D)
            if len(all_states_tuple) != num_layers_total:
                print(f"\nWarning: number of returned hidden state layers ({len(all_states_tuple)}) does not match expected ({num_layers_total}) (batch {i//batch_size}).")
                continue

            batch_input_ids = inputs["input_ids"]               # (B, T_padded)
            batch_offsets = inputs["offset_mapping"]            # (B, T_padded, 2)
            batch_mask = inputs["attention_mask"]               # (B, T_padded)

            # Extract data for each sample in the batch
            for b_idx in range(batch_input_ids.size(0)):
                seq_len_actual = int(batch_mask[b_idx].sum())
                if seq_len_actual == 0: continue

                # Extract IDs and Offsets from padding
                valid_ids = batch_input_ids[b_idx, :seq_len_actual].detach().cpu()
                valid_offsets = batch_offsets[b_idx, :seq_len_actual].detach().cpu() # Keep as Tensor

                # Extract the status of all request layers for this sample
                for layer_idx in valid_layer_indices:
                    if layer_idx >= len(all_states_tuple): continue 
                    layer_states_batch = all_states_tuple[layer_idx] # (B, T_padded, D)

                    valid_states = layer_states_batch[b_idx, :seq_len_actual, :].detach().cpu()

                    hidden_details_by_layer[layer_idx].append(
                        (valid_states, valid_ids, valid_offsets)
                    )

            del inputs, outputs, all_states_tuple, layer_states_batch, valid_ids, valid_offsets, valid_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Extraction of hidden state, IDs and Offsets is completed.\n")
    return hidden_details_by_layer

# 4.  Simple Self-Test
if __name__ == "__main__":
    print("Running models.py Self-Test")
    test_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print("\nSelf-Test: Loading model (force_float32=True)...")
    model, tok = load_model(test_model_name, device=config.DEVICE, force_float32=True)

    sample_sequences = [
        "Input: A B C D E\nOutput: E D C B A<|eot_id|>",
        "Input: apple banana orange grape lemon\nOutput: lemon grape orange banana apple<|eot_id|>",
    ]
    # Request layers 0, 16, and last (-1 maps to 32 for Llama 3 8B)
    layers_req_hs = [0, 16, -1]
    layers_req_attn = [0, 15, 31] # Attention layers indexed 0 to N-1

    # Test Hidden State Extraction (New Function)
    print("\nTesting Hidden State Extraction (with IDs and Offsets)")
    num_layers_total = model.config.num_hidden_layers + 1
    actual_hs_indices = sorted([idx if idx >= 0 else num_layers_total + idx for idx in layers_req_hs if 0 <= (idx if idx >= 0 else num_layers_total + idx) < num_layers_total])

    # Check if the function exists before calling
    if 'extract_hidden_states_with_details' in locals() or 'extract_hidden_states_with_details' in globals():
        hs_details = extract_hidden_states_with_details(model, tok, sample_sequences, actual_hs_indices, batch_size=1)
        if hs_details:
            print("Extraction successful. Sample shapes:")
            for l_act in actual_hs_indices:
                 if l_act in hs_details:
                      print(f"  Layer {l_act}: {len(hs_details[l_act])} samples")
                      if hs_details[l_act]:
                           # Check the first sample's tuple shapes
                           s, i, o = hs_details[l_act][0]
                           print(f"    e.g., Sample 0 Shapes - States:{s.shape}, IDs:{i.shape}, Offsets:{o.shape}")
                 else:
                      print(f"  Layer {l_act}: Data not found in result.")
        else:
            print("Hidden state (with details) extraction failed or returned empty.")
    else:
        print("ERROR: Function 'extract_hidden_states_with_details' not found in models.py!")
        
    print("\nSelf-Test complete.")