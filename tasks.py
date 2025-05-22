import re
import random
import string
from typing import List, Dict, Tuple, TypedDict, Any
import config 

WORD_SET = [
    "apple", "banana", "orange", "grape", "lemon", "lime", "melon", "peach",
    "table", "chair", "lamp", "book", "pen", "pencil", "paper", "folder",
    "house", "tree", "road", "car", "bus", "train", "plane", "boat",
    "cloud", "rain", "snow", "sun", "moon", "star", "wind", "storm",
    "dog", "cat", "fish", "bird", "lion", "tiger", "bear", "horse",
    "red", "blue", "green", "yellow", "purple", "black", "white", "gray",
    "happy", "sad", "angry", "calm", "fast", "slow", "big", "small",
    "run", "walk", "jump", "read", "write", "sing", "dance", "play",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
]

def generate_word_copy_sequence(seq_len: int, word_set: List[str] = WORD_SET) -> Tuple[str, str]:
    original_words = random.choices(word_set, k=seq_len)
    original_seq_spaced = " ".join(original_words)
    return original_seq_spaced, original_seq_spaced

def generate_word_reverse_sequence(seq_len: int, word_set: List[str] = WORD_SET) -> Tuple[str, str]:
    original_words = random.choices(word_set, k=seq_len)
    reversed_words = original_words[::-1]
    original_seq_spaced = " ".join(original_words)
    reversed_seq_spaced = " ".join(reversed_words)
    return original_seq_spaced, reversed_seq_spaced

def generate_char_reverse_sequence(seq_len: int, char_set: str = string.ascii_uppercase) -> Tuple[str, str]:
    original_chars = random.choices(char_set, k=seq_len)
    reversed_chars = original_chars[::-1]
    original_seq_spaced = " ".join(original_chars)
    reversed_seq_spaced = " ".join(reversed_chars)
    return original_seq_spaced, reversed_seq_spaced

def generate_char_copy_sequence(seq_len: int, char_set: str = string.ascii_uppercase) -> Tuple[str, str]:
    original_chars = random.choices(char_set, k=seq_len)
    original_seq_spaced = " ".join(original_chars)
    return original_seq_spaced, original_seq_spaced

class DataItem(TypedDict, total=False):
    input_seq: str
    target_no_marker: str
    target: str
    messages: List[Dict[str, str]]
    prompt: str
    
def generate_patterned_data(
    num_examples: int = config.NUM_EXAMPLES,
    seq_len: int = config.SEQUENCE_LENGTH,
    num_shots: int = config.NUM_SHOTS,
    pattern_type: str = config.PATTERN_TYPE,
    end_marker: str = config.END_MARKER,
    word_set_override: List[str] = None
) -> List[Dict[str, Any]]: 

    if seq_len < 1: raise ValueError("seq_len must ≥ 1")
    if num_shots < 0: raise ValueError("num_shots can't be negative")

    if pattern_type == "copy_word":
        sequence_generator = generate_word_copy_sequence
        using_word_set = word_set_override if word_set_override else WORD_SET
        task_desc = f"Word Copying (L={seq_len} words)"
    elif pattern_type == "reverse_word":
        sequence_generator = generate_word_reverse_sequence
        using_word_set = word_set_override if word_set_override else WORD_SET
        task_desc = f"Word Reverse (L={seq_len} words)"
    elif pattern_type == "copy": 
        sequence_generator = generate_char_copy_sequence
        using_word_set = string.ascii_uppercase 
        task_desc = f"Character Copying (L={seq_len} chars)"
    elif pattern_type == "reverse": 
        sequence_generator = generate_char_reverse_sequence
        using_word_set = string.ascii_uppercase
        task_desc = f"Character Reverse (L={seq_len} chars)"
    else:
        raise ValueError(f"Unknow pattern_type: {pattern_type}")

    print(f"Genrate {task_desc} task data (Chat Template Mode: {config.USE_CHAT_TEMPLATE})...")
    data: List[Dict[str, Any]] = []

    for example_idx in range(num_examples):
        current_messages = []
        current_raw_prompt = "" 
        
        # 1. System Prompt
        if config.USE_CHAT_TEMPLATE and config.SYSTEM_PROMPT:
            current_messages.append({"role": "system", "content": config.SYSTEM_PROMPT})

        used_sequences_in_prompt = set()

        # 2. Construct a Few-Shot Context
        if num_shots > 0:
            attempts = 0
            max_attempts_few_shot = num_shots * 10
            few_shot_pairs_generated = [] # Store (input, target_no_marker)

            while len(few_shot_pairs_generated) < num_shots and attempts < max_attempts_few_shot:
                example_input, example_target_no_marker = sequence_generator(seq_len, using_word_set)
                if example_input not in used_sequences_in_prompt:
                    few_shot_pairs_generated.append((example_input, example_target_no_marker))
                    used_sequences_in_prompt.add(example_input)
                attempts += 1
            
            if len(few_shot_pairs_generated) < num_shots:
                 Print(f "Warning (sample {example_idx+1}): failed to generate enough ({num_shots}) unique few-shot samples. True: {len(few_shot_pairs_generated)}。")

            for fs_input, fs_target_no_marker in few_shot_pairs_generated:
                if config.USE_CHAT_TEMPLATE:
                    current_messages.append({"role": "user", "content": f"Input: {fs_input}"})
                    current_messages.append({"role": "assistant", "content": fs_target_no_marker}) 
                else:
                    current_raw_prompt += f"Input: {fs_input}\nOutput: {fs_target_no_marker}{end_marker}\n\n"
        
        # 3. Generate Test Samples
        attempts = 0
        max_attempts_test = 20
        test_input, test_target_no_marker = sequence_generator(seq_len, using_word_set)
        while test_input in used_sequences_in_prompt and attempts < max_attempts_test:
            test_input, test_target_no_marker = sequence_generator(seq_len, using_word_set)
            attempts += 1
        
        if test_input in used_sequences_in_prompt:
             print(f"Warning (sample {example_idx+1}): Failed to generate test sample different from few-shot.")

        # 4. Combine the Final Prompt
        if config.USE_CHAT_TEMPLATE:
            current_messages.append({"role": "user", "content": f"Input: {test_input}"})
        else:
            current_raw_prompt += f"Input: {test_input}\nOutput: " 

        final_target_with_marker = f"{test_target_no_marker}{end_marker}"

        data_item = {
            "input_seq": test_input, 
            "target_no_marker": test_target_no_marker, 
            "target": final_target_with_marker, 
        }
        if config.USE_CHAT_TEMPLATE:
            data_item["messages"] = current_messages
        else:
            data_item["prompt"] = current_raw_prompt
        
        data.append(data_item)

    return data

if __name__ == '__main__':

    def print_test_item(item: Dict[str, Any], item_index: int, use_chat_template: bool):
        print(f"\nSample {item_index + 1} (Chat Template: {use_chat_template})")
        if use_chat_template:
            if 'messages' in item:
                print(f"Messages:")
                for msg in item['messages']:
                    print(f" Role: {msg['role']}, Content: '{msg['content']}'")
            else:
                print("Messages: [no 'messages' field generated]")
        else:
            if 'prompt' in item:
                print(f"Prompt:\n{item['prompt']}")
            else:
                print("Prompt: [no 'prompt' field generated]")
        print(f"Target (with Marker, for evaluation): '{item.get('target', '[N/A]')}'")
        print(f"Target (no Marker, ground truth): '{item.get('target_no_marker', '[N/A]')}'")
        print(f"--- Input Seq (ground truth test input): '{item.get('input_seq', '[N/A]')}'")

    # Test Scenario 1: Using Chat Template
    original_use_chat_template = config.USE_CHAT_TEMPLATE
    original_num_shots = config.NUM_SHOTS
    original_seq_len = config.SEQUENCE_LENGTH
    original_pattern_type = config.PATTERN_TYPE

    config.USE_CHAT_TEMPLATE = True
    config.NUM_SHOTS = 1
    config.SEQUENCE_LENGTH = 3
    config.PATTERN_TYPE = "reverse_word"
    
    print(f"\nTesting Few-Shot Word Reverse generation (USE_CHAT_TEMPLATE={config.USE_CHAT_TEMPLATE})")
    test_data_chat = generate_patterned_data(num_examples=1)  # num_examples=1 for a quick test
    for i, item in enumerate(test_data_chat):
        print_test_item(item, i, config.USE_CHAT_TEMPLATE)

    # Test Scenario 2: Using Raw Prompt 
    config.USE_CHAT_TEMPLATE = False  
    # Keep other parameters (NUM_SHOTS, SEQUENCE_LENGTH, PATTERN_TYPE) the same as Scenario 1 for direct comparison
    
    print(f"\nTesting Few-Shot Word Reverse generation (USE_CHAT_TEMPLATE={config.USE_CHAT_TEMPLATE})")
    test_data_raw = generate_patterned_data(num_examples=1)
    for i, item in enumerate(test_data_raw):
        print_test_item(item, i, config.USE_CHAT_TEMPLATE)

    # Restore original config values (in case there are further tests or to avoid global side effects)
    config.USE_CHAT_TEMPLATE = original_use_chat_template
    config.NUM_SHOTS = original_num_shots
    config.SEQUENCE_LENGTH = original_seq_len
    config.PATTERN_TYPE = original_pattern_type
