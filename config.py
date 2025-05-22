import torch
import string

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Chat Mode
USE_CHAT_TEMPLATE = True
SYSTEM_PROMPT    = "You are a helpful assistant."

# Task Configuration
PATTERN_TYPE = "reverse_word"
SEQUENCE_LENGTH = 5
NUM_EXAMPLES = 100

WORD_EXTRACTION_STRICT = False
MIN_WORD_RATIO = 0.8 

NUM_SHOTS = 3

# End Markers
# For Llama 3 Instruct, using <|eot_id|> (End of Turn) is more reliable as the end-of-generation signal
# We also append it to the target sequence to maintain consistency.
END_MARKER = "<|eot_id|>"
EOS_TOKEN_ID = 128009            # <|eot_id|> corresponds to this token ID
# Llama 3 also has <|end_of_text|> (128001), which can be used to stop generation.
# The 'terminators' list in run.py can include both, but target matching will be based on END_MARKER / EOS_TOKEN_ID.
EOS_TOKEN = END_MARKER           # For potential compatibility, set EOS_TOKEN equal to END_MARKER

# Hidden State Extraction Configuration
# 0: After the embedding layer
# N: After the Nth Transformer block
# -1: After the last Transformer block
LAYER_INDICES = [0, 16, -1]
task_max_new_tokens = 80

# Analysis Configuration
ANALYSIS_PARAMS = {
    "pca_n_components": 2, 
    "explained_variance_threshold": None,
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

# Directory Names
# Automatically generate output directory names based on the configuration
shot_suffix = f"_shots{NUM_SHOTS}" if NUM_SHOTS > 0 else "_zeroshot"

# Task is explicitly word-level
task_suffix = f"_L{SEQUENCE_LENGTH}words"
model_dir_name = MODEL_NAME.split('/')[-1]  # Get the base name of the model

# Assemble the final directory name
base_dir_name = f"{PATTERN_TYPE}_stdtok_noiseless_{model_dir_name}{task_suffix}{shot_suffix}"
OUTPUT_DIR = f"results_{base_dir_name}"
FIGURES_DIR = f"figures_{base_dir_name}"